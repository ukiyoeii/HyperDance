"""把第一版 BRACE ST-GCN（Toprock/Footwork/Powermove）导出为 ONNX，并验证、打包成本地部署包。

可重复执行：每次都会重新生成 exports/breaking_local_bundle/ 和 exports/breaking_local_bundle.zip。
只在 CPU 上运行；只读取验证集 brace_val.pkl，不读取测试集；不训练、不改动任何权重或网络层。

用法（在项目根目录）：
    env/dance-stgcn/bin/python training/scripts/export_brace_onnx.py

输入约定  skeleton  float32 [1,1,1,60,17,3] = batch, clip, person, frame, joint, (x, y, score)
          坐标已按 dance_core/pose.py 归一化；模型内部不再归一化。
输出约定  logits    float32 [1,3]，类别顺序 Toprock, Footwork, Powermove；softmax 由调用方完成。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''          # 强制 CPU，不触碰任何 GPU

import argparse, hashlib, json, pickle, platform, shutil, sys, zipfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CLASSES = ['Toprock', 'Footwork', 'Powermove']
INPUT_NAME, OUTPUT_NAME = 'skeleton', 'logits'
INPUT_SHAPE = (1, 1, 1, 60, 17, 3)
ATOL, RTOL = 1e-4, 1e-3
PER_CLASS = 10


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def compare(got, ref):
    """got 与 ref 的误差；通过标准与 numpy.allclose 相同：|got-ref| <= atol + rtol*|ref|。"""
    got, ref = got.astype(np.float64), ref.astype(np.float64)
    diff = np.abs(got - ref)
    return dict(max_abs_error=float(diff.max()),
                max_rel_error=float((diff / np.maximum(np.abs(ref), 1e-12)).max()),
                within_tolerance=bool((diff <= ATOL + RTOL * np.abs(ref)).all()))


class LogitsWrapper(torch.nn.Module):
    """只是把原模型的"骨干 + 分类头"原样串起来并返回 logits，不新增、不删除任何层。
    调用的是 RecognizerGCN.extract_feat(stage='head')，即训练和验证时走的同一段代码。"""

    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer

    def forward(self, skeleton):
        logits, _ = self.recognizer.extract_feat(skeleton, stage='head')
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-dir', default='outputs/formal_v1_20261114_122723/brace')
    ap.add_argument('--checkpoint', default='best_acc_mean1_epoch_10.pth')
    ap.add_argument('--expected-sha256',
                    default='da9f73a0917a5a308b8edad99ddcfe314e2134c1c13801c4ea225487a802f28c')
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--out', default='exports/breaking_local_bundle')
    args = ap.parse_args()
    assert not torch.cuda.is_available(), '本脚本只允许在 CPU 上运行'
    torch.set_num_threads(8)

    import onnx, onnxruntime as ort
    import mmaction, mmcv, mmengine
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.runner import load_checkpoint
    from mmaction.registry import DATASETS, MODELS
    from mmaction.utils import register_all_modules
    register_all_modules(init_default_scope=True)

    work = ROOT / args.work_dir
    ckpt = work / args.checkpoint
    out = ROOT / args.out
    data_dir = ROOT / 'data/processed/stgcn_v1'

    # ---------- 1. 核对检查点哈希与实际配置 ----------
    ckpt_sha = sha256(ckpt)
    assert ckpt_sha == args.expected_sha256, '检查点 SHA-256 不符: %s' % ckpt_sha
    cfg_files = sorted(work.glob('*.py'))
    assert len(cfg_files) == 1, cfg_files
    cfg = Config.fromfile(str(cfg_files[0]))                 # 训练时 MMEngine 保存的"实际配置"
    m = cfg.model
    pipeline_types = [p.type for p in cfg.val_dataloader.dataset.pipeline]
    assert m.type == 'RecognizerGCN' and m.backbone.type == 'STGCN' and m.cls_head.type == 'GCNHead'
    assert m.backbone.in_channels == 3 and m.backbone.num_person == 1
    assert m.backbone.graph_cfg.layout == 'coco' and m.cls_head.num_classes == 3
    assert pipeline_types == ['GenSkeFeat', 'FormatGCNInput', 'PackActionInputs'], pipeline_types
    assert 'PreNormalize2D' not in pipeline_types
    label_map = json.loads((data_dir / 'label_map.json').read_text())
    assert label_map['actions'] == {'toprock': 0, 'footwork': 1, 'powermove': 2}
    assert label_map['skeleton'] == 'COCO-17'
    preprocess = json.loads((data_dir / 'preprocess.json').read_text())
    assert preprocess['target_fps'] == 30 and preprocess['brace']['window_seconds'] == 2

    model = MODELS.build(cfg.model)
    raw_ckpt = torch.load(str(ckpt), map_location='cpu')
    ckpt_epoch = int(raw_ckpt['meta']['epoch'])
    state_keys = set(raw_ckpt['state_dict'])
    model_keys = set(model.state_dict())
    assert state_keys == model_keys, (sorted(state_keys ^ model_keys))   # 结构与权重一一对应
    del raw_ckpt
    load_checkpoint(model, str(ckpt), map_location='cpu', strict=True)
    model.eval()
    wrapper = LogitsWrapper(model).eval()
    assert not any(mod.training for mod in wrapper.modules())
    print('[1] 检查点与配置核对通过  sha256=%s  epoch=%d' % (ckpt_sha, ckpt_epoch), flush=True)

    # ---------- 2. 导出 ----------
    if out.exists():
        shutil.rmtree(out)
    (out / 'scripts').mkdir(parents=True)
    onnx_path = out / 'breaking_stgcn.onnx'
    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(wrapper, (dummy,), str(onnx_path), opset_version=args.opset,
                          input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                          do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL,
                          dynamic_axes=None)
    om = onnx.load(str(onnx_path))
    onnx.checker.check_model(om, full_check=True)
    opsets = {(o.domain or 'ai.onnx'): int(o.version) for o in om.opset_import}
    op_types = sorted({n.op_type for n in om.graph.node})

    def dims(v):
        return [int(d.dim_value) for d in v.type.tensor_type.shape.dim]
    assert [i.name for i in om.graph.input] == [INPUT_NAME] and dims(om.graph.input[0]) == list(INPUT_SHAPE)
    assert [o.name for o in om.graph.output] == [OUTPUT_NAME] and dims(om.graph.output[0]) == [1, 3]
    assert om.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert om.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert 'Softmax' not in op_types, '输出必须是 logits'
    n_init = int(sum(np.prod(i.dims) for i in om.graph.initializer))
    print('[2] 导出并通过 ONNX checker  opset=%s  节点类型=%s' % (opsets, op_types), flush=True)

    # ---------- 3. 验证：验证集每类固定 10 个窗口 ----------
    dcfg = cfg.val_dataloader.dataset
    assert os.path.basename(dcfg.ann_file) == 'brace_val.pkl' and dcfg.split == 'val'
    dcfg.ann_file = str(data_dir / 'brace_val.pkl')
    assert Path(dcfg.ann_file).name == 'brace_val.pkl'           # 只读验证集
    dataset = DATASETS.build(dcfg)
    raw = pickle.load(open(dcfg.ann_file, 'rb'))['annotations']
    assert len(raw) == len(dataset)
    labels_all = np.array([a['label'] for a in raw])
    picked = []
    for c in range(3):                                           # 固定规则：该类窗口按文件顺序等间隔取 10 个
        idx = np.flatnonzero(labels_all == c)
        picked += [int(idx[j]) for j in np.round(np.linspace(0, len(idx) - 1, PER_CLASS)).astype(int)]
    assert len(set(picked)) == 3 * PER_CLASS

    inputs, pipe_diff = [], 0.
    for i in picked:
        a = raw[i]
        x = np.concatenate([a['keypoint'][0], a['keypoint_score'][0][..., None]], -1).astype(np.float32)
        assert x.shape == (60, 17, 3)
        ds_x = dataset[i]['inputs'].numpy()                      # 训练/验证管道实际送进模型的张量 [1,1,60,17,3]
        pipe_diff = max(pipe_diff, float(np.abs(ds_x[0, 0] - x).max()))
        inputs.append(x[None, None, None])
    inputs = np.stack(inputs)                                    # [30,1,1,1,60,17,3]
    assert pipe_diff == 0., pipe_diff
    labels = labels_all[picked]
    sample_ids = np.array([raw[i]['frame_dir'] for i in picked])

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(onnx_path), so, providers=['CPUExecutionProvider'])
    onnx_logits = np.concatenate([sess.run([OUTPUT_NAME], {INPUT_NAME: x})[0] for x in inputs])
    with torch.no_grad():
        torch_logits = np.concatenate([wrapper(torch.from_numpy(x)).numpy() for x in inputs])
        # 另用 MMAction2 官方验证路径（数据管道 + test_step）得到概率，确认导出语义与训练时的评估一致
        official_prob = np.stack([p.pred_score.float().numpy() for p in
                                  model.test_step(pseudo_collate([dataset[i] for i in picked]))])
    assert onnx_logits.shape == torch_logits.shape == (3 * PER_CLASS, 3)
    onnx_prob, torch_prob = softmax(onnx_logits), softmax(torch_logits)
    onnx_pred, torch_pred, off_pred = onnx_logits.argmax(1), torch_logits.argmax(1), official_prob.argmax(1)

    c_logits, c_prob = compare(onnx_logits, torch_logits), compare(onnx_prob, torch_prob)
    c_off = compare(onnx_prob, official_prob)
    agree = float((onnx_pred == torch_pred).mean())
    agree_off = float((onnx_pred == off_pred).mean())
    passed = bool(c_logits['within_tolerance'] and c_prob['within_tolerance'] and c_off['within_tolerance']
                  and agree == 1.0 and agree_off == 1.0)
    validation = dict(
        passed=passed, onnx_checker='passed (full_check=True)',
        tolerance=dict(atol=ATOL, rtol=RTOL, rule='|onnx - pytorch| <= atol + rtol * |pytorch|'),
        samples=dict(source='brace_val.pkl（验证集）', per_class=PER_CLASS, total=len(picked),
                     rule='每类窗口按文件顺序等间隔取 10 个', val_indices=picked,
                     sample_ids=sample_ids.tolist(), labels=labels.tolist()),
        input_equals_training_pipeline_max_abs_diff=pipe_diff,
        logits_onnx_vs_pytorch=c_logits,
        softmax_onnx_vs_pytorch=c_prob,
        softmax_onnx_vs_mmaction_test_step=c_off,
        class_agreement_onnx_vs_pytorch=agree,
        class_agreement_onnx_vs_mmaction_test_step=agree_off,
        info_only=dict(note='以下仅为这 30 个验证窗口上的数字，不是模型的正式指标',
                       onnx_correct=int((onnx_pred == labels).sum()),
                       pytorch_correct=int((torch_pred == labels).sum()),
                       min_top1_top2_logit_gap=float(np.min(np.sort(torch_logits, 1)[:, -1]
                                                            - np.sort(torch_logits, 1)[:, -2]))),
        runtime=dict(onnxruntime_provider='CPUExecutionProvider', device='cpu'))
    (out / 'export_validation.json').write_text(json.dumps(validation, indent=2, ensure_ascii=False))
    np.savez_compressed(out / 'smoke_inputs_outputs.npz',
                        skeleton=inputs.astype(np.float32),             # [30,1,1,1,60,17,3]，逐个送入模型
                        expected_logits=torch_logits.astype(np.float32),  # 原 PyTorch 模型的输出
                        expected_probs=torch_prob.astype(np.float32),
                        expected_pred=torch_pred.astype(np.int64),
                        onnx_logits_at_export=onnx_logits.astype(np.float32),
                        labels=labels.astype(np.int64), sample_ids=sample_ids,
                        class_names=np.array(CLASSES))
    print('[3] 验证 %s  logits %s  softmax %s  类别一致率 %.4f'
          % ('通过' if passed else '未通过', c_logits, c_prob, agree), flush=True)

    # ---------- 4. 元数据与打包 ----------
    versions = dict(python=platform.python_version(), torch=torch.__version__, onnx=onnx.__version__,
                    onnxruntime=ort.__version__, numpy=np.__version__, mmaction2=mmaction.__version__,
                    mmengine=mmengine.__version__, mmcv=mmcv.__version__)
    metadata = dict(
        model_name='breaking_stgcn', model_file='breaking_stgcn.onnx', version='formal_v1',
        task='Breaking 动作三分类（BRACE）',
        input=dict(name=INPUT_NAME, dtype='float32', shape=list(INPUT_SHAPE),
                   axes=['batch', 'clip', 'person', 'frame', 'joint', 'channel'],
                   channels=['x', 'y', 'score'], dynamic_axes=False),
        output=dict(name=OUTPUT_NAME, dtype='float32', shape=[1, 3], kind='logits',
                    note='未做 softmax；由本地端对最后一维做 softmax 得到概率'),
        classes=CLASSES, class_to_index={c: i for i, c in enumerate(CLASSES)},
        fps=30, window_seconds=2, window_frames=60,
        skeleton=dict(layout='COCO-17', joints=[
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'], num_person=1),
        normalization=dict(
            already_normalized_input_required=True, model_applies_normalization=False,
            coordinate_space='clip_body_bbox_isotropic_v1',
            implementation='dance_core/pose.py: resample_pose -> window_starts -> normalise_window',
            parameters_file='preprocess.json',
            summary=['按真实时间重采样到 30 fps；置信度低于 0.2 的点视为缺失；只在间隔不超过 0.2 秒时插值',
                     '每个 60 帧窗口：取 12 个身体点（肩肘腕髋膝踝）坐标的 1%~99% 分位范围，'
                     '中心 = 范围中点，尺度 = 最大边长 × 0.6，(xy - 中心) / 尺度，x 和 y 用同一尺度',
                     '缺失点和归一化后绝对值超过 2 的点：x、y、score 全部置零',
                     '不旋转、不翻转，倒立姿态保持原样；不要再使用 PreNormalize2D',
                     'normalise_window 返回 None 的窗口质量不合格，不应送入模型']),
        source_checkpoint=dict(file=args.work_dir + '/' + args.checkpoint, sha256=ckpt_sha, epoch=ckpt_epoch,
                               selected_by='验证集 acc/mean1 最佳', seed=42),
        model_config=cfg.model.to_dict(),
        export=dict(opset=opsets, ir_version=int(om.ir_version), producer='%s %s' % (om.producer_name, om.producer_version),
                    exporter='torch.onnx.export (TorchScript)', mode='eval', device='cpu',
                    do_constant_folding=True, op_types=op_types, num_weight_values=n_init,
                    onnx_sha256=sha256(onnx_path), tool_versions=versions,
                    script='scripts/export_brace_onnx.py'),
        validation_summary=dict(passed=passed, max_abs_error_logits=c_logits['max_abs_error'],
                                max_abs_error_softmax=c_prob['max_abs_error'], class_agreement=agree,
                                details='export_validation.json'),
        caveats=['单一随机种子的第一版模型，验证集结果为初步结果；测试集未使用',
                 '验证集窗口级 Footwork 与 Powermove 之间混淆较多，网站端建议对同一段的多个窗口概率取平均'])
    (out / 'model_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    shutil.copy(data_dir / 'label_map.json', out / 'label_map.json')
    shutil.copy(data_dir / 'preprocess.json', out / 'preprocess.json')
    (out / 'dance_core').mkdir()
    for name in ('__init__.py', 'pose.py'):
        shutil.copy(ROOT / 'dance_core' / name, out / 'dance_core' / name)
    shutil.copy(Path(__file__), out / 'scripts' / 'export_brace_onnx.py')
    shutil.copy(Path(__file__).with_name('verify_local_bundle.py'), out / 'verify_local_bundle.py')
    shutil.copy(Path(__file__).with_name('breaking_local_bundle_README.md'), out / 'README.md')

    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt')
    (out / 'SHA256SUMS.txt').write_text(''.join('%s  %s\n' % (sha256(p), p.relative_to(out).as_posix()) for p in files))
    zip_path = out.with_suffix('.zip')
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(p for p in out.rglob('*') if p.is_file()):
            z.write(p, (Path(out.name) / p.relative_to(out)).as_posix())
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
    print('[4] 打包完成 %s  %d 字节  sha256=%s' % (zip_path.relative_to(ROOT), zip_path.stat().st_size, sha256(zip_path)))
    print('RESULT', 'PASSED' if passed else 'FAILED')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
