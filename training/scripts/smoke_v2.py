"""第二版两个配置的前向/反向检查：真实配置 + 真实训练管道 + 一个批次。不保存模型，不读取测试集。"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(device, out):
    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmaction.registry import DATASETS, MODELS
    from mmaction.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    report = {}
    for name, cfg_file, classes, shape in (('A_aist_aug', 'stgcn_aist_v2a_aug.py', 10, [1, 1, 120, 17, 3]),
                                           ('B_brace_motion', 'stgcn_brace_v2b_motion.py', 3, [1, 1, 60, 17, 6])):
        cfg = Config.fromfile(str(ROOT / 'training/configs/v2' / cfg_file))
        torch.manual_seed(42)
        res = {}
        for split, dcfg in (('train', cfg.train_dataloader.dataset), ('val', cfg.val_dataloader.dataset)):
            ds = DATASETS.build(dcfg)
            idx = torch.linspace(0, len(ds) - 1, 32).long().tolist()
            batch = pseudo_collate([ds[i] for i in idx])
            assert all(list(x.shape) == shape for x in batch['inputs']), batch['inputs'][0].shape
            if split == 'train':
                model = MODELS.build(cfg.model).to(device); model.init_weights(); model.train()
                first = model.backbone.gcn[0].gcn.conv.weight.shape[1]
                assert first == shape[-1], 'first conv in_channels %d != %d' % (first, shape[-1])
                prepared = model.data_preprocessor(batch, training=True)
                losses = model(**prepared, mode='loss')
                loss = sum(v.mean() for k, v in losses.items() if 'loss' in k)
                assert torch.isfinite(loss)
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all().item() for g in grads)
                gnorm = float(torch.sqrt(sum((g ** 2).sum() for g in grads)))
                res.update(train_samples=len(ds), batch_loss=float(loss), grad_norm=gnorm, first_layer_in_channels=int(first))
            else:
                model.eval()
                with torch.inference_mode():
                    preds = model.test_step(batch)
                assert all(tuple(p.pred_score.shape) == (classes,) and torch.isfinite(p.pred_score).all() for p in preds)
                res.update(val_samples=len(ds), predict_shape_ok=True)
        res.update(input_shape=shape, classes=classes, forward_backward='passed')
        report[name] = res
        print('PASS', name, res, flush=True)
        del model
        torch.cuda.empty_cache()
    Path(out).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--device', default='cuda:0'); p.add_argument('--out', required=True)
    a = p.parse_args(); main(a.device, a.out)
