"""第二版自定义变换的运行前检查（CPU，不需要 GPU，不读取测试集）。
合成数组只用于单元测试边界行为；真实数据检查使用训练/验证文件原样内容，不修改、不伪造缺失点。
"""
import pickle, sys, tempfile, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mmengine.config import Config
from mmaction.registry import DATASETS, TRANSFORMS
from mmaction.utils import register_all_modules
register_all_modules(init_default_scope=True)
import dance_mm
from dance_mm.transforms import AppendFrameMotion2D, RandomIsotropicScale2D, RandomMonotonicTimeWarp2D

passed = []
def ok(name, cond, detail=''):
    if not cond:
        raise AssertionError('FAILED: %s %s' % (name, detail))
    passed.append(name); print('PASS', name, detail, flush=True)

DATA = ROOT / 'data/processed/stgcn_v1'
aist = pickle.load(open(DATA / 'aist_train.pkl', 'rb'))['annotations']
brace = pickle.load(open(DATA / 'brace_train.pkl', 'rb'))['annotations']

# ---- 1. 注册 + 配置保存/重载
for n in ('RandomIsotropicScale2D', 'RandomMonotonicTimeWarp2D', 'AppendFrameMotion2D'):
    ok('registered ' + n, TRANSFORMS.get(n) is getattr(dance_mm, n))
cfgs = {}
for key, f in (('A', 'stgcn_aist_v2a_aug.py'), ('B', 'stgcn_brace_v2b_motion.py')):
    c = Config.fromfile(str(ROOT / 'training/configs/v2' / f))
    d = tempfile.mkdtemp(); c.dump(d + '/c.py'); c2 = Config.fromfile(d + '/c.py')
    ok('config %s dump+reload identical' % key, c2.to_dict() == c.to_dict())
    cfgs[key] = c2
v1a = Config.fromfile(str(ROOT / 'training/configs/formal/stgcn_aist_formal_v1.py'))
v1b = Config.fromfile(str(ROOT / 'training/configs/formal/stgcn_brace_formal_v1.py'))
def same_except(a, b, drop):
    a, b = a.to_dict(), b.to_dict()
    for k in drop:
        a.pop(k, None); b.pop(k, None)
    return a == b
A, B = cfgs['A'], cfgs['B']
ok('A: only train pipeline differs from v1',
   same_except(A, v1a, ['train_pipeline', 'train_dataloader', 'custom_imports', 'work_dir'])
   and {k: v for k, v in A.train_dataloader.to_dict().items() if k != 'dataset'} == {k: v for k, v in v1a.train_dataloader.to_dict().items() if k != 'dataset'}
   and {k: v for k, v in A.train_dataloader.dataset.to_dict().items() if k != 'pipeline'} == {k: v for k, v in v1a.train_dataloader.dataset.to_dict().items() if k != 'pipeline'})
ok('A: val pipeline identical to v1 (no augmentation)', A.val_dataloader.to_dict() == v1a.val_dataloader.to_dict(),
   str([p['type'] for p in A.val_dataloader.dataset.pipeline]))
ok('A: train pipeline order', [p['type'] for p in A.train_dataloader.dataset.pipeline] ==
   ['RandomMonotonicTimeWarp2D', 'RandomIsotropicScale2D', 'GenSkeFeat', 'FormatGCNInput', 'PackActionInputs'])
bd, vd = B.to_dict(), v1b.to_dict()
ok('B: in_channels=6 and nothing else in model differs', bd['model']['backbone'].pop('in_channels') == 6 and
   vd['model']['backbone'].pop('in_channels') == 3 and bd['model'] == vd['model'])
ok('B: train and val pipelines identical', B.train_dataloader.dataset.pipeline == B.val_dataloader.dataset.pipeline,
   str([p['type'] for p in B.val_dataloader.dataset.pipeline]))
for key, c, v in (('A', A, v1a), ('B', B, v1b)):
    ok('%s: optimizer/schedule/epochs/hooks/seed/batch/drop_last equal to v1' % key, all(
        c[k].to_dict() == v[k].to_dict() if hasattr(c[k], 'to_dict') else c[k] == v[k]
        for k in ('optim_wrapper', 'param_scheduler', 'train_cfg', 'custom_hooks', 'randomness', 'default_hooks', 'val_evaluator'))
       and c.train_dataloader.batch_size == 32 and c.train_dataloader.drop_last is True and c.load_from is None and c.resume is False)

# ---- 2. 等比例缩放
np.random.seed(0)
T = RandomIsotropicScale2D(prob=1.0, scale_range=(0.85, 1.15))
with_missing = [a for a in aist if (a['keypoint_score'] == 0).any()][:300]
ok('real AIST samples with missing joints available', len(with_missing) == 300)
scales = []
for a in with_missing:
    r = T.transform(dict(keypoint=a['keypoint'].copy(), keypoint_score=a['keypoint_score'].copy()))
    v = a['keypoint_score'] > 0
    assert np.array_equal(r['keypoint_score'], a['keypoint_score'])
    assert (r['keypoint'][~v] == 0).all() and np.isfinite(r['keypoint']).all() and r['keypoint'].dtype == np.float32
    nz = v[..., None] & (np.abs(a['keypoint']) > 1e-3)
    ratio = r['keypoint'][nz] / a['keypoint'][nz]
    assert np.allclose(ratio, r['aug_scale'], rtol=1e-4), 'not isotropic'
    scales.append(r['aug_scale'])
ok('scale: isotropic, score unchanged, invalid stay zero, finite', True, 'scale range seen %.3f..%.3f' % (min(scales), max(scales)))
ok('scale: factors inside [0.85, 1.15]', 0.85 <= min(scales) and max(scales) <= 1.15)
T5 = RandomIsotropicScale2D(prob=0.5)
hits = sum('aug_scale' in T5.transform(dict(keypoint=aist[0]['keypoint'].copy(), keypoint_score=aist[0]['keypoint_score'].copy())) for _ in range(4000))
ok('scale: applied with probability ~0.5', abs(hits / 4000 - 0.5) < 0.03, '%.3f' % (hits / 4000))
neg = aist[0]['keypoint'][aist[0]['keypoint_score'] > 0]
ok('validity is not judged by coordinate sign (negative coords of valid points are kept)', (neg < 0).any())

# ---- 3. 时间扰动
W = RandomMonotonicTimeWarp2D(prob=1.0, max_shift=0.05)
mins, maxs = [], []
for _ in range(5000):
    phi = W.sample_map(120); d = np.diff(phi)
    assert phi[0] == 0 and phi[-1] == 119 and (d > 0).all() and phi.min() >= 0 and phi.max() <= 119
    mins.append(d.min()); maxs.append(d.max())
ok('warp map: phi(0)=0, phi(T-1)=T-1, strictly increasing, inside window (5000 draws)', True,
   'local speed %.3f..%.3f x' % (min(mins), max(maxs)))
ok('warp map: local speed within the pre-registered 0.6..1.4 x', min(mins) >= 0.6 - 1e-6 and max(maxs) <= 1.4 + 1e-6)
# 合成斜坡：x = 帧号。全部有效时输出必须等于 phi(t)（验证时间顺序与插值）
ramp = np.zeros((1, 120, 17, 2), np.float32); ramp[..., 0] = np.arange(120)[None, :, None]; ramp[..., 1] = -1.0
sc = np.full((1, 120, 17), 0.9, np.float32)
phi = W.sample_map(120); k, s = W.resample(ramp, sc, phi)
ok('warp: all-valid ramp reproduces phi exactly, order preserved', np.allclose(k[0, :, 0, 0], phi, atol=1e-3) and (np.diff(k[0, :, 0, 0]) > 0).all()
   and np.allclose(k[..., 1], -1.0) and np.allclose(s, 0.9))
ok('warp: output keeps 120 frames and both end frames', k.shape == ramp.shape and np.array_equal(k[:, 0], ramp[:, 0]) and np.array_equal(k[:, -1], ramp[:, -1]))
# 缺失边界：关节 3 在第 50..69 帧无效；关节 5 全程无效；关节 7 的置信度逐帧不同
sc2 = sc.copy(); r2 = ramp.copy()
sc2[0, 50:70, 3] = 0; r2[0, 50:70, 3] = 0
sc2[0, :, 5] = 0; r2[0, :, 5] = 0
sc2[0, :, 7] = np.linspace(0.3, 1.0, 120)
for _ in range(300):
    phi = W.sample_map(120); k, s = W.resample(r2, sc2, phi)
    i0 = np.floor(phi).astype(int); i1 = np.minimum(i0 + 1, 119); w = phi - i0
    assert (k[s == 0] == 0).all() and np.isfinite(k).all() and np.isfinite(s).all()
    assert (s[0, :, 5] == 0).all() and (k[0, :, 5] == 0).all()                       # 全无效关节保持全零
    for t in range(120):
        a_ok, b_ok = sc2[0, i0[t], 3] > 0, sc2[0, i1[t], 3] > 0
        x = k[0, t, 3, 0]
        if a_ok and b_ok:
            assert abs(x - phi[t]) < 1e-3                                           # 两端有效：插值
        else:
            near = i0[t] if w[t] < 0.5 else i1[t]
            if sc2[0, near, 3] > 0:
                assert x == near and s[0, t, 3] == sc2[0, near, 3]                  # 只拷贝更近的有效帧，不与零插值
            else:
                assert x == 0 and s[0, t, 3] == 0                                   # 更近的帧无效：输出零
        lo, hi = sorted((sc2[0, i0[t], 7], sc2[0, i1[t], 7]))
        assert s[0, t, 7] <= hi + 1e-6 and (w[t] == 0 or abs(s[0, t, 7] - lo) < 1e-6)  # 置信度保守取较小值
ok('warp: gap boundary never interpolates with zeros; all-invalid joint stays zero; score = min of neighbours (300 draws)', True)
for a in with_missing:
    k, s = W.resample(a['keypoint'], a['keypoint_score'], W.sample_map(120))
    assert k.shape == a['keypoint'].shape and s.shape == a['keypoint_score'].shape and k.dtype == np.float32
    assert (k[s == 0] == 0).all() and np.isfinite(k).all() and (s >= 0).all() and (s <= 1).all()
    assert np.array_equal(k[:, 0], a['keypoint'][:, 0]) and np.array_equal(k[:, -1], a['keypoint'][:, -1])
    assert np.abs(k).max() <= np.abs(a['keypoint']).max() + 1e-5                    # 不外推
ok('warp on 300 real AIST windows with missing joints: zero mask, finite, ends kept, no extrapolation', True)

# ---- 4. 运动特征
M = AppendFrameMotion2D()
def motion_check(xy, score):
    kp3 = np.concatenate([xy, score[..., None]], -1)
    out = M.transform(dict(keypoint=kp3.copy()))['keypoint']
    assert out.shape == xy.shape[:3] + (6,) and out.dtype == np.float32 and np.isfinite(out).all()
    assert np.array_equal(out[..., :3], kp3)                                        # 原三通道不变
    assert (out[:, -1, :, 3:] == 0).all()                                           # 最后一帧运动为零
    both = (score[:, :-1] > 0) & (score[:, 1:] > 0)
    assert (out[:, :-1, :, 3:][~both] == 0).all()                                   # 任一帧无效：三个运动通道为零
    assert np.allclose(out[:, :-1, :, 3:5][both], (xy[:, 1:] - xy[:, :-1])[both])
    assert np.allclose(out[:, :-1, :, 5][both], np.minimum(score[:, :-1], score[:, 1:])[both])
    return out
out = motion_check(r2[:, :60], sc2[:, :60])
ok('motion synthetic: dx=+1 on valid ramp, zeros across the gap boundary and on all-invalid joint',
   np.allclose(out[0, :49, 3, 3], 1) and (out[0, 49:70, 3, 3:] == 0).all() and (out[0, :, 5] == 0).all()
   and np.allclose(out[0, :59, 0, 3], 1) and (out[0, :59, 0, 4] == 0).all())
for a in brace[:400] + [dict(keypoint=x['keypoint'][:, :60], keypoint_score=x['keypoint_score'][:, :60]) for x in with_missing]:
    motion_check(a['keypoint'], a['keypoint_score'])
ok('motion on 400 real BRACE windows + 300 real AIST windows with missing joints', True)

# ---- 5. 通过真实配置构建数据集，检查实际输入形状
np.random.seed(42)
dsA = DATASETS.build(A.train_dataloader.dataset); dsAv = DATASETS.build(A.val_dataloader.dataset)
changed = 0
for i in range(0, 400, 2):
    x = dsA[i]['inputs'].numpy(); raw = np.concatenate([aist[i]['keypoint'][0], aist[i]['keypoint_score'][0][..., None]], -1)
    assert x.shape == (1, 1, 120, 17, 3) and np.isfinite(x).all() and (x[0, 0, :, :, :2][x[0, 0, :, :, 2] == 0] == 0).all()
    changed += int(not np.array_equal(x[0, 0], raw))
ok('A train dataset: shape [1,1,120,17,3], finite, zero mask; augmentation active', 0.6 < changed / 200 < 0.9, 'changed %.2f (expected ~0.75)' % (changed / 200))
aval = pickle.load(open(DATA / 'aist_val.pkl', 'rb'))['annotations']
ok('A val dataset: input equals raw file exactly (no augmentation)', all(
    np.array_equal(dsAv[i]['inputs'].numpy()[0, 0], np.concatenate([aval[i]['keypoint'][0], aval[i]['keypoint_score'][0][..., None]], -1)) for i in range(0, 1274, 13)))
dsB = DATASETS.build(B.train_dataloader.dataset); dsBv = DATASETS.build(B.val_dataloader.dataset)
xb = dsB[0]['inputs'].numpy(); xv = dsBv[0]['inputs'].numpy()
ok('B train/val dataset: shape [1,1,60,17,6], finite', xb.shape == (1, 1, 60, 17, 6) and xv.shape == (1, 1, 60, 17, 6) and np.isfinite(xb).all())
ok('B: first three channels equal raw (x, y, score); motion channels match definition',
   np.array_equal(xb[0, 0, :, :, :3], np.concatenate([brace[0]['keypoint'][0], brace[0]['keypoint_score'][0][..., None]], -1))
   and np.allclose(xb[0, 0, :-1, :, 3:5], brace[0]['keypoint'][0, 1:] - brace[0]['keypoint'][0, :-1]) and (xb[0, 0, -1, :, 3:] == 0).all())
print(json.dumps(dict(status='all passed', checks=len(passed)), ensure_ascii=False))
