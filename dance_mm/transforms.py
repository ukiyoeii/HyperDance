"""第二版对照实验用的骨架变换。

共同约定（与数据准备一致）：
- 有效点只看置信度掩码 score > 0；坐标已经归一化，负坐标是正常的，不能用坐标判断有效性。
- 无效点的坐标和置信度始终保持为 0。
- 不做再次归一化。
"""
import numpy as np
from mmcv.transforms import BaseTransform

from mmaction.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomIsotropicScale2D(BaseTransform):
    """训练增强：以概率 prob 把整个窗口的 x、y 同乘一个系数（等比例，围绕原点）。

    需要键：keypoint (M, T, V, 2)、keypoint_score (M, T, V)。放在 GenSkeFeat 之前。
    """

    def __init__(self, prob=0.5, scale_range=(0.85, 1.15)):
        assert 0 <= prob <= 1 and 0 < scale_range[0] <= scale_range[1]
        self.prob, self.scale_range = prob, tuple(scale_range)

    def transform(self, results):
        if np.random.rand() >= self.prob:
            return results
        scale = np.random.uniform(*self.scale_range)
        valid = results['keypoint_score'] > 0
        keypoint = results['keypoint'] * np.float32(scale)
        keypoint[~valid] = 0                       # 无效点保持为零
        results['keypoint'] = keypoint.astype(np.float32)
        results['aug_scale'] = float(scale)
        return results

    def __repr__(self):
        return '%s(prob=%s, scale_range=%s)' % (type(self).__name__, self.prob, self.scale_range)


@TRANSFORMS.register_module()
class RandomMonotonicTimeWarp2D(BaseTransform):
    """训练增强：以概率 prob 做轻度、单调的局部时间扰动（只改变局部节奏）。

    固定方案（运行前确定，不根据结果调整）：
    - 输出仍是 T 帧。输出第 t 帧取自源时间 phi(t)，phi 是分段线性函数。
    - phi(0) = 0，phi(T-1) = T-1：两端时间不变，不向窗口外取数据。
    - 在输出时间轴 25%、50%、75% 处设 3 个内部控制点，每个控制点对应的源时间
      在原位置基础上独立平移 U(-max_shift, +max_shift) * (T-1)，默认 max_shift = 0.05。
      相邻控制点间隔 0.25，最大相对平移 0.10，所以 phi 严格递增（不反向、不循环），
      局部播放速度在 0.6 倍到 1.4 倍之间。
    - 源时间落在两帧之间时：该关节相邻两帧都有效才做线性插值，置信度取两者较小值；
      否则不插值，取时间上更近的那一帧（它若无效，输出就是零）。

    需要键：keypoint (M, T, V, 2)、keypoint_score (M, T, V)。放在 GenSkeFeat 之前。
    """

    KNOTS = (0.25, 0.5, 0.75)

    def __init__(self, prob=0.5, max_shift=0.05):
        assert 0 <= prob <= 1
        # 保证严格单调：相邻控制点间隔必须大于两倍最大平移
        assert 0 <= max_shift < min(np.diff((0,) + self.KNOTS + (1,))) / 2
        self.prob, self.max_shift = prob, max_shift

    def sample_map(self, num_frames):
        """返回长度 num_frames 的源时间 phi（单位：帧，浮点数）。"""
        last = num_frames - 1
        out_knots = np.array((0,) + self.KNOTS + (1,), dtype=np.float64)
        shift = np.random.uniform(-self.max_shift, self.max_shift, size=len(self.KNOTS))
        src_knots = out_knots.copy()
        src_knots[1:-1] += shift
        phi = np.interp(np.arange(num_frames) / last, out_knots, src_knots) * last
        phi[0], phi[-1] = 0.0, float(last)
        return np.clip(phi, 0.0, float(last))

    @staticmethod
    def resample(keypoint, score, phi):
        """按源时间 phi 重采样。keypoint (M, T, V, 2)，score (M, T, V)。"""
        last = keypoint.shape[1] - 1
        i0 = np.floor(phi).astype(int)
        i1 = np.minimum(i0 + 1, last)
        w = (phi - i0).astype(np.float32)                       # (T,)
        k0, k1, s0, s1 = keypoint[:, i0], keypoint[:, i1], score[:, i0], score[:, i1]
        wk = w[None, :, None]
        both = (s0 > 0) & (s1 > 0)
        exact = np.broadcast_to(wk == 0, s0.shape)              # 正好落在整数帧上：原样拷贝
        nearest_is_0 = np.broadcast_to(wk < 0.5, s0.shape)
        out_k = np.where(nearest_is_0[..., None], k0, k1)       # 默认：取更近的一帧，不插值
        out_s = np.where(nearest_is_0, s0, s1)
        interp = both & ~exact                                  # 两端都有效才线性插值
        lerp = k0 * (1 - wk[..., None]) + k1 * wk[..., None]
        out_k = np.where(interp[..., None], lerp, out_k)
        out_s = np.where(interp, np.minimum(s0, s1), out_s)     # 置信度保守取较小值
        out_k = np.where(exact[..., None], k0, out_k)
        out_s = np.where(exact, s0, out_s)
        out_k = np.where((out_s > 0)[..., None], out_k, 0)      # 无效点保持为零
        return out_k.astype(np.float32), out_s.astype(np.float32)

    def transform(self, results):
        if np.random.rand() >= self.prob:
            return results
        phi = self.sample_map(results['keypoint'].shape[1])
        results['keypoint'], results['keypoint_score'] = self.resample(
            results['keypoint'], results['keypoint_score'], phi)
        results['aug_time_warp'] = True
        return results

    def __repr__(self):
        return '%s(prob=%s, max_shift=%s, knots=%s)' % (type(self).__name__, self.prob, self.max_shift, self.KNOTS)


@TRANSFORMS.register_module()
class AppendFrameMotion2D(BaseTransform):
    """在 (x, y, score) 后追加相邻帧运动通道 (dx, dy, motion_score)，得到 6 通道。

    - dx[t] = x[t+1] - x[t]，dy 同理；只有该关节在 t 和 t+1 两帧都有效 (score > 0) 才计算。
    - motion_score[t] = min(score[t], score[t+1])。
    - 任一帧无效时三个运动通道全部为 0；最后一帧的运动通道为 0。
    - 没有随机性，训练和验证使用完全相同的生成方式。

    需要键：keypoint (M, T, V, 3)，即放在 GenSkeFeat(feats=['j']) 之后、FormatGCNInput 之前。
    """

    def transform(self, results):
        keypoint = results['keypoint']
        assert keypoint.ndim == 4 and keypoint.shape[-1] == 3, keypoint.shape
        assert 'keypoint_score' not in results, 'AppendFrameMotion2D must come after GenSkeFeat'
        xy, score = keypoint[..., :2], keypoint[..., 2]
        motion = np.zeros(keypoint.shape, dtype=np.float32)
        both = (score[:, :-1] > 0) & (score[:, 1:] > 0)
        motion[:, :-1, :, :2] = np.where(both[..., None], xy[:, 1:] - xy[:, :-1], 0)
        motion[:, :-1, :, 2] = np.where(both, np.minimum(score[:, :-1], score[:, 1:]), 0)
        results['keypoint'] = np.concatenate([keypoint, motion], axis=-1).astype(np.float32)
        return results

    def __repr__(self):
        return '%s()' % type(self).__name__
