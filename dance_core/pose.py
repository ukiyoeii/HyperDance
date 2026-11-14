"""COCO-17 temporal processing; NumPy only, shared by train and inference.

Input: pixel coordinates (T,17,3), with final dimension x,y,confidence.
Output windows: body-centred, isotropically scaled coordinates and scores.
Do not apply MMAction2 PreNormalize2D to these already normalised windows.
"""
import numpy as np

BODY = slice(5, 17)


def resample_pose(points, times, fps=30, confidence=.2, max_gap_seconds=.2):
    points = np.asarray(points, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if points.shape != (len(times), 17, 3) or len(times) < 2:
        raise ValueError('Expected (T,17,3) keypoints with at least two timestamps')
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0) or fps <= 0:
        raise ValueError('Timestamps must be finite and strictly increasing; fps > 0')
    if not 0 < confidence <= 1 or max_gap_seconds <= 0:
        raise ValueError('Invalid confidence threshold or maximum gap')
    times = times - times[0]
    grid = np.arange(int(np.floor(times[-1] * fps + 1e-5)) + 1) / fps
    nominal_dt = np.median(np.diff(times))
    output = np.zeros((len(grid), 17, 3), dtype=np.float32)
    observed = np.zeros((len(grid), 17), dtype=bool)
    valid = (np.isfinite(points).all(axis=-1) & (points[..., 2] >= confidence)
             & (points[..., :2] >= 0).all(axis=-1)
             & (points[..., :2] != 0).any(axis=-1))
    for joint in range(17):
        indices = np.flatnonzero(valid[:, joint])
        if not len(indices):
            continue
        t = times[indices]
        p = points[indices, joint]
        right = np.searchsorted(t, grid)
        right = np.clip(right, 0, len(t) - 1)
        left = np.maximum(right - 1, 0)
        nearest = np.where(np.abs(t[right] - grid) <= np.abs(t[left] - grid), right, left)
        exact = np.abs(t[nearest] - grid) < 1e-4
        gaps = t[right] - t[left]
        inside = (grid >= t[0]) & (grid <= t[-1])
        supported = exact | (inside & (gaps > 0) & (gaps <= max_gap_seconds + 1e-8))
        for axis in (0, 1):
            output[:, joint, axis] = np.interp(grid, t, p[:, axis])
        score = np.minimum(p[left, 2], p[right, 2])
        regular = gaps <= nominal_dt * 1.6 + 1e-6
        score[~regular] = np.minimum(score[~regular], .5)
        score[exact] = p[nearest[exact], 2]
        output[:, joint, 2] = np.clip(score, 0, 1)
        output[~supported, joint] = 0
        observed[:, joint] = supported & (exact | regular)
    return output, observed, grid


def window_starts(frame_count, window_frames, stride_frames):
    if window_frames <= 0 or stride_frames <= 0:
        raise ValueError('Window and stride must be positive')
    # Keep full windows only. Never loop/pad a short action or cross a label boundary.
    return range(0, max(frame_count - window_frames + 1, 0), stride_frames)


def normalise_window(points, observed, min_observed=.7, min_supported=.8,
                     min_frame_fraction=.9, max_imputed=.2):
    points = np.asarray(points, dtype=np.float32)
    observed = np.asarray(observed, dtype=bool)
    if points.ndim != 3 or points.shape[1:] != (17, 3) or observed.shape != points.shape[:2]:
        raise ValueError('Invalid window shape')
    if not np.isfinite(points).all():
        raise ValueError('Resampled windows must be finite')
    valid = points[..., 2] > 0
    stats = dict(observed_body_fraction=float(observed[:, BODY].mean()),
                 supported_body_fraction=float(valid[:, BODY].mean()),
                 imputed_body_fraction=float((valid & ~observed)[:, BODY].mean()),
                 supported_frame_fraction=float((valid[:, BODY].sum(axis=1) >= 6).mean()))
    if stats['observed_body_fraction'] < min_observed:
        return None, stats, 'low_observed_body'
    if stats['supported_body_fraction'] < min_supported:
        return None, stats, 'low_supported_body'
    if stats['supported_frame_fraction'] < min_frame_fraction:
        return None, stats, 'too_many_missing_frames'
    if stats['imputed_body_fraction'] > max_imputed:
        return None, stats, 'too_much_interpolation'
    body_xy = points[:, BODY, :2][valid[:, BODY]]
    low, high = np.percentile(body_xy, [1, 99], axis=0)
    scale = float(np.max(high - low)) * .6
    if scale < 12:
        return None, stats, 'body_extent_too_small'
    center = (low + high) / 2
    result = points.copy()
    result[..., :2] = (result[..., :2] - center) / scale
    # Keep upside-down poses and limb aspect ratios. No rotation/axis flipping.
    # Extreme outliers and unsupported joints are zero, including their confidence.
    valid &= (np.abs(result[..., :2]) <= 2).all(axis=-1)
    result[~valid] = 0
    final_body = valid[:, BODY]
    stats['final_body_fraction'] = float(final_body.mean())
    stats['final_frame_fraction'] = float((final_body.sum(axis=1) >= 6).mean())
    if stats['final_body_fraction'] < min_supported or stats['final_frame_fraction'] < min_frame_fraction:
        return None, stats, 'outlier_removal_quality'
    return result, stats, None
