"""Local-only RTMPose + exported ST-GCN inference. No training dependencies."""
import hashlib
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'models/breaking_v1'
# Import the exact, hash-verified preprocessing implementation shipped with the model.
import importlib.util


def load_deployed_preprocessing():
    spec = importlib.util.spec_from_file_location('deployed_pose', MODEL_DIR / 'dance_core/pose.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
         (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
POSE_DIR = ROOT / 'models/pose'


def softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    values = np.exp(values - values.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


class ActionModel:
    def __init__(self):
        if not (MODEL_DIR / 'model_metadata.json').is_file():
            raise RuntimeError('缺少模型部署包，请先按 docs/SETUP.md 放置 models/breaking_v1。')
        self.metadata = json.loads((MODEL_DIR / 'model_metadata.json').read_text(encoding='utf-8'))
        self.config = json.loads((MODEL_DIR / 'preprocess.json').read_text(encoding='utf-8'))
        for line in (MODEL_DIR / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
            digest, name = line.split('  ', 1)
            path = (MODEL_DIR / name).resolve()
            if not path.is_relative_to(MODEL_DIR.resolve()):
                raise ValueError('Invalid model manifest path')
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Model bundle file hash mismatch: ' + name)
        self.pose_processing = load_deployed_preprocessing()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(MODEL_DIR / self.metadata['model_file']),
                                            sess_options=options, providers=['CPUExecutionProvider'])
        self.classes = self.metadata['classes']
        actual = self.session.get_inputs()[0]
        if actual.name != 'skeleton' or actual.shape != [1, 1, 1, 60, 17, 3]:
            raise ValueError('Unexpected deployed model input')

    def predict(self, window):
        value = np.asarray(window, dtype=np.float32).reshape(1, 1, 1, 60, 17, 3)
        logits = self.session.run(['logits'], {'skeleton': value})[0]
        if not np.isfinite(logits).all():
            raise ValueError('Non-finite model output')
        return softmax(logits)[0]

    def classify_track(self, track, timestamps, duration, smooth=True):
        fps = self.config['target_fps']
        sampled, observed, grid = self.pose_processing.resample_pose(track, timestamps, fps,
            self.config['confidence_threshold'], self.config['max_gap_seconds'])
        history = deque(maxlen=3 if smooth else 1)
        predictions = []
        for start in self.pose_processing.window_starts(len(grid), 60, 15):
            end_time = (start + 60) / fps
            if end_time > duration + .0001:
                continue
            window, quality, reason = self.pose_processing.normalise_window(sampled[start:start+60],
                                      observed[start:start+60], **self.config['quality'])
            if reason:
                history.clear()
                predictions.append(dict(start=round(start/fps, 4), end=round(end_time, 4),
                    state='insufficient_pose', label=None, scores=None, raw_scores=None, reason=reason))
                continue
            raw = self.predict(window)
            history.append(raw)
            scores = np.mean(history, axis=0)
            order = np.argsort(scores)
            # UI heuristic only, not calibrated certainty or measured accuracy.
            uncertain = scores[order[-1]] < .55 or scores[order[-1]] - scores[order[-2]] < .10
            predictions.append(dict(start=round(start/fps, 4), end=round(end_time, 4),
                state='uncertain' if uncertain else 'predicted', label=self.classes[int(order[-1])],
                scores=np.round(scores, 6).tolist(), raw_scores=np.round(raw, 6).tolist(),
                reason=None))
        return predictions


def bbox_iou(a, b):
    low = np.maximum(a[:2], b[:2])
    high = np.minimum(a[2:4], b[2:4])
    intersection = np.maximum(high - low, 0).prod()
    aa = np.maximum(a[2:4] - a[:2], 0).prod()
    bb = np.maximum(b[2:4] - b[:2], 0).prod()
    return float(intersection / max(aa + bb - intersection, 1))


class PoseEngine:
    def __init__(self):
        from rtmlib import YOLOX, RTMPose
        paths = (POSE_DIR / 'detector.onnx', POSE_DIR / 'rtmpose.onnx')
        if not all(path.is_file() for path in paths):
            raise RuntimeError('姿态模型尚未安装，请运行安装脚本。')
        self.detector = YOLOX(str(paths[0]), model_input_size=(416, 416), backend='onnxruntime', device='cpu')
        self.pose = RTMPose(str(paths[1]), model_input_size=(192, 256), to_openpose=False,
                           backend='onnxruntime', device='cpu')
        # Keep the laptop responsive: bound both sessions' CPU thread pools.
        for tool, path in zip((self.detector, self.pose), paths):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            tool.session = ort.InferenceSession(str(path), sess_options=options,
                                                providers=['CPUExecutionProvider'])

    def infer(self, frame, previous=None, point=None):
        h, w = frame.shape[:2]
        scale = min(1., 960 / max(w, h))
        image = cv2.resize(frame, (round(w*scale), round(h*scale))) if scale < 1 else frame
        boxes = np.asarray(self.detector(image), dtype=float).reshape(-1, 4)
        boxes = boxes[np.all(np.isfinite(boxes), axis=1)]
        if not len(boxes):
            return np.zeros((17, 3), np.float32), previous, 0, False
        normalized = boxes / [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
        selected = None
        if previous is None:
            if point is not None:
                p = np.array(point)
                contains = np.all(normalized[:, :2] <= p, axis=1) & np.all(normalized[:, 2:] >= p, axis=1)
                eligible = np.flatnonzero(contains)
                if not len(eligible):
                    return np.zeros((17, 3), np.float32), None, len(boxes), False
                areas = np.prod(normalized[eligible, 2:] - normalized[eligible, :2], axis=1)
                selected = eligible[int(np.argmin(areas))]
            else:
                selected = int(np.argmax(np.prod(normalized[:, 2:] - normalized[:, :2], axis=1)))
        else:
            overlaps = np.array([bbox_iou(previous, box) for box in normalized])
            selected = int(np.argmax(overlaps))
            center = (normalized[selected, :2] + normalized[selected, 2:]) / 2
            prev_center = (previous[:2] + previous[2:]) / 2
            distance = np.linalg.norm(center - prev_center)
            if overlaps[selected] < .05 and distance > .08:
                # Prefer an explicit gap to silently switching to another person.
                return np.zeros((17, 3), np.float32), previous, len(boxes), False
        keypoints, scores = self.pose(image, bboxes=[boxes[selected].tolist()])
        keypoints, scores = np.asarray(keypoints)[0], np.asarray(scores)[0]
        if keypoints.shape != (17, 2) or scores.shape != (17,):
            raise ValueError('Pose model must output COCO-17')
        output = np.column_stack((keypoints / scale, scores)).astype(np.float32)
        output[~np.isfinite(output).all(axis=1)] = 0
        return output, normalized[selected], len(boxes), True


def analyze_video(path, action_model, pose_engine, point, smooth, progress, cancelled):
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError('视频无法解码或帧率无效。')
    track, timestamps, overlays = [], [], []
    previous, index, missing, multi = None, 0, 0, 0
    start_time = time.perf_counter()
    try:
        while True:
            if cancelled():
                raise InterruptedError('分析已取消。')
            ok, frame = capture.read()
            if not ok:
                break
            points, previous, people, found = pose_engine.infer(frame, previous, point)
            multi += people > 1
            missing += not found
            track.append(points)
            timestamps.append(index / fps)
            if index % 2 == 0:
                shown = points.copy()
                shown[:, 0] /= frame.shape[1]
                shown[:, 1] /= frame.shape[0]
                shown[~np.isfinite(shown)] = 0
                overlays.append(np.round(shown, 5).tolist())
            index += 1
            if index % 10 == 0:
                elapsed = time.perf_counter() - start_time
                progress(min(index/max(count, 1), .99), index, elapsed)
    finally:
        capture.release()
    if index < 2:
        raise ValueError('视频没有足够的可解码帧。')
    duration = index / fps
    predictions = action_model.classify_track(np.stack(track), np.array(timestamps), duration, smooth)
    elapsed = time.perf_counter() - start_time
    return dict(classes=action_model.classes, predictions=predictions, duration=duration,
        overlays=overlays, overlay_fps=fps/2, edges=EDGES, smooth=smooth,
        processed_frames=index, processing_seconds=round(elapsed, 3),
        processing_fps=round(index/max(elapsed, .001), 2),
        missing_person_frames=missing, multi_person_frames=multi,
        model='BRACE ST-GCN 第一版', runtime='CPU',
        score_note='模型分数未经过校准，不等于识别正确率。')
