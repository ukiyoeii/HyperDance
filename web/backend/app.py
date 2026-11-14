"""Loopback-only local video analysis service."""
import asyncio
import json
import logging
import math
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .inference import ActionModel, PoseEngine, ROOT, POSE_DIR, analyze_video

STORE = ROOT / 'outputs/local_web'
STORE.mkdir(parents=True, exist_ok=True)
LOCK = threading.RLock()
JOBS = {}
EVENTS = {}
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dance-video')
ACTION = None
POSE = None
MAX_BYTES = 300 * 1024 * 1024


def save(job):
    with LOCK:
        path = STORE / job['id'] / 'job.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)


@asynccontextmanager
async def lifespan(app):
    global ACTION
    ACTION = await run_in_threadpool(ActionModel)
    for path in STORE.glob('*/job.json'):
        try:
            job = json.loads(path.read_text(encoding='utf-8'))
            if job['state'] in ('queued', 'running'):
                job.update(state='interrupted', message='上次分析因服务关闭而中断，可以重新分析。')
                save(job)
            JOBS[job['id']] = job
        except (ValueError, KeyError, OSError):
            logging.exception('Could not restore job %s', path)
    yield
    for event in EVENTS.values():
        event.set()


app = FastAPI(title='Breaking 本地动作分析', lifespan=lifespan, docs_url=None, redoc_url=None)


def get_job(job_id):
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise HTTPException(404, '任务不存在。')
    with LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, '任务不存在。')
    return job


def probe(path):
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first = capture.read()
        if not ok or not math.isfinite(fps) or fps <= 0 or frames <= 0:
            raise ValueError('无法读取视频，请换用 MP4、MOV 或 WebM 文件。')
        duration = frames / fps
        if duration < 2:
            raise ValueError('视频至少需要 2 秒，建议先用 5–20 秒的单人片段。')
        if duration > 300.1:
            raise ValueError('当前版本每次支持最长 5 分钟的视频。')
        return dict(fps=fps, frames=frames, duration=round(duration, 3),
                    width=first.shape[1], height=first.shape[0])
    finally:
        capture.release()


def prepare_video(directory, source):
    import imageio_ffmpeg
    probe(source)
    preview = directory / 'video.mp4'
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
        '-i', str(source), '-map', '0:v:0', '-map', '0:a:0?',
        '-vf', 'scale=w=trunc(min(1280\\,iw)/2)*2:h=-2,fps=30',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-movflags', '+faststart', str(preview)]
    result = subprocess.run(command, capture_output=True, timeout=300, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        logging.error('ffmpeg: %s', result.stderr.decode(errors='replace'))
        raise ValueError('视频格式转换失败，请换一个视频重试。')
    metadata = probe(preview)
    capture = cv2.VideoCapture(str(preview))
    try:
        ok, first = capture.read()
        if not ok or not cv2.imwrite(str(directory / 'poster.jpg'), first):
            raise ValueError('视频封面生成失败。')
    finally:
        capture.release()
    return metadata


@app.get('/api/health')
def health():
    return dict(ready=ACTION is not None,
        pose_ready=all((POSE_DIR / name).is_file() for name in ('detector.onnx', 'rtmpose.onnx')),
        classes=['Toprock', 'Footwork', 'Powermove'], device='CPU', local_only=True)


@app.get('/api/jobs')
def jobs():
    with LOCK:
        return sorted([dict(j) for j in JOBS.values()], key=lambda j: j['created'], reverse=True)[:30]


@app.post('/api/jobs')
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'}:
        raise HTTPException(400, '请选择 MP4、MOV、MKV、AVI 或 WebM 视频。')
    with LOCK:
        if len(JOBS) >= 100:
            raise HTTPException(400, '本地已有 100 个任务，请先整理 outputs/local_web 后再添加。')
    job_id = uuid.uuid4().hex
    directory = STORE / job_id
    directory.mkdir()
    source = directory / ('original' + suffix)
    size = 0
    try:
        with source.open('wb') as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError('当前版本单个视频最大 300 MB。')
                handle.write(chunk)
        if not size:
            raise ValueError('文件是空的。')
        metadata = await run_in_threadpool(prepare_video, directory, source)
    except (ValueError, subprocess.TimeoutExpired) as error:
        raise HTTPException(400, str(error)) from error
    finally:
        await file.close()
    job = dict(id=job_id, filename=Path(file.filename or 'video').name, created=time.time(),
        state='ready', progress=0., message='视频已准备好，可以开始分析。', metadata=metadata,
        video_url='/api/jobs/%s/video' % job_id, poster_url='/api/jobs/%s/poster' % job_id)
    with LOCK:
        JOBS[job_id] = job
    save(job)
    return job


class StartRequest(BaseModel):
    point: tuple[float, float] | None = None
    smooth: bool = True


def run_job(job_id, point, smooth):
    global POSE
    job = get_job(job_id)
    event = EVENTS[job_id]
    try:
        with LOCK:
            job.update(state='running', message='正在加载姿态模型…', progress=0.)
        save(job)
        if POSE is None:
            POSE = PoseEngine()
        def progress(fraction, frames, elapsed):
            with LOCK:
                job.update(progress=round(fraction, 4), processed_frames=frames,
                           elapsed=round(elapsed, 1), message='正在追踪舞者并提取骨架…')
            save(job)
        result = analyze_video(STORE / job_id / 'video.mp4', ACTION, POSE,
                               point, smooth, progress, event.is_set)
        if event.is_set():
            raise InterruptedError('分析已取消。')
        target = STORE / job_id / 'result.json'
        target.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf-8')
        with LOCK:
            job.update(state='done', progress=1., message='分析完成，播放视频即可查看动作。',
                       summary={k: result[k] for k in ('processing_seconds', 'processing_fps',
                           'missing_person_frames', 'multi_person_frames', 'processed_frames')})
    except InterruptedError:
        with LOCK:
            job.update(state='cancelled', message='分析已取消，可以重新开始。')
    except Exception as error:
        logging.exception('Analysis failed: %s', job_id)
        with LOCK:
            job.update(state='error', message='分析失败：' + str(error))
    finally:
        save(job)


@app.post('/api/jobs/{job_id}/start')
def start(job_id: str, request: StartRequest):
    job = get_job(job_id)
    if request.point is not None and not all(math.isfinite(v) and 0 <= v <= 1 for v in request.point):
        raise HTTPException(400, '人物选择位置无效。')
    if not all((POSE_DIR / name).is_file() for name in ('detector.onnx', 'rtmpose.onnx')):
        raise HTTPException(503, '姿态模型尚未安装，请先运行安装脚本。')
    with LOCK:
        if job['state'] in ('queued', 'running'):
            raise HTTPException(409, '这个视频正在分析中。')
        if sum(j['state'] in ('queued', 'running') for j in JOBS.values()) >= 3:
            raise HTTPException(429, '已有 3 个视频正在排队，请稍后再试。')
        job.update(state='queued', progress=0., message='已加入本机分析队列。')
        EVENTS[job_id] = threading.Event()
        save(job)
        EXECUTOR.submit(run_job, job_id, request.point, request.smooth)
        return dict(job)


@app.post('/api/jobs/{job_id}/cancel')
def cancel(job_id: str):
    job = get_job(job_id)
    if job_id in EVENTS:
        EVENTS[job_id].set()
    return dict(job)


@app.get('/api/jobs/{job_id}')
def status(job_id: str):
    return dict(get_job(job_id))


@app.get('/api/jobs/{job_id}/result')
def result(job_id: str):
    job = get_job(job_id)
    if job['state'] != 'done':
        raise HTTPException(409, '分析还没有完成。')
    return FileResponse(STORE / job_id / 'result.json', media_type='application/json')


@app.get('/api/jobs/{job_id}/video')
def video(job_id: str):
    get_job(job_id)
    return FileResponse(STORE / job_id / 'video.mp4', media_type='video/mp4')


@app.get('/api/jobs/{job_id}/poster')
def poster(job_id: str):
    get_job(job_id)
    return FileResponse(STORE / job_id / 'poster.jpg', media_type='image/jpeg')


@app.get('/api/jobs/{job_id}/export')
def export(job_id: str):
    job = get_job(job_id)
    if job['state'] != 'done':
        raise HTTPException(409, '分析还没有完成。')
    content = json.loads((STORE / job_id / 'result.json').read_text(encoding='utf-8'))
    content.pop('overlays', None)
    target = STORE / job_id / 'predictions.json'
    target.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
    return FileResponse(target, media_type='application/json', filename='breaking_predictions.json')


app.mount('/', StaticFiles(directory=ROOT / 'web/frontend', html=True), name='frontend')
