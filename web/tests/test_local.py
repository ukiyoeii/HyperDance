"""Real local ONNX and HTTP/video-pipeline tests; no Breaking accuracy claims."""
import hashlib
import json
import subprocess
import time
import unittest
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from fastapi.testclient import TestClient

from web.backend import app as backend
from web.backend.inference import ActionModel, MODEL_DIR, PoseEngine, ROOT

# Test media never appear in the user's website history.
backend.STORE = ROOT / 'outputs/local_tests/jobs'
backend.STORE.mkdir(parents=True, exist_ok=True)


class LocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT / 'outputs/local_tests'
        cls.video = cls.directory / 'pipeline_fixture.mp4'
        source = cls.directory / 'official_pose_fixture.jpg'
        if not source.exists():
            raise unittest.SkipTest('Run tools/fetch_test_fixture.py before the optional local integration tests')
        if not (MODEL_DIR / 'model_metadata.json').is_file():
            raise unittest.SkipTest('Install the trained deployment bundle first; see docs/SETUP.md')
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
            '-loop', '1', '-framerate', '30', '-i', str(source), '-t', '3',
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(cls.video)]
        subprocess.run(command, check=True, capture_output=True, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))

    def test_onnx_reproduces_all_export_examples(self):
        model = ActionModel()
        fixture = np.load(MODEL_DIR / 'smoke_inputs_outputs.npz')
        outputs = np.concatenate([model.session.run(['logits'], {'skeleton': x})[0] for x in fixture['skeleton']])
        np.testing.assert_allclose(outputs, fixture['expected_logits'], atol=1e-4, rtol=1e-3)
        self.assertTrue(np.array_equal(outputs.argmax(axis=1), fixture['expected_pred']))

    def test_pose_detects_real_person_and_rejects_blank(self):
        pose = PoseEngine()
        frame = cv2.imread(str(self.directory / 'official_pose_fixture.jpg'))
        points, box, people, found = pose.infer(frame)
        self.assertTrue(found)
        self.assertGreater(people, 0)
        self.assertEqual(points.shape, (17, 3))
        self.assertTrue(np.isfinite(points).all())
        self.assertGreater(np.count_nonzero(points[5:, 2] >= .2), 5)
        blank, _, _, found = pose.infer(np.zeros((480, 640, 3), np.uint8))
        self.assertFalse(found)
        self.assertFalse(blank.any())

    def test_upload_analyze_play_export_and_error_paths(self):
        with TestClient(backend.app) as client:
            self.assertTrue(client.get('/api/health').json()['pose_ready'])
            self.assertEqual(client.get('/').status_code, 200)
            bad = client.post('/api/jobs', files={'file': ('bad.txt', b'hello', 'text/plain')})
            self.assertEqual(bad.status_code, 400)
            self.assertEqual(client.get('/api/jobs/not-a-valid-id').status_code, 404)
            with self.video.open('rb') as handle:
                uploaded = client.post('/api/jobs', files={'file': ('pipeline_fixture.mp4', handle, 'video/mp4')})
            self.assertEqual(uploaded.status_code, 200, uploaded.text)
            job = uploaded.json()
            self.assertEqual(job['state'], 'ready')
            prefix = '/api/jobs/' + job['id']
            self.assertEqual(client.get(prefix+'/result').status_code, 409)
            self.assertEqual(client.get(prefix+'/poster').status_code, 200)
            self.assertEqual(client.get(prefix+'/video', headers={'Range':'bytes=0-1023'}).status_code, 206)
            response = client.post(prefix+'/start', json={'smooth':True})
            self.assertEqual(response.status_code, 200, response.text)
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                current = client.get(prefix).json()
                if current['state'] not in ('running', 'queued'):
                    break
                time.sleep(.15)
            self.assertEqual(current['state'], 'done', current)
            result = client.get(prefix+'/result').json()
            self.assertGreaterEqual(len(result['predictions']), 2)
            self.assertTrue(any(p['scores'] is not None for p in result['predictions']))
            self.assertGreater(len(result['overlays']), 0)
            exported = client.get(prefix+'/export')
            self.assertEqual(exported.status_code, 200)
            self.assertNotIn('overlays', exported.json())
            self.assertIn('attachment', exported.headers['content-disposition'])
            (self.directory/'pipeline_report.json').write_text(json.dumps(dict(
                status='passed', fixture='Official COCO still image repeated as 3s video; not a dance accuracy evaluation',
                processed_frames=result['processed_frames'], processing_seconds=result['processing_seconds'],
                processing_fps=result['processing_fps'], missing_person_frames=result['missing_person_frames'],
                predictions=len(result['predictions'])), indent=2), encoding='utf-8')


if __name__ == '__main__':
    unittest.main(verbosity=2)
