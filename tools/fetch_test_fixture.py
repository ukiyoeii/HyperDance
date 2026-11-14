"""Download the official MMPose test image for optional integration tests."""
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/coco/000000000785.jpg'

if __name__ == '__main__':
    target = ROOT / 'outputs/local_tests/official_pose_fixture.jpg'
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=30) as response:
        target.write_bytes(response.read())
    print(target)
