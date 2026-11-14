"""Download official RTMPose/YOLOX inference assets into this project only."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'models/pose'
PREFIX = 'https://download.openmmlab.com/mmpose/v1/projects/'
ASSETS = {
    'detector': 'rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip',
    'rtmpose': 'rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
}


def prepare():
    TARGET.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, relative in ASSETS.items():
        target = TARGET / (name + '.onnx')
        archive_path = TARGET / Path(relative).name
        selected_url = None
        if not target.exists():
            errors = []
            for url in (PREFIX + relative, 'https://huggingface.co/Tau-J/RTMPose/resolve/main/' + relative):
                try:
                    print('Downloading ' + url, flush=True)
                    request = Request(url, headers={'User-Agent': 'DanceLocal/1.0'})
                    partial = archive_path.with_suffix('.zip.part')
                    with urlopen(request, timeout=45) as response, partial.open('wb') as output:
                        while block := response.read(1024 * 1024):
                            output.write(block)
                    partial.replace(archive_path)
                    with ZipFile(archive_path) as archive:
                        candidates = [entry for entry in archive.infolist() if entry.filename.endswith('end2end.onnx')]
                        if len(candidates) != 1:
                            raise ValueError('Expected exactly one exported ONNX model')
                        partial_model = target.with_suffix('.onnx.part')
                        with archive.open(candidates[0]) as source, partial_model.open('wb') as output:
                            while block := source.read(1024 * 1024):
                                output.write(block)
                        partial_model.replace(target)
                    selected_url = url
                    break
                except Exception as error:
                    errors.append('%s: %s' % (url, error))
                    print('Download failed, trying official mirror: %s' % error, flush=True)
            if not target.exists():
                raise RuntimeError('\n'.join(errors))
        manifest[name] = dict(file=target.name, bytes=target.stat().st_size,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(), source=selected_url or PREFIX + relative)
        print('%s ready: %.1f MiB' % (name, target.stat().st_size / 2**20), flush=True)
    (TARGET / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    prepare()
