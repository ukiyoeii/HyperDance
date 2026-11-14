"""Audit the downloaded official skeleton data and build an offline preview.

Run with a Python environment containing NumPy. No training libraries required.
Raw data is read only. Preview coordinates are NOT training features.
"""
import argparse
import csv
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np

GENRES = {
    'gBR': ('Breaking', '霹雳舞'), 'gPO': ('Popping', '震感舞'),
    'gLO': ('Locking', '锁舞'), 'gWA': ('Waacking', '甩手舞'),
    'gMH': ('Middle Hip-hop', 'Middle Hip-hop'),
    'gLH': ('LA-style Hip-hop', 'LA-style Hip-hop'),
    'gHO': ('House', 'House'), 'gKR': ('Krump', 'Krump'),
    'gJS': ('Street Jazz', '街头爵士'), 'gJB': ('Ballet Jazz', '芭蕾爵士'),
}
ACTION_IDS = {'toprock': 0, 'footwork': 1, 'powermove': 2}
BRACE_OFFICIAL_IDS = {'toprock': 0, 'powermove': 1, 'footwork': 2}
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7),
         (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


class NumpyUnpickler(pickle.Unpickler):
    """Only accept the array constructors used in official AIST++ files."""
    def find_class(self, module, name):
        allowed = {('numpy', 'ndarray'), ('numpy', 'dtype'),
                   ('numpy.core.multiarray', '_reconstruct'),
                   ('numpy.core.multiarray', 'scalar')}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError('Unexpected pickle global: %s.%s' % (module, name))
        return super().find_class(module, name)


def load_aist(path):
    with path.open('rb') as handle:
        return NumpyUnpickler(handle).load()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def valid_points(points):
    return (np.isfinite(points).all(axis=-1) & (points[..., 0] > 0)
            & (points[..., 1] > 0) & (points[..., 2] >= .2))


def preview_clip(points, times, title, source, source_id, **extra):
    """Take six seconds at <=15fps; keep original elapsed times and gaps."""
    times = np.asarray(times, dtype=float)
    times = times - times[0]
    requested = np.arange(0, min(float(times[-1]), 6) + .001, 1 / 15)
    indices = np.unique(np.minimum(np.searchsorted(times, requested), len(times) - 1))
    points = np.asarray(points[indices], dtype=float).copy()
    valid = valid_points(points)
    xy = points[..., :2][valid]
    if len(xy) < 17:
        raise ValueError('Too few valid points for preview')
    low, high = np.percentile(xy, [1, 99], axis=0)
    center = (high + low) / 2
    scale = max(float(np.max(high - low)), 1) * 1.15
    points[..., :2] = (points[..., :2] - center) / scale + .5
    points[~valid] = [0, 0, 0]
    return dict(title=title, source=source, source_id=source_id,
                frames=np.round(points, 4).tolist(),
                times=np.round(times[indices], 4).tolist(),
                duration=round(min(float(times[-1]) + 1 / 15, 6.067), 4), **extra)


def audit(root):
    raw = root / 'data/raw'
    output = root / 'outputs/data_check'
    output.mkdir(parents=True, exist_ok=True)
    aist_root = raw / 'aistplusplus'
    ignore_path = aist_root / 'ignore_list.txt'
    ignore = set(ignore_path.read_text(encoding='utf-8').split()) if ignore_path.exists() else set()
    split_zip = root / 'data/downloads/aistplusplus/splits.zip'
    official_all = None
    if split_zip.exists():
        with ZipFile(split_zip) as archive:
            for name in archive.namelist():
                if name.startswith('splits/') and name.endswith('.txt'):
                    destination = aist_root / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(name))
        official_all = set((aist_root / 'splits/all.txt').read_text().split())
    aist_rows, brace_rows, errors, clips = [], [], [], []
    aist_previewed = set()
    for index, path in enumerate(sorted((aist_root / 'keypoints2d').glob('*.pkl'))):
        try:
            data = load_aist(path)
            points = data['keypoints2d']
            if points.ndim != 4 or points.shape[0] != 9 or points.shape[2:] != (17, 3):
                raise ValueError('Unexpected keypoint shape %s' % (points.shape,))
            times = np.asarray(data['timestamps'], dtype=float) / 1e6
            if len(times) != points.shape[1] or len(times) < 2 or not np.all(np.diff(times) > 0):
                raise ValueError('Invalid timestamps')
            parts = path.stem.split('_')
            code = parts[0]
            if code not in GENRES:
                raise ValueError('Unknown genre %s' % code)
            good = valid_points(points)
            excluded = path.stem in ignore
            listed = path.stem in official_all if official_all is not None else None
            aist_rows.append(dict(sequence_id=path.stem, genre_code=code,
                genre=GENRES[code][0], dancer=parts[3], views=points.shape[0],
                frames=points.shape[1], fps=round(1 / np.median(np.diff(times)), 3),
                duration_seconds=round(float(times[-1] - times[0] + np.median(np.diff(times))), 3),
                nonfinite_values=int((~np.isfinite(points)).sum()),
                low_conf_or_invalid_ratio=round(float(1 - good.mean()), 6),
                official_ignored=excluded, official_all_listed=listed,
                path=path.relative_to(root).as_posix()))
            if code not in aist_previewed and not excluded and listed is not False:
                view = int(np.argmax(good.mean(axis=(1, 2))))
                clips.append(preview_clip(points[view], times,
                    GENRES[code][0] + ' / ' + GENRES[code][1], 'AIST++', path.stem,
                    view='c%02d' % (view + 1), fps=60))
                aist_previewed.add(code)
        except Exception as error:
            errors.append(dict(dataset='aistplusplus', path=str(path), error=str(error)))
        if (index + 1) % 250 == 0:
            print('AIST++ checked %d' % (index + 1), flush=True)

    brace_root = raw / 'brace'
    repo = brace_root / 'brace-main'
    with (repo / 'videos_info.csv').open(encoding='utf-8-sig', newline='') as handle:
        video_fps = {r['video_id']: float(r['fps']) for r in csv.DictReader(handle)}
    with (repo / 'annotations/segments.csv').open(encoding='utf-8-sig', newline='') as handle:
        segments = list(csv.DictReader(handle))
    paths = list((brace_root / 'dataset').rglob('*.json'))
    file_map = {}
    for path in paths:
        match = re.fullmatch(r'(.+)_(\d+)-(\d+)_(toprock|footwork|powermove)\.json(?:\.json)?', path.name)
        if not match:
            errors.append(dict(dataset='brace', path=str(path), error='Unrecognised filename'))
            continue
        video, start, end, action = match.groups()
        key = (video, int(start), int(end), action)
        if key in file_map:
            errors.append(dict(dataset='brace', path=str(path), error='Duplicate segment key'))
        file_map[key] = path
    referenced, brace_previewed = set(), set()
    for index, row in enumerate(segments):
        try:
            action = row['dance_type']
            key = (row['video_id'], int(row['start_frame']), int(row['end_frame']), action)
            if int(row['dance_type_id']) != BRACE_OFFICIAL_IDS[action]:
                raise ValueError('Official action ID mismatch')
            path = file_map[key]
            referenced.add(path)
            frames = json.loads(path.read_text(encoding='utf-8'))
            names = sorted(frames, key=lambda name: int(re.search(r'img-(\d+)', name).group(1)))
            numbers = np.array([int(re.search(r'img-(\d+)', name).group(1)) for name in names])
            points = np.array([frames[name]['keypoints'] for name in names], dtype=float)
            if points.shape != (len(names), 17, 3) or len(names) < 2:
                raise ValueError('Unexpected keypoint shape %s' % (points.shape,))
            if not all(name.startswith(row['video_id'] + '/') for name in names):
                raise ValueError('Video ID mismatch')
            if numbers[0] < key[1] or numbers[-1] > key[2] or not np.all(np.diff(numbers) > 0):
                raise ValueError('Frame range mismatch')
            fps = video_fps[row['video_id']]
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError('Invalid video FPS')
            good = valid_points(points)
            times = (numbers - numbers[0]) / fps
            brace_rows.append(dict(segment_id=row['uid'], action=action,
                official_label=int(row['dance_type_id']), training_label=ACTION_IDS[action],
                dancer=row['dancer'], video_id=row['video_id'], year=row['year'],
                start_frame=key[1], end_frame=key[2], frames=len(names), fps=round(fps, 5),
                duration_seconds=round(float(times[-1] + 1 / fps), 3),
                missing_frame_count=int(numbers[-1] - numbers[0] + 1 - len(numbers)),
                nonfinite_values=int((~np.isfinite(points)).sum()),
                low_conf_or_invalid_ratio=round(float(1 - good.mean()), 6),
                all_scores_one=bool(np.all(points[..., 2] == 1)),
                path=path.relative_to(root).as_posix()))
            if action not in brace_previewed and good.mean() >= .95 and times[-1] >= 3:
                clips.append(preview_clip(points, times, action.title(), 'BRACE', row['uid'],
                    view='比赛原机位', fps=round(fps, 3)))
                brace_previewed.add(action)
        except Exception as error:
            errors.append(dict(dataset='brace', path=row.get('uid', ''), error=str(error)))
        if (index + 1) % 400 == 0:
            print('BRACE checked %d' % (index + 1), flush=True)
    for path in set(paths) - referenced:
        errors.append(dict(dataset='brace', path=str(path), error='File not matched to annotations'))

    write_csv(output / 'aist_inventory.csv', aist_rows)
    write_csv(output / 'brace_inventory.csv', brace_rows)
    summary = dict(
        aist=dict(parsed=len(aist_rows), ignored=sum(r['official_ignored'] for r in aist_rows),
            official_filter_available=ignore_path.exists(), official_all_available=official_all is not None,
            official_all_count=len(official_all) if official_all is not None else None,
            missing_official_sequences=sorted((official_all or set()) - {r['sequence_id'] for r in aist_rows}),
            eligible_by_official_lists=sum(not r['official_ignored'] and r['official_all_listed'] is not False for r in aist_rows),
            raw_classes=dict(Counter(r['genre'] for r in aist_rows)),
            eligible_classes=dict(Counter(r['genre'] for r in aist_rows if not r['official_ignored'] and r['official_all_listed'] is not False)),
            files_with_nonfinite=sum(r['nonfinite_values'] > 0 for r in aist_rows)),
        brace=dict(parsed=len(brace_rows), annotated=len(segments), skeleton_files=len(paths),
            classes=dict(Counter(r['action'] for r in brace_rows)),
            files_with_nonfinite=sum(r['nonfinite_values'] > 0 for r in brace_rows),
            files_with_missing_frames=sum(r['missing_frame_count'] > 0 for r in brace_rows),
            fps_values=sorted(set(r['fps'] for r in brace_rows))),
        errors=errors, preview_count=len(clips),
        notes=['Preview labels come from dataset annotations, NOT model predictions.',
               'AIST++ views are not independent performances. Split before cutting windows.',
               'BRACE score=1 is a placeholder, not measured pose confidence.',
               'Official generation splits are reference material, not automatically classification splits.',
               'Preview uses per-clip display scaling; do not train on preview coordinates.'])
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    mapping = dict(genres=[dict(id=i, code=code, name=names[0]) for i, (code, names) in enumerate(GENRES.items())],
                   actions=ACTION_IDS, brace_official_actions=BRACE_OFFICIAL_IDS,
                   skeleton='COCO-17', edges=EDGES)
    (root / 'data/splits/label_map.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding='utf-8')
    template = Path(__file__).with_name('preview_template.html').read_text(encoding='utf-8')
    payload = json.dumps(dict(clips=clips, summary=summary, edges=EDGES), ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    (output / 'skeleton_preview.html').write_text(template.replace('__PREVIEW_DATA__', payload), encoding='utf-8')
    (output / 'preview_samples.json').write_text(json.dumps(dict(clips=clips, edges=EDGES), ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    raise SystemExit(audit(args.root.resolve()))
