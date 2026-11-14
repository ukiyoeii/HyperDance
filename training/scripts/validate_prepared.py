"""Validate every generated sample, label, manifest, split boundary and hash."""
import argparse
import csv
import hashlib
import itertools
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def records(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root, version):
    output = root / 'data/processed' / version
    ready = json.loads((output / 'READY.json').read_text())
    config = json.loads((output / 'preprocess.json').read_text())
    summary = json.loads((root / 'outputs' / version / 'summary.json').read_text(encoding='utf-8'))
    labels = json.loads((output / 'label_map.json').read_text(encoding='utf-8'))
    manifest_dir = root / 'data/splits' / version
    windows = records(manifest_dir / 'windows.csv')
    by_id = {r['sample_id']: r for r in windows}
    require(len(by_id) == len(windows), 'Duplicate sample IDs in manifest')
    require(len(windows) == ready['samples'], 'READY sample count mismatch')
    sources = {}
    for task in ('aist', 'brace'):
        source_key = 'sequence_id' if task == 'aist' else 'segment_id'
        sources[task] = {r[source_key]: r for r in records(manifest_dir / (task + '_sources.csv'))}
    seen, checked, split_sets, result = set(), 0, {}, {}
    for task in ('aist', 'brace'):
        expected_frames = round(config[task]['window_seconds'] * config['target_fps'])
        expected_labels = set(r['id'] for r in labels['genres']) if task == 'aist' else set(labels['actions'].values())
        split_sets[task] = {}
        result[task] = {}
        for split in ('train', 'val', 'test'):
            path = output / ('%s_%s.pkl' % (task, split))
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            require(digest.hexdigest() == summary[task][split]['sha256'], 'File hash mismatch: %s' % path)
            with path.open('rb') as handle:
                data = pickle.load(handle)
            annotations = data['annotations']
            require(len(annotations) == summary[task][split]['windows'], 'Annotation count mismatch')
            require(data['split'][split] == [a['frame_dir'] for a in annotations], 'Split IDs differ from annotations')
            groups = dict(source_id=set(), dancer=set(), video_id=set())
            counts = Counter()
            for annotation in annotations:
                sample_id = annotation['frame_dir']
                require(sample_id not in seen, 'Sample is present in multiple files')
                seen.add(sample_id)
                row = by_id[sample_id]
                require(row['task'] == task and row['split'] == split, 'Manifest split mismatch')
                source = sources[task][annotation['source_id']]
                require(source['split'] == split, 'Window crossed source split')
                xy, score = annotation['keypoint'], annotation['keypoint_score']
                require(xy.shape == (1, expected_frames, 17, 2), 'Invalid coordinate shape')
                require(score.shape == (1, expected_frames, 17), 'Invalid score shape')
                require(xy.dtype == np.float32 and score.dtype == np.float32, 'Unexpected array dtype')
                require(np.isfinite(xy).all() and np.isfinite(score).all(), 'NaN/Inf in training data')
                require((np.abs(xy) <= 2).all() and ((score >= 0) & (score <= 1)).all(), 'Invalid value range')
                require((xy[score == 0] == 0).all(), 'Unsupported joints must have zero coordinates')
                require(annotation['coordinate_space'] == 'clip_body_bbox_isotropic_v1', 'Wrong coordinate convention')
                require(annotation['total_frames'] == expected_frames, 'Wrong total_frames')
                require(annotation['fps'] == config['target_fps'], 'Wrong FPS')
                require(annotation['label'] in expected_labels, 'Invalid class ID')
                require(annotation['label'] == int(row['label']), 'Window/manifest label mismatch')
                if task == 'aist':
                    wanted = next(g['id'] for g in labels['genres'] if g['code'] == source['genre_code'])
                    require(annotation['label'] == wanted, 'AIST++ genre mapping mismatch')
                    lower, upper = 0., float(source['duration_seconds'])
                    wanted_views = config[task]['train_views' if split == 'train' else 'eval_views']
                    require(annotation['view'] in wanted_views, 'Unexpected AIST++ view')
                else:
                    require(annotation['label'] == labels['actions'][source['action']], 'BRACE class mapping mismatch')
                    lower = int(source['start_frame']) / float(source['fps'])
                    upper = (int(source['end_frame']) + 1) / float(source['fps'])
                require(float(row['start_seconds']) >= lower - .002, 'Window starts before source')
                require(float(row['end_seconds']) <= upper + .002, 'Window crosses source end/label boundary')
                require(abs(float(row['end_seconds']) - float(row['start_seconds']) - expected_frames / config['target_fps']) < 1e-4,
                        'Window duration mismatch')
                for key in groups:
                    groups[key].add(annotation[key])
                counts[annotation['label']] += 1
                checked += 1
            require(set(counts) == expected_labels, 'A split is missing one or more classes')
            split_sets[task][split] = groups
            result[task][split] = dict(checked=len(annotations), independent_sources=len(groups['source_id']),
                                       dancers=len(groups['dancer']), videos=len(groups['video_id']))
            print('PASS %s/%s: %d samples' % (task, split, len(annotations)), flush=True)
            del annotations, data
        keys = ('source_id', 'dancer') if task == 'aist' else ('source_id', 'video_id')
        for a, b in itertools.combinations(('train', 'val', 'test'), 2):
            for key in keys:
                require(not split_sets[task][a][key] & split_sets[task][b][key], 'Cross-split leakage: %s/%s' % (task, key))
    require(seen == set(by_id), 'Unmatched window manifest entries')
    result.update(status='passed', checked_samples=checked,
                  framework_runtime_tested=False,
                  notes='Full array/label/split/hash validation; MMAction2 loader/model still needs GPU-host smoke test.')
    (root / 'outputs' / version / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('PASS all %d samples; no source leakage in required protocol groups.' % checked)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--version', default='stgcn_v1')
    args = parser.parse_args()
    validate(args.root.resolve(), args.version)
