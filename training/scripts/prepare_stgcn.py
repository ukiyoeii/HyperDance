"""Build reproducible, leakage-audited MMAction2 PoseDataset annotation files.

This prepares data, not a trained model. Run after inspect_data.py.
"""
import argparse
import csv
import hashlib
import itertools
import json
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dance_core.pose import normalise_window, resample_pose, window_starts
from inspect_data import load_aist, write_csv

SPLITS = ('train', 'val', 'test')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def aist_assignments(rows, seed, fold):
    rng = random.Random(seed)
    result, dancers_by_split = {}, {s: [] for s in SPLITS}
    for genre in sorted({r['genre'] for r in rows}):
        dancers = sorted({r['dancer'] for r in rows if r['genre'] == genre})
        if len(dancers) != 3:
            raise ValueError('Expected exactly three dancers per AIST++ genre')
        rng.shuffle(dancers)
        for index, split in enumerate(SPLITS):
            dancer = dancers[(index + fold) % 3]
            dancers_by_split[split].append(dancer)
            for row in rows:
                if row['genre'] == genre and row['dancer'] == dancer:
                    result[row['sequence_id']] = split
    return result, dancers_by_split


def brace_assignments(rows, seed):
    # Search fixed-seed video partitions using source segment counts only.
    # No pose values or evaluation predictions are used to choose partitions.
    groups = defaultdict(Counter)
    for row in rows:
        groups[row['video_id']][row['action']] += 1
    videos = sorted(groups)
    nval = max(1, round(len(videos) * .15))
    ntest = nval
    ntrain = len(videos) - nval - ntest
    if ntrain < 1:
        raise ValueError('Not enough source videos')
    classes = sorted({r['action'] for r in rows})
    total = Counter(r['action'] for r in rows)
    ratios = (.7, .15, .15)
    rng = random.Random(seed)
    best = None
    for _ in range(512):
        shuffled = videos.copy()
        rng.shuffle(shuffled)
        chunks = (shuffled[:ntrain], shuffled[ntrain:ntrain+nval], shuffled[ntrain+nval:])
        counts = [sum((groups[v] for v in group), Counter()) for group in chunks]
        if any(c[label] == 0 for c in counts for label in classes):
            continue
        loss = sum((counts[i][label] / total[label] - ratios[i]) ** 2
                   for i in range(3) for label in classes)
        if best is None or loss < best[0]:
            best = (loss, chunks)
    if best is None:
        raise ValueError('Could not produce a partition containing every action')
    assignments = {video: split for split, group in zip(SPLITS, best[1]) for video in group}
    return assignments


def overlap_report(rows, assignments, group_key, keys):
    output = {}
    for key in keys:
        sets = {s: {r[key] for r in rows if assignments[r[group_key]] == s} for s in SPLITS}
        output[key] = {a + '_' + b: sorted(sets[a] & sets[b])
                       for a, b in itertools.combinations(SPLITS, 2)}
    return output


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def prepare(root, config_path):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    label_path = root / 'data/splits/label_map.json'
    labels = json.loads(label_path.read_text(encoding='utf-8'))
    version = config['version']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', version):
        raise ValueError('Version must be a plain directory name')
    destination = root / 'data/processed' / version
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError('Output directory must be empty; use a new version for a new build: %s' % destination)
    report_dir = root / 'outputs' / version
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = root / 'data/splits' / version
    manifest_dir.mkdir(parents=True, exist_ok=True)
    inventory = root / 'outputs/data_check'
    aist = [r for r in read_csv(inventory / 'aist_inventory.csv')
            if r['official_ignored'] == 'False' and r['official_all_listed'] == 'True']
    brace = read_csv(inventory / 'brace_inventory.csv')
    aist_map, dancers = aist_assignments(aist, config['seed'], config['aist']['fold'])
    brace_map = brace_assignments(brace, config['seed'])
    aist_overlaps = overlap_report(aist, aist_map, 'sequence_id', ['sequence_id', 'dancer'])
    brace_overlaps = overlap_report(brace, brace_map, 'video_id', ['segment_id', 'video_id', 'dancer'])
    assert not any(v for overlaps in aist_overlaps.values() for v in overlaps.values())
    assert not any(v for key in ('segment_id', 'video_id') for v in brace_overlaps[key].values())
    source_manifests = {}
    for task, rows, assignments, group_key in (
            ('aist', aist, aist_map, 'sequence_id'), ('brace', brace, brace_map, 'video_id')):
        records = [dict(r, split=assignments[r[group_key]]) for r in rows]
        write_csv(manifest_dir / (task + '_sources.csv'), records)
        source_manifests[task] = records
    protocol = dict(config=config, aist_dancers=dancers,
                    overlaps=dict(aist=aist_overlaps, brace=brace_overlaps),
                    limitations=[
                        'AIST++ is cross-dancer; shared basic choreographies can occur across splits.',
                        'AIST++ uses three training views but only c01 for validation/test.',
                        'BRACE is cross-video, NOT cross-dancer. Dancer overlaps are reported.',
                        'Source splits are fixed before quality filtering or window generation.',
                        'Window accuracy is not a count of independent performances.'])
    (manifest_dir / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    (destination / 'preprocess.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (destination / 'label_map.json').write_text(label_path.read_text(encoding='utf-8'), encoding='utf-8')
    label_ids = {r['code']: r['id'] for r in labels['genres']}
    fps = config['target_fps']
    summary, sample_rows, rejected = {}, [], []
    retained = {}
    for task in ('aist', 'brace'):
        task_config = config[task]
        frames_per_window = round(task_config['window_seconds'] * fps)
        label_names = {r['id']: r['name'] for r in labels['genres']} if task == 'aist' else {
            value: key for key, value in labels['actions'].items()}
        summary[task] = {}
        for split in SPLITS:
            annotations = []
            candidates = [r for r in source_manifests[task] if r['split'] == split]
            used_sources = set()
            dropped = Counter()
            stride = round(task_config['train_stride_seconds' if split == 'train' else 'eval_stride_seconds'] * fps)
            for number, row in enumerate(candidates):
                path = root / row['path']
                source = row['sequence_id'] if task == 'aist' else row['segment_id']
                if task == 'aist':
                    raw = load_aist(path)
                    times = np.asarray(raw['timestamps'], dtype=np.float64) / 1e6
                    views = task_config['train_views' if split == 'train' else 'eval_views']
                    poses = [(view, raw['keypoints2d'][int(view[1:]) - 1]) for view in views]
                    label = label_ids[row['genre_code']]
                    offset = 0.
                    video_id = row['sequence_id']
                else:
                    raw = json.loads(path.read_text(encoding='utf-8'))
                    names = sorted(raw, key=lambda name: int(re.search(r'img-(\d+)', name).group(1)))
                    numbers = np.array([int(re.search(r'img-(\d+)', name).group(1)) for name in names])
                    poses = [('original', np.array([raw[name]['keypoints'] for name in names], dtype=np.float32))]
                    times = (numbers - numbers[0]) / float(row['fps'])
                    offset = numbers[0] / float(row['fps'])
                    label = labels['actions'][row['action']]
                    if label != int(row['training_label']):
                        raise ValueError('Action label map changed after inspection')
                    video_id = row['video_id']
                for view, points in poses:
                    sampled, observed, grid = resample_pose(points, times, fps,
                        config['confidence_threshold'], config['max_gap_seconds'])
                    all_starts = list(window_starts(len(grid), frames_per_window, stride))
                    # A target frame represents a full 1/fps bin. Source last-frame
                    # support can be shorter than that bin after downsampling.
                    duration = float(times[-1] - times[0] + np.median(np.diff(times)))
                    starts = [start for start in all_starts
                              if (start + frames_per_window) / fps <= duration + 1e-4]
                    for start in set(all_starts) - set(starts):
                        dropped['partial_frame_tail'] += 1
                        rejected.append(dict(task=task, split=split, source_id=source,
                            view=view, start_seconds=round(start / fps, 4), reason='partial_frame_tail'))
                    if not all_starts:
                        dropped['source_view_shorter_than_window'] += 1
                        rejected.append(dict(task=task, split=split, source_id=source,
                            view=view, start_seconds=0, reason='source_view_shorter_than_window'))
                    for start in starts:
                        end = start + frames_per_window
                        window, stats, reason = normalise_window(sampled[start:end], observed[start:end], **config['quality'])
                        if reason:
                            dropped[reason] += 1
                            rejected.append(dict(task=task, split=split, source_id=source, view=view,
                                start_seconds=round(start / fps, 4), reason=reason))
                            continue
                        uid = '%s_%s_%s_f%06d' % (task, source, view, start)
                        # Copies ensure pickles own compact arrays, not views of entire sequences.
                        annotation = dict(frame_dir=uid, total_frames=frames_per_window, label=label,
                            keypoint=window[None, ..., :2].copy(),
                            keypoint_score=window[None, ..., 2].copy(),
                            source_id=source, video_id=video_id, dancer=row['dancer'], view=view,
                            window_start_seconds=round(offset + start / fps, 6),
                            fps=fps, coordinate_space='clip_body_bbox_isotropic_v1')
                        annotations.append(annotation)
                        used_sources.add(source)
                        sample_rows.append(dict(sample_id=uid, task=task, split=split, source_id=source,
                            video_id=video_id, dancer=row['dancer'], view=view, label=label,
                            label_name=label_names[label], total_frames=frames_per_window,
                            start_seconds=annotation['window_start_seconds'],
                            end_seconds=round(offset + end / fps, 6),
                            **{key: round(value, 6) for key, value in stats.items()}))
                        if (task, label) not in retained:
                            retained[(task, label)] = (annotation, row)
                if (number + 1) % 100 == 0:
                    print('%s %s: %d/%d sources, %d windows' % (task, split, number + 1, len(candidates), len(annotations)), flush=True)
            counts = Counter(a['label'] for a in annotations)
            if set(counts) != set(label_names):
                raise ValueError('Missing classes after filtering in %s/%s: %s' % (task, split, counts))
            target = destination / ('%s_%s.pkl' % (task, split))
            temporary = target.with_suffix('.pkl.tmp')
            with temporary.open('wb') as handle:
                pickle.dump(dict(split={split: [a['frame_dir'] for a in annotations]}, annotations=annotations),
                            handle, protocol=4)
            temporary.replace(target)
            summary[task][split] = dict(source_count=len(candidates), retained_sources=len(used_sources),
                windows=len(annotations), classes={label_names[k]: counts[k] for k in sorted(counts)},
                dropped=dict(dropped), bytes=target.stat().st_size, sha256=sha256(target))
            print('%s %s saved: %d windows / %.1f MiB' % (task, split, len(annotations), target.stat().st_size / 2**20), flush=True)
            del annotations
    write_csv(manifest_dir / 'windows.csv', sample_rows)
    write_csv(report_dir / 'rejected_windows.csv', rejected)
    # Compare only the post-filtering sets as well as the pre-filtering source partition.
    for task in ('aist', 'brace'):
        task_windows = [r for r in sample_rows if r['task'] == task]
        for key in (('source_id', 'dancer') if task == 'aist' else ('source_id', 'video_id')):
            sets = {s: {r[key] for r in task_windows if r['split'] == s} for s in SPLITS}
            assert all(not sets[a] & sets[b] for a, b in itertools.combinations(SPLITS, 2))
    summary['protocol'] = protocol
    summary['input_hashes'] = {path.relative_to(root).as_posix(): sha256(path) for path in (
        config_path, label_path, inventory / 'aist_inventory.csv', inventory / 'brace_inventory.csv',
        root / 'data/raw/aistplusplus/ignore_list.txt', root / 'data/raw/aistplusplus/splits/all.txt',
        root / 'dance_core/pose.py', Path(__file__))}
    summary['rejected_records'] = len(rejected)
    summary['output_bytes'] = sum(r['bytes'] for task in ('aist', 'brace') for r in summary[task].values())
    summary['status'] = 'prepared_not_trained'
    (report_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    # One normalised training sample per class, for visual QA against the raw preview.
    gallery = []
    genre_names = {row['id']: row['name'] for row in labels['genres']}
    action_names = {value: key.title() for key, value in labels['actions'].items()}
    for (task, label), (annotation, _) in sorted(retained.items()):
        kp, score = annotation['keypoint'][0], annotation['keypoint_score'][0]
        display = np.concatenate([kp / 2 + .5, score[..., None]], axis=-1)
        display[score == 0] = 0
        duration = len(display) / fps
        title = (genre_names if task == 'aist' else action_names)[label]
        gallery.append(dict(title=title, source='AIST++' if task == 'aist' else 'BRACE',
            source_id=annotation['source_id'], view=annotation['view'], fps=fps,
            frames=np.round(display[::2], 5).tolist(), times=(np.arange(0, len(display), 2) / fps).tolist(), duration=duration))
    (report_dir / 'normalised_samples.json').write_text(json.dumps(dict(clips=gallery, edges=labels['edges']), ensure_ascii=False), encoding='utf-8')
    preview_summary = dict(aist=dict(eligible_by_official_lists=summary['aist']['train']['windows']),
                           brace=dict(parsed=summary['brace']['train']['windows']))
    template = Path(__file__).with_name('preview_template.html').read_text(encoding='utf-8')
    template = (template.replace('街舞骨架 · 数据检查', '街舞骨架 · 训练输入检查')
                .replace('STEP 01', 'STEP 02')
                .replace('先认识数据里的舞者', '模型将看到这样的骨架')
                .replace('真实数据集骨架预览', '处理后的训练骨架预览')
                .replace('AIST++ 官方名单内序列', 'AIST++ 训练窗口')
                .replace('BRACE 动作片段', 'BRACE 训练窗口')
                .replace('每段最多预览约 6 秒；画面按整段骨架范围等比例缩放，隐藏无效或低置信度关节点。',
                         '统一为 30 fps；舞种窗口 4 秒、动作窗口 2 秒。已经处理短缺失并等比例缩放，无效关节点为零。')
                .replace('AIST++ 选择一个有效点较多的机位；BRACE 的置信度字段是占位值，不代表真实检测可靠性。',
                         '每类展示一个训练窗口。没有旋转或翻转骨架；BRACE 原始置信度是占位值。'))
    payload = json.dumps(dict(clips=gallery, edges=labels['edges'], summary=preview_summary),
                         ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    (report_dir / 'normalised_preview.html').write_text(template.replace('__PREVIEW_DATA__', payload), encoding='utf-8')
    # A marker appears only after all six files and manifests have succeeded.
    (destination / 'READY.json').write_text(json.dumps(dict(status='prepared_not_trained',
        summary='outputs/%s/summary.json' % version, samples=len(sample_rows)), indent=2), encoding='utf-8')
    print(json.dumps({task: summary[task] for task in ('aist', 'brace')}, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, default=ROOT / 'training/configs/preprocess_v1.json')
    args = parser.parse_args()
    prepare(args.root.resolve(), args.config.resolve())
