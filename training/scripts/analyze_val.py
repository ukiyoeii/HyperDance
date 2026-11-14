"""只在验证集上分析最佳检查点（不读取测试集）。

用法（必须通过 tools/gpu_run.sh 启动）：
  python training/scripts/analyze_val.py --task aist --work-dir outputs/formal_v1_xxx/aist --device cuda:0

输出到 <work-dir>/val_analysis/：
  metrics.json            全部数字
  report.md               便于阅读的小结
  curves.png / curves.csv 训练损失、验证总准确率、验证平均类别准确率曲线
  confusion_window.png / .csv   窗口级混淆矩阵
  confusion_source.png / .csv   按原始来源汇总后的混淆矩阵
  window_predictions.csv  每个验证窗口的预测
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 配色：蓝/橙两个类别色 + 单色（蓝）由浅到深的顺序色，用于混淆矩阵
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE_RAMP = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def class_names(task):
    labels = json.loads((ROOT / 'data/processed/stgcn_v1/label_map.json').read_text(encoding='utf-8'))
    if task == 'aist':
        return [g['name'] for g in sorted(labels['genres'], key=lambda g: g['id'])]
    return [k.capitalize() for k, _ in sorted(labels['actions'].items(), key=lambda kv: kv[1])]


def metrics(y, prob, names):
    """y: 真实类别；prob: 每类概率。返回准确率、每类召回/精确率/F1、宏平均 F1、混淆矩阵。"""
    n = len(names)
    pred = prob.argmax(1)
    cm = np.zeros((n, n), dtype=int)           # 行 = 真实类别，列 = 预测类别
    for t, p in zip(y, pred):
        cm[t, p] += 1
    support, predicted, hit = cm.sum(1), cm.sum(0), np.diag(cm)
    recall = np.divide(hit, support, out=np.zeros(n), where=support > 0)
    precision = np.divide(hit, predicted, out=np.zeros(n), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n), where=(precision + recall) > 0)
    confusions = []
    for t in range(n):
        for p in range(n):
            if t != p and cm[t, p] > 0:
                confusions.append(dict(true=names[t], predicted=names[p], count=int(cm[t, p]),
                                       share_of_true_class=round(float(cm[t, p] / support[t]), 4)))
    confusions.sort(key=lambda c: (-c['count'], -c['share_of_true_class']))
    return dict(
        n=int(len(y)), accuracy=float(hit.sum() / len(y)),
        mean_class_accuracy=float(recall.mean()), macro_f1=float(f1.mean()),
        per_class={names[i]: dict(support=int(support[i]), recall=float(recall[i]),
                                  precision=float(precision[i]), f1=float(f1[i])) for i in range(n)},
        confusion_matrix=cm.tolist(), top_confusions=confusions[:8])


def plot_confusion(cm, names, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cm = np.asarray(cm)
    rate = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(names)
    size = max(4.6, 0.72 * n + 2.4)
    fig, ax = plt.subplots(figsize=(size + 0.8, size), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(rate, cmap=LinearSegmentedColormap.from_list('blue_seq', BLUE_RAMP), vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, '%d\n%.0f%%' % (cm[i, j], 100 * rate[i, j]), ha='center', va='center',
                    fontsize=8, color='white' if rate[i, j] > 0.5 else INK)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=40, ha='right', color=INK2, fontsize=9)
    ax.set_yticklabels(names, color=INK2, fontsize=9)
    ax.set_xticks(np.arange(-.5, n), minor=True); ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=2); ax.tick_params(which='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel('Predicted class', color=INK2); ax.set_ylabel('True class', color=INK2)
    ax.set_title(title + '\ncell = count and share of the true class (row)', color=INK, fontsize=10.5, loc='left')
    fig.tight_layout(); fig.savefig(path, dpi=160, facecolor=SURFACE); plt.close(fig)


def curves(work_dir, out, best_epoch):
    """从 MMEngine 的 scalars.json 读取曲线。损失和准确率量纲不同，分成上下两张图，不共用坐标轴。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(str(work_dir / '*/vis_data/scalars.json')))
    if len(files) != 1:
        raise ValueError('Expected exactly one scalars.json, found %d' % len(files))
    loss, lr, val = defaultdict(list), {}, {}
    for line in open(files[0], encoding='utf-8'):
        row = json.loads(line)
        if 'acc/mean1' in row:
            val[int(row['step'])] = (row['acc/top1'], row['acc/mean1'])
        elif 'loss' in row:
            loss[int(row['epoch'])].append(row['loss']); lr[int(row['epoch'])] = row['lr']
    epochs = sorted(val)
    rows = [dict(epoch=e, lr=lr.get(e), train_loss_mean_of_logged=float(np.mean(loss[e])) if loss[e] else None,
                 train_loss_last_logged=loss[e][-1] if loss[e] else None,
                 val_top1=val[e][0], val_mean1=val[e][1]) for e in epochs]
    with open(out / 'curves.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, facecolor=SURFACE)
    tl = [r['train_loss_mean_of_logged'] for r in rows]
    top1, mean1 = [r['val_top1'] for r in rows], [r['val_mean1'] for r in rows]
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.grid(axis='y', color=GRID, linewidth=.8); ax.set_axisbelow(True)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, length=0)
        ax.axvline(best_epoch, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    axes[0].plot(epochs, tl, color=BLUE, linewidth=2)
    axes[0].set_title('Training loss (mean of logged values per epoch)', color=INK, fontsize=10.5, loc='left')
    axes[0].set_ylim(bottom=0)
    axes[0].annotate('best epoch %d' % best_epoch, (best_epoch, axes[0].get_ylim()[1]), xytext=(4, -12),
                     textcoords='offset points', color=INK2, fontsize=8.5)
    axes[1].plot(epochs, top1, color=BLUE, linewidth=2, label='Validation overall accuracy (top-1)')
    axes[1].plot(epochs, mean1, color=ORANGE, linewidth=2, label='Validation mean class accuracy (acc/mean1)')
    axes[1].scatter([best_epoch], [val[best_epoch][1]], s=46, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3)
    axes[1].annotate('%.3f' % val[best_epoch][1], (best_epoch, val[best_epoch][1]), xytext=(6, -14),
                     textcoords='offset points', color=INK, fontsize=8.5)
    axes[1].set_title('Validation accuracy per epoch', color=INK, fontsize=10.5, loc='left')
    axes[1].set_ylim(0, 1); axes[1].set_xlabel('Epoch', color=INK2)
    axes[1].legend(frameon=False, loc='lower right', fontsize=8.5, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(out / 'curves.png', dpi=160, facecolor=SURFACE); plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('aist', 'brace'), required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    out = work_dir / 'val_analysis'
    out.mkdir(exist_ok=True)

    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.runner import load_checkpoint
    from mmaction.registry import DATASETS, MODELS
    from mmaction.utils import register_all_modules
    register_all_modules(init_default_scope=True)

    configs = glob.glob(str(work_dir / '*.py'))
    best = glob.glob(str(work_dir / 'best_acc_mean1_epoch_*.pth'))
    if len(configs) != 1 or len(best) != 1:
        raise ValueError('Need exactly one saved config and one best checkpoint: %s %s' % (configs, best))
    best_epoch = int(Path(best[0]).stem.rsplit('_', 1)[1])
    cfg = Config.fromfile(configs[0])
    # 硬性保护：只允许读取验证集文件
    ann = cfg.val_dataloader.dataset.ann_file
    if os.path.basename(ann) != '%s_val.pkl' % args.task or cfg.val_dataloader.dataset.split != 'val':
        raise ValueError('Refusing to analyse a non-validation file: %s' % ann)
    names = class_names(args.task)
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, best[0], map_location='cpu')
    model.to(args.device).eval()

    probs, labels, sources, ids = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(dataset), 32):
            index = range(start, min(start + 32, len(dataset)))
            for sample, i in zip(model.test_step(pseudo_collate([dataset[i] for i in index])), index):
                info = dataset.get_data_info(i)
                probs.append(sample.pred_score.detach().float().cpu().numpy())
                labels.append(int(info['label'])); sources.append(info['source_id'])
                ids.append(os.path.basename(info['frame_dir']))
    prob, y = np.stack(probs), np.asarray(labels)
    if not np.isfinite(prob).all():
        raise ValueError('Non-finite prediction scores')

    window = metrics(y, prob, names)

    # 按原始来源汇总：同一来源的所有窗口概率取平均，再取最大者
    groups = defaultdict(list)
    for i, s in enumerate(sources):
        groups[s].append(i)
    inconsistent = [s for s, idx in groups.items() if len(set(y[idx].tolist())) != 1]
    if inconsistent:
        raise ValueError('Windows of one source disagree on the label: %s' % inconsistent[:5])
    order = sorted(groups)
    src_prob = np.stack([prob[groups[s]].mean(0) for s in order])
    src_y = np.asarray([y[groups[s][0]] for s in order])
    source = metrics(src_y, src_prob, names)
    per_source = np.asarray([len(groups[s]) for s in order])
    source.update(independent_sources=len(order), label_consistency='all windows of every source share one label',
                  windows_per_source=dict(min=int(per_source.min()), median=float(np.median(per_source)),
                                          max=int(per_source.max())),
                  aggregation='mean of window probabilities per source, then argmax')

    rows = curves(work_dir, out, best_epoch)
    logged = next(r for r in rows if r['epoch'] == best_epoch)
    # 一致性检查：这里重新算出的窗口级指标应与训练日志在最佳轮记录的一致
    consistent = (abs(logged['val_top1'] - window['accuracy']) < 2e-3 and
                  abs(logged['val_mean1'] - window['mean_class_accuracy']) < 2e-3)

    unit = 'original performance (sequence)' if args.task == 'aist' else 'original action segment'
    caveat = ('AIST++: split by dancer, but base choreographies can repeat across dancers.' if args.task == 'aist'
              else 'BRACE: split by whole video; dancers can repeat across splits.')
    result = dict(task=args.task, split='val', checkpoint=best[0], best_epoch=best_epoch,
                  epochs_run=len(rows), classes=names, source_unit=unit, caveat=caveat,
                  window_level=window, source_level=source,
                  logged_at_best_epoch=dict(top1=logged['val_top1'], mean1=logged['val_mean1']),
                  recomputed_matches_training_log=bool(consistent),
                  test_set_used=False)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    for level, m in (('window', window), ('source', source)):
        with open(out / ('confusion_%s.csv' % level), 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle); writer.writerow(['true \\ predicted'] + names)
            for name, row in zip(names, m['confusion_matrix']):
                writer.writerow([name] + row)
        label = '%d windows' % m['n'] if level == 'window' else '%d independent sources (%s)' % (m['n'], unit)
        plot_confusion(m['confusion_matrix'], names,
                       '%s validation, %s level: %s' % (args.task.upper(), level, label),
                       out / ('confusion_%s.png' % level))
    with open(out / 'window_predictions.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_id', 'source_id', 'label', 'prediction'] + ['p_' + n for n in names])
        for i in range(len(y)):
            writer.writerow([ids[i], sources[i], names[y[i]], names[prob[i].argmax()]] + ['%.5f' % v for v in prob[i]])

    lines = ['# %s validation analysis (best checkpoint, epoch %d)' % (args.task.upper(), best_epoch), '',
             'Checkpoint: `%s`' % best[0], '', 'Validation set only; the test set was not opened.', caveat, '',
             '| level | n | accuracy | mean class accuracy | macro F1 |', '|---|---:|---:|---:|---:|']
    for level, m in (('window', window), ('source (%s)' % unit, source)):
        lines.append('| %s | %d | %.4f | %.4f | %.4f |' % (level, m['n'], m['accuracy'], m['mean_class_accuracy'], m['macro_f1']))
    lines += ['', '## Per-class recall', '', '| class | windows | window recall | sources | source recall |', '|---|---:|---:|---:|---:|']
    for n in names:
        w, s = window['per_class'][n], source['per_class'][n]
        lines.append('| %s | %d | %.3f | %d | %.3f |' % (n, w['support'], w['recall'], s['support'], s['recall']))
    lines += ['', '## Main confusions (window level)', '']
    lines += ['- %s -> %s: %d windows (%.1f%% of true %s)' % (c['true'], c['predicted'], c['count'],
              100 * c['share_of_true_class'], c['true']) for c in window['top_confusions']]
    lines += ['', '## Main confusions (source level)', '']
    lines += ['- %s -> %s: %d sources (%.1f%% of true %s)' % (c['true'], c['predicted'], c['count'],
              100 * c['share_of_true_class'], c['true']) for c in source['top_confusions']]
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print('\nrecomputed window metrics match training log:', consistent)
    print('output:', out)


if __name__ == '__main__':
    main()
