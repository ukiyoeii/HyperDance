"""GPU-host check: actual MMAction2 config, loader, forward and backward.

Run only after installing the compatible training environment on the remote host.
It does not start a full training job or save model checkpoints.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def smoke(device):
    # Must be configured before the first CUDA/cuBLAS operation.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import mmaction
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmaction.registry import DATASETS, MODELS
    from mmaction.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment')
    report = dict(torch=torch.__version__, mmaction=mmaction.__version__, device=device, tasks={})
    for task, num_classes, frames in (('aist', 10, 120), ('brace', 3, 60)):
        cfg = Config.fromfile(str(ROOT / 'training/configs' / ('stgcn_%s.py' % task)))
        dataset = DATASETS.build(cfg.train_dataloader.dataset)
        batch = pseudo_collate([dataset[i] for i in range(min(2, len(dataset)))])
        sample_shape = list(batch['inputs'][0].shape)
        if sample_shape != [1, 1, frames, 17, 3]:
            raise ValueError('Unexpected MMAction2 pipeline output: %s' % sample_shape)
        model = MODELS.build(cfg.model).to(device)
        model.init_weights()
        model.train()
        prepared = model.data_preprocessor(batch, training=True)
        losses = model(**prepared, mode='loss')
        loss_terms = []
        for name, value in losses.items():
            if 'loss' in name:
                loss_terms.append(sum(v.mean() for v in value) if isinstance(value, list) else value.mean())
        if not loss_terms:
            raise ValueError('No training loss returned')
        loss = sum(loss_terms)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise ValueError('Missing or non-finite gradients')
        model.eval()
        with torch.inference_mode():
            predictions = model(**prepared, mode='predict')
        if not all(tuple(p.pred_score.shape) == (num_classes,) for p in predictions):
            raise ValueError('Wrong classifier output dimensions')
        report['tasks'][task] = dict(dataset_samples=len(dataset), sample_shape=sample_shape,
                                     batch_loss=float(loss.detach().cpu()), classes=num_classes,
                                     forward_backward='passed')
        print('PASS %s: %d classes, loss=%f' % (task, num_classes, float(loss.detach().cpu())), flush=True)
        del model, dataset, prepared, batch, gradients, loss, loss_terms, predictions
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    target = ROOT / 'outputs/stgcn_v1/framework_smoke.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    smoke(args.device)
