"""Single-GPU baseline for our pre-normalised COCO-17 window annotations.

Target framework: MMAction2 1.2.x. Runtime validation is done on the GPU host.
Do not insert PreNormalize2D: data already uses body-relative coordinates.
"""
from pathlib import Path as _Path

# MMEngine text configs do not expose __file__. This template is substituted
# by Config.fromfile before execution; fileDirname is training/configs.
_project = _Path(r'{{ fileDirname }}').resolve().parents[1]
_data = _project / 'data/processed/stgcn_v1'
default_scope = 'mmaction'
model = dict(
    type='RecognizerGCN',
    backbone=dict(type='STGCN', in_channels=3, num_person=1,
                  graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=10, in_channels=256,
                  dropout=.3, topk=(1,), average_clips='prob'))
train_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='FormatGCNInput', num_person=1, mode='zero'),
    dict(type='PackActionInputs', meta_keys=('frame_dir', 'source_id', 'video_id',
         'dancer', 'view', 'window_start_seconds', 'fps', 'coordinate_space'))]
val_pipeline = train_pipeline
test_pipeline = train_pipeline
train_dataloader = dict(
    batch_size=32, num_workers=2, persistent_workers=True, drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='PoseDataset', ann_file=str(_data / 'aist_train.pkl'),
                 split='train', pipeline=train_pipeline, start_index=0))
val_dataloader = dict(
    batch_size=32, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='PoseDataset', ann_file=str(_data / 'aist_val.pkl'),
                 split='val', pipeline=val_pipeline, start_index=0, test_mode=True))
test_dataloader = dict(
    batch_size=32, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='PoseDataset', ann_file=str(_data / 'aist_test.pkl'),
                 split='test', pipeline=test_pipeline, start_index=0, test_mode=True))
val_evaluator = dict(type='AccMetric', metric_list=('top_k_accuracy', 'mean_class_accuracy'),
                     metric_options=dict(top_k_accuracy=dict(topk=(1,))))
test_evaluator = val_evaluator
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=.001, weight_decay=.0001),
                     clip_grad=dict(max_norm=5, norm_type=2))
param_scheduler = [dict(type='LinearLR', start_factor=.1, by_epoch=True, begin=0, end=5),
                   dict(type='CosineAnnealingLR', eta_min=.00001, T_max=55,
                        begin=5, end=60, by_epoch=True)]
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'), timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3,
                    save_best='acc/mean1', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'))
env_cfg = dict(cudnn_benchmark=False,
               mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
               dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
visualizer = dict(type='ActionVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
randomness = dict(seed=42, deterministic=True)
load_from = None
resume = False
auto_scale_lr = dict(enable=False, base_batch_size=32)
work_dir = str(_project / 'outputs/train_aist_stgcn_v1')
