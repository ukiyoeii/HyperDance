"""第二版实验 A：AIST++ 训练增强配方（等比例缩放 + 轻度单调时间扰动）。
继承第一版正式配置；模型、4 秒窗口、验证管道、优化器、调度、batch_size、drop_last、
60 轮上限、提前停止、种子 42 全部不变。只改训练管道。
"""
_base_ = '../formal/stgcn_aist_formal_v1.py'
custom_imports = dict(imports=['dance_mm'], allow_failed_imports=False)

train_pipeline = [
    dict(type='RandomMonotonicTimeWarp2D', prob=0.5, max_shift=0.05),
    dict(type='RandomIsotropicScale2D', prob=0.5, scale_range=(0.85, 1.15)),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='FormatGCNInput', num_person=1, mode='zero'),
    dict(type='PackActionInputs', meta_keys=('frame_dir', 'source_id', 'video_id',
         'dancer', 'view', 'window_start_seconds', 'fps', 'coordinate_space'))]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
# 验证/测试管道沿用基础配置（无增强），这里不做任何改动
