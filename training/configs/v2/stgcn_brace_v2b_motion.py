"""第二版实验 B：BRACE 显式运动特征（x, y, score, dx, dy, motion_score 共 6 通道）。
继承第一版正式配置；2 秒窗口、数据、模型主体、优化器、调度、batch_size、drop_last、
60 轮上限、提前停止、种子 42 全部不变。只改输入通道数和（训练=验证）特征生成。
"""
_base_ = '../formal/stgcn_brace_formal_v1.py'
custom_imports = dict(imports=['dance_mm'], allow_failed_imports=False)

model = dict(backbone=dict(in_channels=6))
train_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='AppendFrameMotion2D'),
    dict(type='FormatGCNInput', num_person=1, mode='zero'),
    dict(type='PackActionInputs', meta_keys=('frame_dir', 'source_id', 'video_id',
         'dancer', 'view', 'window_start_seconds', 'fps', 'coordinate_space'))]
val_pipeline = train_pipeline
test_pipeline = train_pipeline
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))   # 仅声明，本轮不构建测试集
