"""第一版正式训练配置（brace）。继承 ../stgcn_brace.py，只改下面明确约定的几项。
模型结构、数据处理、数据划分、优化器、60 轮上限、5 轮预热 + 余弦调度、随机种子 42 全部沿用正式基础配置。
"""
_base_ = '../stgcn_brace.py'

# 1) 训练集丢弃最后不足一个批次的样本（避免只有 1 个样本的批次）；验证/测试不丢弃
train_dataloader = dict(batch_size=32, drop_last=True, sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(drop_last=False)
test_dataloader = dict(drop_last=False)   # 仅声明；本轮训练与分析都不会构建测试集加载器

# 2) 每轮保存检查点：保留最近 3 个 + 验证集 acc/mean1 最佳的 1 个
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3,
                                     save_best='acc/mean1', rule='greater'))

# 3) 固定的提前停止规则：验证 acc/mean1 连续 15 轮没有提升至少 0.001 就停止
custom_hooks = [dict(type='EarlyStoppingHook', monitor='acc/mean1', rule='greater',
                     min_delta=0.001, patience=15, strict=True, check_finite=True)]

# 4) 从随机初始化开始，不续接任何权重/优化器/调度器
load_from = None
resume = False
randomness = dict(seed=42, deterministic=True)
