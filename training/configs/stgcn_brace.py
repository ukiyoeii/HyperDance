from pathlib import Path as _Path

_base_ = './stgcn_base.py'
_project = _Path(r'{{ fileDirname }}').resolve().parents[1]
_data = _project / 'data/processed/stgcn_v1'
model = dict(cls_head=dict(num_classes=3))
train_dataloader = dict(dataset=dict(ann_file=str(_data / 'brace_train.pkl')))
val_dataloader = dict(dataset=dict(ann_file=str(_data / 'brace_val.pkl')))
test_dataloader = dict(dataset=dict(ann_file=str(_data / 'brace_test.pkl')))
work_dir = str(_project / 'outputs/train_brace_stgcn_v1')
