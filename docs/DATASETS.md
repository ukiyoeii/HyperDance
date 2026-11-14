# 数据准备

- [AIST++ 官方](https://google.github.io/aistplusplus_dataset/)：二维 COCO 17 点关键点、官方划分及忽略列表。
- [BRACE 官方](https://github.com/markendo/brace)：骨架与动作标注，三类 Toprock、Footwork、Powermove。

遵循各自许可，原始数据不随仓库分发。下载包放 `data/downloads/`，解压数据放 `data/raw/`，生成数据放 `data/processed/stgcn_v1/`，划分清单放 `data/splits/stgcn_v1/`。

先查看 `training/scripts/inspect_data.py --help`、`prepare_stgcn.py --help` 和 `validate_prepared.py --help`，根据解压目录提供路径。处理参数见 `training/configs/preprocess_v1.json`，类别映射见 `data/splits/label_map.json`。

`dance_core/pose.py` 共用预处理包含置信度筛选、短缺失插值、30 fps 重采样、窗口切分、尺度归一化，并保留倒立方向。来源分组在窗口切分前完成。

预处理单元测试：

```bat
python -m unittest discover -s training/scripts -p test_pose_processing.py -v
```
