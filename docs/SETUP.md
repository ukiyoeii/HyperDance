# Windows 本地安装

使用 Windows 和 Miniconda。脚本默认寻找 `%USERPROFILE%\miniconda3\python.exe`，其他安装位置需调整 `setup_local.bat`。项目环境位于 `env/local-site`，Python 3.11。

## 准备部署包

仓库不分发训练权重。已有本项目 `breaking_local_bundle.zip` 时，将模型文件、元数据、预处理代码、校验样例及脚本放入 `models/breaking_v1/`，保持包内相对结构，确认存在 `model_metadata.json` 和 `verify_local_bundle.py`。没有部署包时，需要按复现说明训练导出；源码本身不含模型。

## 安装启动

运行 `setup_local.bat`：安装锁定依赖、下载官方姿态模型、验证动作部署包。成功后运行 `start_local.bat`，打开 http://127.0.0.1:8000 。保持启动窗口开启。

视频与分析产物保存在本机 `outputs/local_web/`。当前是上传后后台分析，再随播放展示结果；不是摄像头直播系统。默认只监听本机。

端口 8000 占用时先打开上述网址；如已有实例，直接使用。需要重启时先关闭旧启动窗口。

可选集成检查（需要已安装模型）：

```bat
env\local-site\python.exe tools\fetch_test_fixture.py
env\local-site\python.exe -m unittest web.tests.test_local -v
```

检查 ONNX 数值一致性、姿态检测、上传分析管道，不评估 Breaking 实际准确率。
