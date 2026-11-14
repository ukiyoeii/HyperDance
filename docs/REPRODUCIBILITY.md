# 训练与导出复现

原实验环境：Python 3.10.22、PyTorch 2.1.2+cu121、MMCV 2.1.0、MMEngine 0.10.5、MMAction2 1.2.0、NumPy 1.26.4。依赖与 MMAction2 源码需自行安装。

先准备并校验数据。V1 配置位于 `training/configs/formal/`，V2 位于 `training/configs/v2/`，自定义变换位于 `dance_mm/`。从项目根运行并确保根目录在 PYTHONPATH 中。MMAction2 源码约定放在 `third_party/mmaction2/`。

Linux 示例：

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python third_party/mmaction2/tools/train.py training/configs/formal/stgcn_brace_formal_v1.py --work-dir outputs/reproduction/brace
```

共享服务器可用 `tools/gpu_run.sh`：最多两张，仅允许配置的物理 2、5 号卡。先设置 `GPU_UUID_2` / `GPU_UUID_5`；未设置时拒绝选卡。租约只协调使用该脚本的任务，不能保证全服务器独占。公开版移除了原机器配置，请在目标机器重新验证。

`training/scripts/analyze_val.py` 分析验证集，`export_brace_onnx.py --help` 查看导出参数。部署模型为 V1 BRACE、opset 17，固定 float32 输入 `skeleton [1,1,1,60,17,3]`，输出 `logits [1,3]`，顺序 Toprock、Footwork、Powermove。softmax 在推理侧执行。

导出应保留元数据、预处理参数、文件哈希和数值对比样例。部署包不随仓库分发，模型选择使用验证集，测试集保留到方案确定后评估。
