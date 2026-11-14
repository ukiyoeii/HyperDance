"""项目自定义的 MMAction2 组件（通过配置里的 custom_imports 加载）。"""
from .transforms import AppendFrameMotion2D, RandomIsotropicScale2D, RandomMonotonicTimeWarp2D  # noqa: F401
