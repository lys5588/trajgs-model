# 环境验收记录

验证日期：2026-11-06（Asia/Shanghai）。Linux x86_64，NVIDIA L40S，驱动 595.91.07。

| 项目 | unimotion | trajgs |
| --- | --- | --- |
| Python | 3.10.22 | 3.12.16 |
| PyTorch | 2.5.1+cu124 | 2.15.0+cu132 |
| torchvision | 0.20.1+cu124 | 0.29.1+cu132 |
| CUDA runtime | 12.4 | 13.2 |
| DeepSpeed | 0.15.3 | 0.19.9 |
| Accelerate | 0.23.0 | 1.15.0 |
| PyArrow | 16.1.0 | 16.1.0 |
| 关键依赖及主模型导入 | 通过 | 通过 |
| torchvision C++ NMS 算子 | 通过 | 通过 |
| OpenCV 写入、decord 解码 MP4 | 通过 | 通过 |
| HF datasets / evaluate / nlgmetricverse 隔离导入 | 通过 | 通过 |
| CMA-VAE 动作编码/解码前向 | 通过 | 通过 |
| spaCy en_core_web_sm 3.8.0 加载 | 通过 | 通过 |
| GPU 矩阵运算 | 通过 | 通过 |

原始 `pip check` 在两套环境中均只报告：

```text
decord 0.6.0 is not supported on this platform
```

原因是上游 wheel 的内部标签误写为 CPython 3.6；实际视频解码在 Python 3.10 和 3.12 均通过。
`scripts/check_dependencies.py` 仅允许此条已知元数据异常，其他错误仍失败退出。
没有改写安装包元数据。来源及处理见 [初始化说明](../docs/bootstrap.md)。

运行结果分别保存于 `unimotion.validation.json`、`trajgs.validation.json`。
精确包版本快照为 `*.lock.txt` 与 `*.linux-64.yml`；它们是安装状态快照，未做第二次从零重建测试。

尚未验证完整训练、DeepSpeed JIT 扩展、多卡训练或有图分支；数据和外部模型权重尚未下载。
标准环境若需要 JIT 编译 CUDA 扩展，应另配 CUDA 12.4 Toolkit；本机系统 Toolkit 为 13.2。
