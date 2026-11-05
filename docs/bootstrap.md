# TrajGS 仓库与环境初始化

本项目以 [wangzy01/UniMotion](https://github.com/wangzy01/UniMotion) 为起点，
对应论文 *UniMotion: A Unified Framework for Motion-Text-Vision Understanding and Generation*。
上游基线提交为 `5124e679b9977e110a1982f10554d76b850afe46`。
保留上游 Git 历史、Apache-2.0 LICENSE 和原 README，方便追踪后续改动。

## Git

- `origin`: `https://github.com/lys5588/trajgs-model.git`
- `upstream`: `https://github.com/wangzy01/UniMotion.git`
- 默认推送到 `origin`，`upstream` 的 push URL 设置为 `DISABLED`。
- 仓库局部提交身份：`lys5588 <2448398161@qq.com>`。
- 拉取策略为 fast-forward only，避免意外产生合并提交。

## 两套环境

| 环境 | 用途 | Python | PyTorch | CUDA runtime | torchvision |
| --- | --- | --- | --- | --- | --- |
| `unimotion` | 上游标准配置复现 | 3.10 | 2.5.1 | 12.4 | 0.20.1 |
| `trajgs` | 项目默认开发环境 | 3.12 | 2.15.0 | 13.2 | 0.29.1 |

`unimotion` 使用上游 `requirements.txt` 固定版本，另补齐 CMA-VAE 所需的
`rich`、`psutil`、`tensorboard`、`easydict`。
两套环境均额外固定 `pyarrow==16.1.0`，因为 NLG 评估依赖的
`datasets==2.9.0` 使用新版 PyArrow 已移除的 `PyExtensionType`。
`trajgs` 的独立清单在 `environments/requirements-trajgs.txt`，除 torch/torchvision 外，
将 DeepSpeed 更新为 0.19.9、Accelerate 更新为 1.15.0、Matplotlib 更新为 3.9.4，
以适配新 PyTorch 与 Python 3.12。其他直接依赖尽量保留上游版本。
torchvision 0.29.1 官方 CUDA 13.2 wheel 的元数据要求 `torch>=2.14.0`；
实际兼容性以本机导入及前向检查为准。

Conda 安装在本仓库 `.local/miniforge3/`，环境文件和缓存不提交。
不需要修改系统 Python 或执行全局 `conda init`。

```bash
# 默认使用 trajgs；此命令需要 source，在当前 bash 中生效
source scripts/activate_env.sh
python --version

# 切换到官方复现环境
source scripts/activate_env.sh unimotion

# 在已安装 Conda 的机器上重新创建、安装并检查两套环境
bash scripts/setup_envs.sh
# 也可只处理一套
bash scripts/setup_envs.sh trajgs

# 复用此次安装的精确版本快照（Linux x86_64）
conda env create -f environments/trajgs.linux-64.yml
# 或在已有匹配 Python 的环境中安装 pip 版本快照
DS_BUILD_OPS=0 python -m pip install --no-build-isolation -r environments/trajgs.lock.txt

# 验收；--cuda 额外执行一次小型 CUDA 矩阵运算
python scripts/check_dependencies.py
python scripts/check_environment.py --cuda
```

安装脚本设置 `DS_BUILD_OPS=0`，不在安装时预编译 DeepSpeed 扩展。
标准环境的 CUDA runtime 为 12.4，本机 `/usr/local/cuda` 编译器为 13.2；
若后续使用需要 JIT 编译的 DeepSpeed 优化器，需为标准环境提供匹配的 12.4 Toolkit。
`trajgs` 可使用本机 13.2 Toolkit。runtime 由 PyTorch wheel 的依赖提供，
`nvidia-smi` 中的 CUDA 版本表示驱动支持能力。

主训练/评估从仓库根目录启动，确保本地 `datasets/` 优先解析；
CMA-VAE 独立训练依照其 README 从 `cma_vae/` 启动。
激活脚本默认 `WANDB_MODE=offline`，可在激活前设置其他值。
Torch 扩展和 Triton 缓存分别放在项目 `.cache/` 下，并按环境名隔离。

## 验证范围

`scripts/check_environment.py` 检查关键依赖、主模型和本地数据加载模块导入，
运行一个无需外部权重的 CMA-VAE 动作编码/解码前向，并加载 spaCy 英文模型。
它还验证 torchvision 编译算子、真实 MP4 视频解码和隔离导入 NLG 评估依赖。
上游 `decord==0.6.0` 的文件名为 `py3-none-manylinux2010_x86_64`，内部 WHEEL
却写成 `cp36-cp36m-manylinux2010_x86_64`，因此原始 `pip check` 会报告平台标签问题。
`scripts/check_dependencies.py` 仅允许这一条已验证的元数据异常，其他依赖错误均失败退出；
实际 decord 视频解码由前向验收脚本检查。没有修改 vendor 文件或跳过其他依赖检查。
这不等于完整训练通过。数据、HRNet、CMA-VAE、LLM 等外部资产仍按上游文档准备。
具体安装及验证结果见 `environments/validation.md`。
