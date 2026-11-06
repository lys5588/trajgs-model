# CMA-VAE 迁入说明

本目录从 `../submodules/Unimotion/cma_vae/` 原样迁入，作为 TrajGS 相机轨迹表示模型的改造基线。来源 checkout：`a3769e9c1acbd28c9a5ce1761cecc778c7863fc6`。

54 个上游受版本控制文件保持内容和相对路径一致。`LICENSE` 从上游仓库根目录附带；本文件是项目新增说明。缓存、权重、数据及外层生成模型未复制。逐文件校验值见 [迁入清单](../docs/cma-vae-import-manifest.json)。

相机任务：文本拍摄指令、暂定 20 个 11D 相机位姿、逐时刻图像及 bbox 区域特征。当前仅完成原版迁入；11D 入口、逐帧 bbox 编码、相机解码头和训练适配尚未实现，详见 [计划](../docs/plan.md)。

原版 `README.md`、配置和文件名为追踪来源而保留。它们仍描述 269D 人体动作任务；其中 `../assets/` 和 `../data_preparation/` 的相对引用来自上游仓库，在当前根目录未迁入，可到 `../submodules/Unimotion/` 查看。不要把原训练入口当作相机训练入口。

原包部分入口使用基于工作目录的绝对导入，原版命令从 `cma_vae/` 下运行。人体数据、HRNet 权重、归一化统计量与训练依赖仍需另行准备；本轮仅验证复制完整性和 Python 语法。
