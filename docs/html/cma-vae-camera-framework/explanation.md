---
title: 从人体动作到相机轨迹
subtitle: CMA-VAE 原版架构与 TrajGS 目标方案
lang: zh
template: doc
theme: shadcn
mode: light
file: cma-vae-camera-framework
---
本轮已迁入原版 CMA-VAE，并定义相机任务接口。目标方案采用 20 个 11D 相机时刻和逐帧 bbox 区域特征。

## A 本轮交付与任务输入

**代码状态：原版迁入完成。相机适配仍是设计方案。**

| 输入 | 目标 batch 形状 | 用途 |
| --- | --- | --- |
| 拍摄任务文本 | B 个字符串 | 后续指令条件生成 |
| 相机位姿特征 | `[B,20,11]` | 连续轨迹表示 |
| 同步 RGB 图像 | `[B,20,3,H,W]` | 提供各时刻参考视图 |
| 2D bbox | `[B,20,4]` | 指定各图像的参考区域 |

20 是时间步数。11D 的分量定义由后续相机模块处理。

## B 原版：人体动作 CMA-VAE

```flow TB
人体动作序列 269D -> 动作投影与时间下采样
动作投影与时间下采样 -> Motion Encoder
Motion Encoder -> 纯动作后验 q_m
动作投影与时间下采样 -> 动作与视觉拼接
单帧图像 -> HRNet 与关节位置采样
HRNet 与关节位置采样 -> 区域编码与关节池化
区域编码与关节池化 -> 单帧特征广播至 L 时刻
单帧特征广播至 L 时刻 -> 动作与视觉拼接
动作与视觉拼接 -> Vision-Motion Encoder
Vision-Motion Encoder -> 联合后验 q_mv
纯动作后验 q_m -> 按样本选择后验并采样
联合后验 q_mv -> 按样本选择后验并采样
按样本选择后验并采样 -> 共享 Decoder
共享 Decoder -> 重建 269D 人体动作
```

两条编码路径均使用位置编码和 SkipTransformer。
有图样本选联合后验；无图样本选纯动作后验。

默认 latent 维度为 256，视觉维度为 128。
默认下采样率为 4，因此 20 个输入时刻产生 5 个 latent 时刻。

依据：[编码器实现](../../../cma_vae/models/architectures/mld_vae_kl_align.py)、[原版配置](../../../cma_vae/configs/config_cma_vae.yaml)。

## C 目标：逐时刻视觉引导的相机 CMA-VAE

**以下为待实现的适配方案。**

```flow TB
相机序列 20 × 11 -> 共享相机投影与时间处理
共享相机投影与时间处理 -> Camera Encoder
Camera Encoder -> 纯相机后验 q_c
共享相机投影与时间处理 -> 同时刻通道拼接
20 张同步图像与 bbox -> 逐帧 bbox 区域裁剪
逐帧 bbox 区域裁剪 -> 视觉编码与区域汇聚
视觉编码与区域汇聚 -> 对齐相机时间分辨率
对齐相机时间分辨率 -> 同时刻通道拼接
同时刻通道拼接 -> Vision-Motion Encoder
Vision-Motion Encoder -> 联合后验 q_cv
纯相机后验 q_c -> 选择后验并采样 latent
联合后验 q_cv -> 选择后验并采样 latent
选择后验并采样 latent --> 相机解码头：下一步定义
```

第 i 个相机特征与第 i 张图像的区域特征配对。
两路沿通道拼接，时间顺序保持一致。

建议首版以 r=1 保留 20 个时刻；该配置尚未实施。
若采用 r=4，视觉分支也须压缩到相同的 5 个时刻。

bbox 暂指矩形区域，不假定像素级分割 mask。
视觉 backbone、区域汇聚方式和特征维度仍待确定。

## D 哪些机制保留，哪些需要替换

| 维度 | 原版基线 | 相机目标方案 |
| --- | --- | --- |
| 序列内容 | 269D 人体动作 | 11D 相机特征 |
| 时间长度 | 可变 T | 暂定 20 |
| 视觉关注区域 | 22 个关节采样点 | 用户提供的 bbox 区域 |
| 视觉时间关系 | 单帧特征广播 | 每时刻独立提取、同步对齐 |
| 双后验机制 | 纯动作与视觉融合后验 | 纯相机与视觉融合后验 |
| 对齐目标 | DPA 后验 KL | 拟保留；逐帧 mask 另行适配 |
| 解码与重建 | 人体特征与关节损失 | 相机解码头及损失待定义 |
| 文本指令 | VAE 无文本路径 | 留给后续条件生成模块 |

原版文件随目录完整保留。人体 loader 和关节工具属于待替换依赖。

## E 双后验怎样共同学习

```flow LR
纯相机后验 q_c -> DPA 损失
联合后验 q_cv -> 停止梯度的目标副本
停止梯度的目标副本 -> DPA 损失
联合后验 q_cv -> 重建与先验 KL
纯相机后验 q_c -> 先验 KL
```

$$
\mathcal{L}_{DPA}=D_{KL}(q_c\,\Vert\,\mathrm{stopgrad}(q_{cv}))
$$

联合编码器仍接受重建和先验 KL 的梯度。
停止梯度仅用于 DPA 的目标副本。

相机重建项待解码头确定。单靠 KL 不足以训练有用的轨迹表示。
原版混合 batch 的 DPA 覆盖整批；相机版需另行定义有效视觉 mask。

## F 从表示学习走向指令生成

```flow LR
拍摄任务文本 -> 条件轨迹生成模块
推理时可用的场景参考图像 -> 条件轨迹生成模块
条件轨迹生成模块 -> 生成轨迹 latent
生成轨迹 latent --> 相机解码头
相机解码头 --> 20 个相机位姿
```

这条生成路径属于后续阶段，当前尚未迁入。
训练时的配对目标视图与推理参考视图需要明确区分。
推理视图如何采样、是否重渲染、数量多少，均待确定。

| 模块 | 当前完成 | 后续工作 |
| --- | --- | --- |
| 数据读取 | 样本字段与对齐约定 | Dataset、bbox crop、mask |
| 模型 | CMA-VAE 原样迁入 | 相机与逐帧视觉适配、解码头 |
| 训练 | 阶段及损失边界 | VAE 预训练、指令条件生成 |

## G 文件与验证

```tree
trajgs-model/
  cma_vae/ | 原版迁入，54 个文件
    models/architectures/ | 双后验与视觉编码
    models/losses/ | DPA 与原重建损失
    models/operator/ | 注意力和位置编码
    models/modeltype/ | 原版训练 wrapper
    data/ | 原版人体数据读取
    configs/ | 原版参数
    IMPORT.md | 迁入说明与适配边界
  docs/
    plan.md | 三模块计划
    cma-vae-import-manifest.json | 文件来源与 SHA-256
    html/cma-vae-camera-framework/ | 本次任务页面
```

迁入文件已逐字节校验。Python 文件已完成语法解析。
当前没有相机训练结果，也未运行模型 forward。

[阅读完整计划](../../plan.md) · [查看迁入说明](../../../cma_vae/IMPORT.md) · [校验清单](../../cma-vae-import-manifest.json)
