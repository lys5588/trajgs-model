---
title: TrajGS · 模块架构图册
subtitle: 已用模块、参考模块与接口差异
lang: zh
template: doc
theme: shadcn
mode: light
file: module-architectures
---
CMA-VAE Motion Encoder 输出轨迹高斯后验。DA3 CameraEnc 输出已知相机的条件 token；两者的网络职责不同。

原版分析作为基线保留。相机双编码器与几何转换已独立实现。

## A 已收录的模块

| 架构页 | 内容 | 当前角色 |
| --- | --- | --- |
| [CLS 与 Patch 融合分析](cls-vs-patch-analysis.html) | 空间信息、相机查询、构图几何与计算量 | 候选分析，未修改模型 |
| [DINOv3 图像处理详解](dinov3-image-flow.html) | 像素、patch、CLS 与 768 维特征的关系 | 当前实现详解 |
| [相机 9D/11D 双编码器](camera-encoders.html) | 逐帧 bbox、DINOv3、256/512 通道与验证状态 | 已实现；预训练权重验证待完成 |
| [四个编码模块逐层对照](encoder-layer-comparison.html) | DA3、输入 Linear、Vision-Motion、Motion；含 QKV、FFN、卷积与跳连 | 当前源码详解 |
| [CMA-VAE Motion Encoder](motion-encoder.html) | 完整 forward、时间前端、长跳连、单块、后验头 | 已迁入基线 |
| [相机 9D / 11D 双向转换](camera-pose-codec.html) | 几何表示、坐标约定、往返边界、调用示例 | 已实现工具模块 |
| [DA3 Camera Encoder](da3-camera-encoder.html) | 9D 参数转换、MLP、Transformer、主干注入 | 参考模块 |

每页均给出层级图、张量形状和本地源码入口。
窄屏可滑动图表，或点击图表右上角展开。

## B 两条网络的核心计算路径

```flow TB
动作序列 B × T × 269 -> 线性投影与时间下采样
线性投影与时间下采样 -> 时间位置编码
时间位置编码 -> 9 层 SkipTransformer
9 层 SkipTransformer -> 高斯 μ 与 logvar
高斯 μ 与 logvar -> 轨迹 latent 采样
已知相机内外参 -> 转换为 B × S × 9
转换为 B × S × 9 -> 两层 MLP 与 LayerNorm
两层 MLP 与 LayerNorm -> 4 层 Transformer
4 层 Transformer -> 输出 LayerNorm
输出 LayerNorm -> 相机条件 token
```

CMA-VAE 图中采用迁入配置 r=4。
DA3 图中保留 S 个视图，输出维度依模型版本而定。

## C 直接对比

| 维度 | CMA-VAE Motion Encoder | DA3 CameraEnc |
| --- | --- | --- |
| 编码对象 | 人体动作特征序列 | 已知相机内外参 |
| 参数入口 | 当前 D=269 | 内部转换为 9D |
| 初始投影 | Linear D → 256 | MLP 9 → d/2 → d |
| 序列长度 | 默认下采样 4 倍 | 保留视图数量 |
| 序列位置编码 | 可学习时间位置 | 模块内无时间位置编码 |
| Transformer | 9 层，4＋1＋4 长跳连 | 4 层顺序堆叠 |
| 块内归一化 | Post-LN | Pre-LN |
| 注意力 heads | 4 | 16 |
| 残差 | 块内相加＋跨层拼接投影 | 块内相加＋LayerScale |
| FFN 宽度 | 1024 | 4d |
| padding mask | lengths 转 key padding mask | forward 未暴露 mask |
| 输出 | μ、logvar、采样 z | 确定性条件 token |
| 内部张量布局 | `[L,B,256]` | `[B,S,d]` |
| 图像或 bbox 输入 | 无；在另一条联合编码路径 | 无；图像由主干处理 |

DA3 的“相机编码”不等于“VAE 相机轨迹编码”。
本项目 11D 已定义为中心 3D、旋转前两列 6D 与 FOV 2D。
DA3 原版 CameraEnc 尚未适配该 11D 输入。

## D 相机实现的当前状态

| 接口 | 状态 |
| --- | --- |
| camera_dim=9 或 11 | 已实现，同一类初始化切换 |
| 时间维不下采样，r=1 | 已实现，始终 20 步 |
| 每时刻 bbox 裁剪 | 已实现，直接拉伸到 224×224 |
| 视觉特征 | 冻结 DINOv3 ViT-B/16 CLS，768→256 |
| 联合融合 | 同时刻 256+256→512 |
| Decoder | 保留结构，输出宽度为 9 或 11 |
| 两个 encoder forward/backward | 真实 DINOv3 架构通过；显式随机权重 |
| 预训练权重集成 | 待用户下载至 /mnt/d/model |

本页 B/C 对比的是原版 CMA-VAE 与 DA3。
相机新实现、接口与逐层通道见新增双编码器页面。

## E 图册维护与来源

此目录作为持续扩展的模块架构库。
每个模块保留一份 Markdown 内容源与一份生成 HTML。
新的已用模块或参考模块在这里增加独立页面。

| 文件 | 用途 |
| --- | --- |
| [README](README.md) | 目录约定、重建命令与验证范围 |
| [源码快照](sources.json) | 来源版本、文件路径和 SHA-256 |
| [项目计划](../../plan.md) | 数据、模型、训练安排 |
| [上一轮整体框架对比](../cma-vae-camera-framework/index.html) | 原版 CMA-VAE 与相机目标方案 |

原版分析与相机新实现分开记录。预训练验证状态以相机页面为准。
