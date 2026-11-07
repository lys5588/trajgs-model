# TrajGS：相机轨迹生成计划

## 问题与当前状态

目标是根据拍摄任务文本与场景参考图像生成相机轨迹。先实现相机版 CMA-VAE 表示，再接入指令条件生成模型。

一条轨迹固定为 20 个时间步。使用同一 CameraCMAVAE 类，通过 camera_dim=9 或 11 初始化输入与输出维度；默认 11。20 是时间序列长度，不是 FPS。

- 已实现：9D/11D 双向几何 codec、纯相机编码器、逐帧 bbox + DINOv3 联合编码器、独立配置和验证入口。
- 已验证：29 项数值/接口测试；9D、11D 的真实 DINOv3 架构在 L40S 上通过 forward/backward 和 Decoder 形状检查。
- 待完成：预训练 DINOv3 权重集成验证。官方直链返回 403，官方 HF 仓库需要授权；用户负责下载到 /mnt/d/model。当前真实架构测试使用显式随机权重，不代表预训练验证通过。
- 后续：真实数据 Dataset、相机损失、DPA、训练循环和文本条件生成。

原版 CMA-VAE 的 54 个迁入文件保持原样。新相机模型独立于人体 loader、HRNet 和人体训练 loss。

## 1. 数据读取

### 1.1 模型入口

| 字段 | 形状 / 类型 | 约定 |
| --- | --- | --- |
| camera_features | [B,20,D] 浮点 Tensor | D=camera_dim，仅支持 9 或 11；有限数值 |
| images | [B,20,3,H,W] | RGB；uint8 或 [0,1] 浮点 |
| bboxes | [B,20,4] | 原图像素 xyxy；每时刻一个目标框 |
| has_images | [B] bool，可选 | 样本级完整视觉可用性 |
| text | 后续数据字段 | 本阶段不输入 CMA-VAE |

有图样本必须提供完整 20 张图和 20 个有效框；不接受逐帧缺失或逐帧 visual_mask。混合 batch 使用 full-batch 图像与 bbox 张量，has_images=False 行可放占位值，不进入视觉处理。省略 has_images 时，同时提供 images/bboxes 表示整批有图；两者都不提供表示整批无图。

本版不支持变长轨迹、时间 padding 或 lengths。数据层后续负责同步采样到 20 帧。真实 Dataset 尚未实现，测试使用合成相机和图像。

### 1.2 相机表示

| D | 向量布局 |
| --- | --- |
| 9 | C2W 相机中心 3 + XYZW 四元数 4 + 垂直/水平 FOV 2 |
| 11 | C2W 相机中心 3 + 旋转前两列 6 + 垂直/水平 FOV 2 |

复用 camera codec。其输入/输出外参默认 W2C，表示内部为 C2W；11D 逆构造使用列向量 Gram-Schmidt。只编码 fx/fy 的 FOV，逆构造默认中心主点、零 skew。

场景参考系与尺度归一化仍由后续数据流程决定。图像 bbox 裁剪不改变相机特征中的原图 FOV；本版 bbox 坐标仅作裁剪，不额外输入网络。

### 1.3 bbox 图像流程

每帧 bbox 左上角 floor、右下角 ceil，并限制到原图范围。非有限坐标、反向/零面积框或裁剪后空区域报错。裁剪后双线性拉伸为 224×224，antialias=True、align_corners=False，不补边、不中心裁剪。

RGB 转 FP32 [0,1]，使用 mean=(0.485,0.456,0.406)、std=(0.229,0.224,0.225) 标准化。按 batch/time 顺序展开为 [B×20,3,224,224]，分块运行骨干，再恢复时间排列。

## 2. 模型

### 2.1 固定通道

| 层 / 配置 | Camera Encoder | Vision-Camera Encoder |
| --- | --- | --- |
| 共享相机前端 | Linear D→256 + learned PE | 同一组权重，使用同一时刻相机特征 |
| 视觉路径 | 无 | DINOv3 CLS 768→Linear 256 |
| 主干输入 | [20,B,256] | 同时刻 concat 后 [20,B,512] |
| SkipTransformer | 9 层，4+1+4 | 9 层，4+1+4 |
| 注意力 | 4 heads，每头 64 | 4 heads，每头 128 |
| FFN | 256→1024→256 | 512→2048→512 |
| 长跳连 Linear | 512→256 | 1024→512 |
| 归一化 / 激活 | Post-LN / GELU | Post-LN / GELU |
| Dropout | 0.1 | 0.1 |
| 后验 Linear | 256→512 | 512→512 |
| mu / logvar / z | 各 [20,B,256] | 各 [20,B,256] |

共享时间位置编码为 [500,1,256]，取前 20 个位置，仅加在相机特征上。视觉投影为单层带 bias 的 Linear(768,256)，不增加激活。联合路径不做 cross-attention、不沿时间拼成 40 帧、不广播首帧。

两个 Transformer 和后验头权重独立。时间下采样和上采样均为 Identity。Normal 后验以 FP32 构造；train/eval 都使用 rsample。

### 2.2 冻结 DINOv3

使用官方 ViT-B/16，取 forward_features 返回的 x_norm_clstoken，宽度 768。骨干参数 requires_grad=False，始终 eval，并在 no_grad 中提取特征；后续视觉投影保留梯度。

源码子模块固定到 6876159a11b4df116f30f667f8c9888617df0751。采用官方本地 PyTorch Hub 加载，无需升级现有 Transformers。默认每次处理 20 张裁剪图，分块只影响显存，不改变帧顺序。

默认预训练权重路径：

```text
/mnt/d/model/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

首个视觉调用才加载骨干，纯相机调用无需权重。缺少源码/权重时明确报错，不隐式下载、不随机回退。独立验证命令提供 --random_backbone，结果明确标记为架构测试。

### 2.3 接口与 Decoder 边界

| 方法 | 返回 / 行为 |
| --- | --- |
| encode_camera(camera_features) | (z_camera, dist_camera)，[20,B,256] |
| encode_joint(camera_features, images, bboxes) | 完整视觉 batch 的 (z_joint, dist_joint) |
| forward(camera_features, images=None, bboxes=None, has_images=None) | 只编码，返回下述字典 |
| decode(z) | [20,B,256] → [B,20,D] 原始预测向量 |

forward 字典中的 latent_camera/dist_camera 覆盖整个 B；latent_joint/dist_joint 仅覆盖有图子 batch，顺序由 joint_indices 指明，无图时为 None。latent 按原 batch 顺序选取联合/纯相机结果；has_images 返回样本级 bool。该选择在 train/eval 中一致，不自动调用 decode。

Decoder 沿用原 all_encoder 的 9 层 SkipTransformer、256 通道、1024 FFN 和 4 heads；末层仍是 Linear，只将输出宽度按 camera_dim 初始化为 9 或 11。此阶段不引入四元数归一化、Gram-Schmidt、FOV 约束或几何损失。输出形状正确不代表输出已是有效相机。

模型、视觉前处理分别位于：
- [camera_cma_vae.py](../cma_vae/models/architectures/camera_cma_vae.py)
- [camera_vision.py](../cma_vae/models/architectures/camera_vision.py)

## 3. 训练与验收

### 3.1 当前验证

使用独立 [配置](../cma_vae/configs/config_camera_vae.yaml)，不调用人体训练入口。启动示例从仓库根目录执行：

```bash
.local/miniforge3/envs/trajgs/bin/python -m cma_vae.smoke_camera_encoders --camera_dim 9
.local/miniforge3/envs/trajgs/bin/python -m cma_vae.smoke_camera_encoders --camera_dim 11
```

可指定 --dinov3_weights PATH 和 --output REPORT.json。默认命令必须加载预训练权重；只有显式 --random_backbone 才允许随机骨干。合成数据 B=2、T=20，分别检查两个 posterior、latent、非零有限梯度、无图、混合 batch 与 decode 形状。

```bash
.local/miniforge3/envs/trajgs/bin/python -m unittest discover -s tests -p 'test_camera*.py' -v
```

29 项测试覆盖已有 codec 与新编码器。确定性的视觉 test double 用于验证 bbox 区域、逐帧顺序、首帧不广播、混合索引和梯度边界。真实 DINOv3 架构的 CUDA 验证另有 9D/11D 报告，仍待预训练权重验证。

完整结果见 [camera-encoder-verification.json](camera-encoder-verification.json)。可视化见 [相机双编码器](html/module-architectures/camera-encoders.html)，原版四模块分析保留在 [原版逐层对照](html/module-architectures/encoder-layer-comparison.html)。

### 3.2 下一阶段

真实数据层需保证相机、图像、bbox 的 20 时刻一致，并按场景划分训练/验证。随后定义相机重建损失、标准正态 KL 和 DPA；当前没有启用人体重建损失，也没有启动训练。

推理时可用的参考视图与训练目标轨迹图像需要分开定义，不能假定未知轨迹的真值图像始终可用。文本条件生成、连续 latent 生成模块和渲染反馈属于后续工作。
