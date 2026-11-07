---
title: 相机 CMA-VAE · 9D / 11D 双编码器
subtitle: 20 步全长序列 · 逐帧 bbox · 冻结 DINOv3 ViT-B/16
lang: zh
template: doc
theme: shadcn
mode: light
file: camera-encoders
---
CameraCMAVAE 已实现。相机通道为 256，联合通道为 512。
两种表示均保持 20 个时间步，输出 256 维 latent。

29 项测试已通过。9D/11D 已在 L40S 上通过真实骨干架构测试。
该测试显式使用随机 DINOv3 权重；预训练验证等待用户提供权重。

## A 9D 与 11D · 启动时选择

同一个类通过 camera_dim 初始化。默认值为 11。
两种方案共享内部架构，不共享两个独立实例的参数。

```flow LR
camera_dim=9\nB × 20 × 9 -> 输入 Linear\n9 → 256
输入 Linear\n9 → 256 -> 双编码器\nlatent 20 × B × 256
双编码器\nlatent 20 × B × 256 -> 显式 decode\n末层 Linear 256 → 9
显式 decode\n末层 Linear 256 → 9 -> 输出 B × 20 × 9
```

```flow LR
camera_dim=11\nB × 20 × 11 -> 输入 Linear\n11 → 256
输入 Linear\n11 → 256 -> 双编码器\nlatent 20 × B × 256
双编码器\nlatent 20 × B × 256 -> 显式 decode\n末层 Linear 256 → 11
显式 decode\n末层 Linear 256 → 11 -> 输出 B × 20 × 11
```

| 项目 | 9D | 11D |
| --- | --- | --- |
| 相机布局 | 中心 3 + XYZW 四元数 4 + FOV 2 | 中心 3 + 旋转前两列 6 + FOV 2 |
| 输入投影权重 | [256,9] | [256,11] |
| 输出投影权重 | [9,256] | [11,256] |
| 时间上下采样 | Identity | Identity |
| Transformer 长度 | 始终 20 | 始终 20 |
| 几何逆转换 | 既有 quaternion codec | 既有 Gram-Schmidt codec |

网络 decode 返回原始向量，不自动调用几何 codec。
输出形状正确不代表随机预测已是有效相机。

## B 每张图独立裁剪 · DINOv3 内部通道

```flow TB
RGB 图像\nB × 20 × 3 × H × W -> 按帧裁剪 bbox
像素 xyxy\nB × 20 × 4 -> 按帧裁剪 bbox
按帧裁剪 bbox -> 双线性拉伸与标准化\nB × 20 × 3 × 224 × 224
双线性拉伸与标准化\nB × 20 × 3 × 224 × 224 -> 按 B 和时间展平\nN = B × 20
按 B 和时间展平\nN = B × 20 -> PatchEmbed Conv2d\n3 → 768 · k16 s16
PatchEmbed Conv2d\n3 → 768 · k16 s16 -> 196 个 patch token\n加入 1 CLS 和 4 storage
196 个 patch token\n加入 1 CLS 和 4 storage -> 12 个 DINOv3 Block\nN × 201 × 768
12 个 DINOv3 Block\nN × 201 × 768 -> 取 x_norm_clstoken\nN × 768
取 x_norm_clstoken\nN × 768 -> 恢复时间轴\nB × 20 × 768
恢复时间轴\nB × 20 × 768 -> 可训练 Linear 768 → 256\nB × 20 × 256
```

| 处理步骤 | 当前实现 |
| --- | --- |
| 坐标 | 原图像素 xyxy；左上 floor、右下 ceil，再裁到边界 |
| 无效框 | 非有限、反向、零面积或裁后为空均报错 |
| Resize | 直接拉伸 224×224；bilinear，antialias=True |
| 输入像素 | RGB uint8，或浮点 [0,1] |
| 标准化 | mean 0.485/0.456/0.406；std 0.229/0.224/0.225 |
| DINOv3 | ViT-B/16；12 层，768 通道，12 heads，每头 64 |
| DINO FFN | 768 → 3072 → 768 |
| 提取结果 | 只取归一化 CLS；不汇聚 storage token |
| 视觉投影 | Linear 768 → 256，带 bias，无额外激活 |
| 冻结范围 | DINO 参数冻结且始终 eval；投影层继续训练 |
| 执行分块 | 默认每块 20 张图；按原顺序恢复结果 |

裁剪不额外输入 bbox 的原图位置和占比。
相机向量中的 FOV 仍对应原图。

来源：[视觉前处理](../../../cma_vae/models/architectures/camera_vision.py) · [DINOv3 官方骨干](../../../submodules/DINOv3/dinov3/hub/backbones.py)。

## C Camera Encoder · 纯相机路径

```flow TB
相机输入\nB × 20 × D -> camera_embedding\nLinear D → 256
camera_embedding\nLinear D → 256 -> camera_downsample\nIdentity
camera_downsample\nIdentity -> 换轴并加入共享 PE\n20 × B × 256
换轴并加入共享 PE\n20 × B × 256 -> camera_encoder\n9 层 SkipTransformer 256
camera_encoder\n9 层 SkipTransformer 256 -> camera_dist_layer\nLinear 256 → 512
camera_dist_layer\nLinear 256 → 512 -> 拆分 mu 与 logvar\n各 20 × B × 256
拆分 mu 与 logvar\n各 20 × B × 256 -> FP32 Normal 与 rsample\nz 为 20 × B × 256
```

可学习 PE 参数为 [500,1,256]，取前 20 项。
本版固定完整 20 步，不接收 lengths 或时间 padding。
没有 CLS 汇聚、时间池化或因果 mask。

| 子模块 | 通道 |
| --- | --- |
| 输入投影 | D → 256 |
| Attention QKV | 256 → 768 |
| Attention heads | 4 × 64 |
| FFN | 256 → 1024 → 256 |
| 每次长跳连 | concat 256+256 → Linear 512→256 |
| 末尾归一化 | LayerNorm 256 |
| 后验参数 | 512 = mu 256 + logvar 256 |

## D Vision-Camera Encoder · 同时刻拼接

```flow TB
相机输入\nB × 20 × D -> 共享 Linear 与 PE\n20 × B × 256
同步图像与 bbox -> 冻结 DINOv3 与视觉投影\nB × 20 × 256
冻结 DINOv3 与视觉投影\nB × 20 × 256 -> 视觉换轴\n20 × B × 256
共享 Linear 与 PE\n20 × B × 256 -> 通道拼接\n20 × B × 512
视觉换轴\n20 × B × 256 -> 通道拼接\n20 × B × 512
通道拼接\n20 × B × 512 -> joint_encoder\n9 层 SkipTransformer 512
joint_encoder\n9 层 SkipTransformer 512 -> joint_dist_layer\nLinear 512 → 512
joint_dist_layer\nLinear 512 → 512 -> mu 与 logvar\n各 20 × B × 256
mu 与 logvar\n各 20 × B × 256 -> FP32 Normal 与 rsample\nz 为 20 × B × 256
```

共享相机前端不调用纯相机 Transformer。
相机和视觉特征在同一时刻拼接，时间长度仍为 20。

| 子模块 | 通道 |
| --- | --- |
| 融合 | camera 256 + vision 256 = 512 |
| Attention QKV | 512 → 1536 |
| Attention heads | 4 × 128 |
| FFN | 512 → 2048 → 512 |
| 每次长跳连 | concat 512+512 → Linear 1024→512 |
| 末尾归一化 | LayerNorm 512 |
| 后验参数 | 512 = mu 256 + logvar 256 |

相较原版 384 维联合路径，当前扩展为 512 维。
原版关节采样、视觉 Conv2d 和首帧广播不参与此模型。

## E 两个 Transformer · 层与跳连

两条分支均为 4 个输入块、1 个中间块、4 个输出块。
d=256 / F=1024 对应相机；d=512 / F=2048 对应联合。

```flow TB
输入 d -> E1 输入块 0 -> E2 输入块 1 -> E3 输入块 2 -> E4 输入块 3 -> M 中间块
M 中间块 -> 拼接 E4 并 Linear 2d→d -> O1 输出块 0
O1 输出块 0 -> 拼接 E3 并 Linear 2d→d -> O2 输出块 1
O2 输出块 1 -> 拼接 E2 并 Linear 2d→d -> O3 输出块 2
O3 输出块 2 -> 拼接 E1 并 Linear 2d→d -> O4 输出块 3
O4 输出块 3 -> 最终 LayerNorm d
E4 输入块 3 --> 拼接 E4 并 Linear 2d→d
E3 输入块 2 --> 拼接 E3 并 Linear 2d→d
E2 输入块 1 --> 拼接 E2 并 Linear 2d→d
E1 输入块 0 --> 拼接 E1 并 Linear 2d→d
```

每个 EncoderLayer 的内部层序列如下。
全程保持 20 个 token，不改变时间长度。

| 顺序 | 相机 d=256 | 联合 d=512 |
| --- | --- | --- |
| QKV 投影 | 256→768 | 512→1536 |
| 切分 Q/K/V | 各 [B,4,20,64] | 各 [B,4,20,128] |
| Self-attention | 每头 20×20；attention dropout 0.1 | 同左 |
| 合并与 out_proj | 256→256 | 512→512 |
| Dropout → Add → norm1 | 0.1；残差；LN256 | 0.1；残差；LN512 |
| linear1 → GELU → Dropout | 256→1024；0.1 | 512→2048；0.1 |
| linear2 | 1024→256 | 2048→512 |
| Dropout → Add → norm2 | 0.1；残差；LN256 | 0.1；残差；LN512 |

这是 Post-LN；没有 LayerScale。
四个跨层 Linear 独立，每个 Transformer 块的参数也独立。
两条分支仅共享最前面的相机 Linear 和 PE。

来源：[CameraCMAVAE](../../../cma_vae/models/architectures/camera_cma_vae.py) · [复用的 Transformer 算子](../../../cma_vae/models/operator/cross_attention.py)。

## F 调用接口与 Decoder

| 方法 / 字段 | 形状与语义 |
| --- | --- |
| encode_camera | 输入 [B,20,D]；返回 z 与 Normal 后验 |
| encode_joint | 接收完整相机、图像、bbox；返回联合 z 与 Normal |
| forward | 只运行编码器；按 has_images 选取有图子 batch |
| latent_camera / dist_camera | 覆盖整个 B，形状 [20,B,256] |
| latent_joint / dist_joint | 覆盖有图 Bv，形状 [20,Bv,256]；无图为 None |
| joint_indices | 有图子 batch 对应的原 batch 索引 |
| latent | 按原 batch 顺序回填联合与纯相机 latent |
| has_images | 样本级 bool [B]；不支持部分缺帧 |

有图样本必须具备完整 20 张图和有效框。
无图样本只进入纯相机分支，不触发 DINOv3 加载。
train/eval 中均按图像可用性选取 latent；两种模式均使用 rsample。

```flow TB
显式传入 z\n20 × B × 256 -> Decoder 独立 PE\n20 × B × 256
Decoder 独立 PE\n20 × B × 256 -> Decoder 9 层 SkipTransformer\n256 通道 · FFN 1024
Decoder 9 层 SkipTransformer\n256 通道 · FFN 1024 -> 换轴与 Identity 上采样\nB × 20 × 256
换轴与 Identity 上采样\nB × 20 × 256 -> final_layer\nLinear 256 → camera_dim
final_layer\nLinear 256 → camera_dim -> 原始预测向量\nB × 20 × D
```

Decoder 保留原 all_encoder 拓扑。
当前只验证接口与形状，相机约束、损失与训练留待下一阶段。

## G 验证状态与启动方式

| 验证 | 结果 |
| --- | --- |
| codec 与新编码器单元测试 | 29 项通过 |
| bbox 区域与时间顺序 | 可辨识图像测试通过；修改一帧只改变对应融合位置 |
| 无图与混合 batch | 通过；联合结果按 joint_indices 回填 |
| 梯度 | 相机投影、两个 encoder、视觉投影均有有限非零梯度 |
| DINO 冻结 | 无梯度，始终 eval |
| 9D · 真实 DINO 架构 | L40S CUDA 通过；显式随机权重 |
| 11D · 真实 DINO 架构 | L40S CUDA 通过；显式随机权重 |
| 预训练权重验证 | 待用户下载；尚未完成 |

默认权重使用 ViT-B/16 LVD-1689M 的官方 PyTorch 文件。
默认位置为 /mnt/d/model/dinov3/ 下的
`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`。

源码固定为 `6876159a11b4df116f30f667f8c9888617df0751`。
模型默认要求预训练文件，缺失时不会随机回退。

从项目根目录运行：

```bash title="预训练权重就位后的验证命令"
.local/miniforge3/envs/trajgs/bin/python -m cma_vae.smoke_camera_encoders --camera_dim 9
.local/miniforge3/envs/trajgs/bin/python -m cma_vae.smoke_camera_encoders --camera_dim 11
```

自定义位置使用 `--dinov3_weights PATH`。
显式添加 `--random_backbone` 仅验证架构。
通过 `--output REPORT.json` 保存本次结果。

| 入口 | 用途 |
| --- | --- |
| [相机配置](../../../cma_vae/configs/config_camera_vae.yaml) | camera_dim 与固定通道参数 |
| [验证入口](../../../cma_vae/smoke_camera_encoders.py) | 实际骨干 forward/backward |
| [验证记录](../../camera-encoder-verification.json) | 9D/11D 报告及预训练状态 |
| [测试源码](../../../tests/test_camera_encoders.py) | 输入契约、配对、梯度与错误用例 |
| [原版逐层对照](encoder-layer-comparison.html) | 原始 CMA-VAE 与 DA3 架构 |
| [返回图册](index.html) | 已用与对比模块导航 |
