---
title: 四个编码模块 · 逐层结构与通道
subtitle: DA3 CameraEnc · CMA-VAE 输入 Linear · Vision-Motion Encoder · Motion Encoder
lang: zh
template: doc
theme: shadcn
mode: light
file: encoder-layer-comparison
---
DA3 的相机 token 宽度为 1024（Large）。
CMA-VAE 的动作宽度为 256，联合宽度为 384，后验 latent 均为 256。

本页分析当前源码。CMA-VAE 使用迁入配置 D=269、r=4。
图中的 T=20 是形状示例。
相机 D=9/11、r=1 已独立实现，见 [相机双编码器](camera-encoders.html)。

| 记号 | 含义 |
| --- | --- |
| B | batch 大小 |
| T=20 | CMA-VAE 输入时间步数 |
| L=5 | 当前 r=4 下的 latent 时间步数 |
| S=20 | DA3 示例视图数；模块内没有时间位置编码 |
| C / d | 特征通道数；不是时间长度 |
| 实线 / 虚线 | 主计算路径 / 残差或长跳连 |

图表可展开查看。下方按四个模块组织，并单独展开 Transformer 与视觉卷积。

## A DA3 Camera Encoder · 整体逐层

`CameraEnc.forward(ext, ixt, image_size)` 先转换相机参数。
输入外参为 W2C；内部求逆得到 C2W。
9D 由相机中心 3D、四元数 XYZW 4D、FOV 2D 组成。

```flow TB
相机外参与内参 -> 几何转换\nB × 20 × 9
几何转换\nB × 20 × 9 -> pose_branch.fc1\nLinear 9 → 512
pose_branch.fc1\nLinear 9 → 512 -> GELU\nDropout p=0
GELU\nDropout p=0 -> pose_branch.fc2\nLinear 512 → 1024
pose_branch.fc2\nLinear 512 → 1024 -> Dropout p=0\ntoken_norm 1024
Dropout p=0\ntoken_norm 1024 -> trunk.0 · Block 1\nB × 20 × 1024
trunk.0 · Block 1\nB × 20 × 1024 -> trunk.1 · Block 2\nB × 20 × 1024
trunk.1 · Block 2\nB × 20 × 1024 -> trunk.2 · Block 3\nB × 20 × 1024
trunk.2 · Block 3\nB × 20 × 1024 -> trunk.3 · Block 4\nB × 20 × 1024
trunk.3 · Block 4\nB × 20 × 1024 -> trunk_norm\nLayerNorm 1024
trunk_norm\nLayerNorm 1024 -> 相机条件 token\nB × 20 × 1024
```

所有 Linear 均带 bias。末层 MLP 后没有 GELU。
四个 Block 参数独立；内部展开见下一节。
输出保留所有视图，没有池化或高斯后验头。

| DA3 配置 | pose_branch 两层 Linear | Block 通道 d | QKV 投影 | heads × head_dim | FFN |
| --- | --- | --- | --- | --- | --- |
| Small | 9 → 192 → 384 | 384 | 384 → 1152 | 16 × 24 | 384 → 1536 → 384 |
| Base | 9 → 384 → 768 | 768 | 768 → 2304 | 16 × 48 | 768 → 3072 → 768 |
| Large · 主图 | 9 → 512 → 1024 | 1024 | 1024 → 3072 | 16 × 64 | 1024 → 4096 → 1024 |
| Giant | 9 → 768 → 1536 | 1536 | 1536 → 4608 | 16 × 96 | 1536 → 6144 → 1536 |

来源：[CameraEnc](../../../submodules/Depth-Anything-3/src/depth_anything_3/model/cam_enc.py) · [Large 配置](../../../submodules/Depth-Anything-3/src/depth_anything_3/configs/da3-large.yaml)。

## B DA3 Block · 注意力与 FFN 内部

这是 A 中每个 Block 的完整计算顺序。
采用 Pre-LN，LayerNorm 位于每条残差支路的起点。

```flow TB
输入 x\nB × S × 1024 -> norm1\nLayerNorm 1024
norm1\nLayerNorm 1024 -> attn.qkv\nLinear 1024 → 3072
attn.qkv\nLinear 1024 → 3072 -> 拆为 Q K V\n各 B × 16 × S × 64
拆为 Q K V\n各 B × 16 × S × 64 -> 视图间 SDPA\n每个 head 计算 S × S 注意力
视图间 SDPA\n每个 head 计算 S × S 注意力 -> 合并 16 个 head\nB × S × 1024
合并 16 个 head\nB × S × 1024 -> attn.proj\nLinear 1024 → 1024
attn.proj\nLinear 1024 → 1024 -> ls1 · LayerScale\n1024 个可学习缩放值
ls1 · LayerScale\n1024 个可学习缩放值 -> 残差相加一\nB × S × 1024
输入 x\nB × S × 1024 --> 残差相加一\nB × S × 1024
残差相加一\nB × S × 1024 -> norm2\nLayerNorm 1024
norm2\nLayerNorm 1024 -> mlp.fc1\nLinear 1024 → 4096
mlp.fc1\nLinear 1024 → 4096 -> FFN GELU\nB × S × 4096
FFN GELU\nB × S × 4096 -> mlp.fc2\nLinear 4096 → 1024
mlp.fc2\nLinear 4096 → 1024 -> ls2 · LayerScale\n1024 个可学习缩放值
ls2 · LayerScale\n1024 个可学习缩放值 -> 残差相加二\nB × S × 1024
残差相加一\nB × S × 1024 --> 残差相加二\nB × S × 1024
```

| 内部细节 | 当前 CameraEnc 设置 |
| --- | --- |
| 注意力轴 | 视图轴 S；S=20 时，每头的注意力矩阵为 20×20 |
| LayerScale | 两组独立参数，各为 [1024]；初始值 0.01 |
| Attention / projection / MLP dropout | 全部 p=0，图中省略恒等 Dropout |
| Q/K 归一化 | 关闭；q_norm、k_norm 为 Identity |
| RoPE / 时间位置编码 | 此模块内均未启用 |
| DropPath | 关闭 |
| mask | CameraEnc.forward 没有传入注意力 mask |

源码调用 SDPA；上表给出逻辑注意力矩阵尺寸，不表示必然显式分配该矩阵。

来源：[Block](../../../submodules/Depth-Anything-3/src/depth_anything_3/model/utils/block.py) · [Attention、Mlp、LayerScale](../../../submodules/Depth-Anything-3/src/depth_anything_3/model/utils/attention.py)。

## C CMA-VAE 最早的 Linear · 两条路径共享

最前层是 `motion_embedding = Linear(nfeats, 256)`。
它对每个时间步独立使用同一组权重，保留时间长度。

```flow TB
动作输入\nB × 20 × 269 -> motion_embedding\nLinear 269 → 256
motion_embedding\nLinear 269 → 256 -> 投影特征\nB × 20 × 256
投影特征\nB × 20 × 256 -> motion_downsample\n共享时间前端
motion_downsample\n共享时间前端 -> encode_motion 路径\nB × 5 × 256
motion_downsample\n共享时间前端 -> encode_joint 路径\nB × 5 × 256
```

| 对象 | 当前形状 / 行为 |
| --- | --- |
| Linear.weight | [256,269] |
| Linear.bias | [256] |
| 单帧计算 | y = x Wᵀ + b |
| Linear 内部 | 一次仿射变换；没有激活、归一化或注意力 |
| 两条路径的共享范围 | motion_embedding、motion_downsample、motion_pos_encoder |
| 两条路径的调用方式 | 各自调用共享模块，不是两套独立权重 |
| 后续不共享 | motion_encoder 与 joint_encoder；两个 posterior Linear |

### Linear 后面的时间前端

`TemporalDownsample` 是独立模块，不属于上述 Linear。
r=4 时，实际层序列如下。

| 执行顺序 | 层与参数 | 输出形状 |
| --- | --- | --- |
| 0 | Permute B,T,C → B,C,T | [B,256,20] |
| 1 | Conv1d 256 → 256；k=4，s=2，p=1 | [B,256,10] |
| 2 | GroupNorm；8 groups，256 channels | [B,256,10] |
| 3 | GELU | [B,256,10] |
| 4 | Conv1d 256 → 256；k=4，s=2，p=1 | [B,256,5] |
| 5 | GroupNorm；8 groups，256 channels | [B,256,5] |
| 6 | GELU | [B,256,5] |
| 7 | Permute B,C,L → B,L,C | [B,5,256] |

相机新实现使用 9 → 256 或 11 → 256。
r=1 跳过时间卷积，保留 20 步。
本页继续展示原版基线，不替换其历史通道。

来源：[CMAVAE 构造与 TemporalDownsample](../../../cma_vae/models/architectures/mld_vae_kl_align.py)。

## D Vision-Motion Encoder · 两路输入怎样融合

此路径对应 `encode_joint`；Transformer 子模块名为 `joint_encoder`。
图中视觉输入使用原版单张参考图接口 `[B,1,H,W,3]`。

```flow TB
动作序列\nB × 20 × 269 -> 共享 Linear 与时间前端\nB × 5 × 256
参考 RGB\nB × 1 × H × W × 3 -> 冻结 HRNet\n选取 256 通道特征图
冻结 HRNet\n选取 256 通道特征图 -> grid_sample\nB × 256 × 1 × 22
首帧 22 个关节点\nB × 1 × 22 × 2 -> grid_sample\nB × 256 × 1 × 22
grid_sample\nB × 256 × 1 × 22 -> VisionEncoder\nB × 128 × 1 × 22
VisionEncoder\nB × 128 × 1 × 22 -> 对 22 个关节取均值\nB × 1 × 128
对 22 个关节取均值\nB × 1 × 128 -> repeat 到 L=5\nB × 5 × 128
共享 Linear 与时间前端\nB × 5 × 256 -> 通道拼接 256 + 128\nB × 5 × 384
repeat 到 L=5\nB × 5 × 128 -> 通道拼接 256 + 128\nB × 5 × 384
通道拼接 256 + 128\nB × 5 × 384 -> 转为 L × B × 384\n只对前 256 通道加位置编码
转为 L × B × 384\n只对前 256 通道加位置编码 -> joint_encoder\n9 层 SkipTransformer · d=384
joint_encoder\n9 层 SkipTransformer · d=384 -> joint_dist_layer\nLinear 384 → 512
joint_dist_layer\nLinear 384 → 512 -> mu 与 logvar\n各 5 × B × 256
mu 与 logvar\n各 5 × B × 256 -> Normal.rsample 与 mask\nz 为 5 × B × 256
```

| 融合位置 | 精确含义 |
| --- | --- |
| grid_sample | 在特征图上按关节位置采样；22 是关节轴 J |
| VisionEncoder | 在 [B,C,T_v,J] 上运行 Conv2d；内部详见 E |
| mean pool | 仅汇聚 J 轴，得到一个 128 维视觉向量 |
| repeat | 将同一个视觉向量复制 L 次 |
| concat | 逐 latent 位置拼接通道；时间长度仍为 L |
| 位置编码 | 共享 [500,1,256] 参数；只加到动作部分 |
| 联合注意力 | 对融合 token 做 self-attention；没有独立 cross-attention 层 |
| 后验头 | 384 → 512，再拆成两份 256 |

这里没有先融合成 256 再进 Transformer。
联合 Transformer 全程为 384，直到后验参数头才改为 512。

`encode_joint` 取首帧动作与关节点，但将 `video_rgb` 原样传给提取器。
因此应提供单帧图像；不能直接按注释传入整段 20 帧视频。
没有图像时，代码使用 [B,L,128] 零特征。

逐帧 bbox 融合已在独立相机模型中实现。
本页分析的原版代码仍使用关节采样与单帧广播。

来源：[encode_joint 与 get_vision_feats](../../../cma_vae/models/architectures/mld_vae_kl_align.py) · [wrapper 的单帧接口](../../../cma_vae/models/modeltype/mld_kl_align.py)。

## E VisionEncoder · 卷积层逐层展开

D 中的 `vision_encoder` 是卷积网络。
它与负责联合建模的 `joint_encoder` 是两个不同子模块。

本表使用当前 r=4、单参考图 T_v=1、关节数 J=22。
Conv2d 的两个空间轴实际是“参考图时间轴 × 关节轴”。

| 顺序 / 模块 | 内部层顺序 | 通道 | 当前输出 [B,C,T_v,J] |
| --- | --- | --- | --- |
| conv_in → norm_in → act | Conv2d k3,p1 → GroupNorm 8 → SiLU | 256 → 256 | [B,256,1,22] |
| layers.0 · ResBlock2d | Conv → GN → SiLU → Conv → GN → Add → SiLU | 256 → 256 → 256 | [B,256,1,22] |
| layers.1 · Downsample2d | Conv2d k3,p1,s(2,1) → GN → SiLU | 256 → 256 | [B,256,1,22] |
| layers.2 · ResBlock2d | Conv → GN → SiLU → Conv → GN → Add → SiLU | 256 → 256 → 256 | [B,256,1,22] |
| layers.3 · Downsample2d | Conv2d k3,p1,s(2,1) → GN → SiLU | 256 → 256 | [B,256,1,22] |
| layers.4 · ResBlock2d | Conv → GN → SiLU → Conv → GN → Add → SiLU | 256 → 256 → 256 | [B,256,1,22] |
| conv_out → norm_out → act | Conv2d k3,p1 → GroupNorm 8 → SiLU | 256 → 128 | [B,128,1,22] |
| vision_pooler | Mean over J → Permute | 128 | [B,1,128] |

表内全部 Conv 均为 3×3、带 bias。
ResBlock 中 stride=1；shortcut 为 Identity。
GroupNorm 使用 8 组，通道数跟随对应卷积的输出。

### 每个 ResBlock2d 的内部

```flow TB
视觉块输入\nB × 256 × T_v × 22 -> conv1\nConv2d 256 → 256
conv1\nConv2d 256 → 256 -> norm1 与 SiLU\nGroupNorm 8 groups
norm1 与 SiLU\nGroupNorm 8 groups -> conv2\nConv2d 256 → 256
conv2\nConv2d 256 → 256 -> norm2\nGroupNorm 8 groups
norm2\nGroupNorm 8 groups -> 视觉残差 Add\n通道保持 256
视觉块输入\nB × 256 × T_v × 22 --> 视觉残差 Add\n通道保持 256
视觉残差 Add\n通道保持 256 -> 输出 SiLU\nB × 256 × T_v × 22
```

时间卷积的长度公式为 ceil(T_v/2)。
因此当前单图路径经历 1 → 1 → 1，关节轴始终为 22。
若单独给该卷积模块 20 帧特征，其长度才会变为 20 → 10 → 5。

| HRNet 前置提取 | 当前调用 |
| --- | --- |
| 输入图像 | [B,3,448,448]，来自单参考帧 |
| 选择的输出 | output_level=3；256 通道，约 1/32 空间分辨率 |
| 448×448 示例 | [B,256,14,14] |
| 采样网格 | [B,22,1,2]，坐标归一化到 [-1,1] |
| 采样结果 | [B,256,22,1] → [B,256,1,22] |

HRNet 是前置图像骨干，本页展开的是 CMA-VAE 内部视觉投影与联合编码层。
上述路径以 HRNet 成功加载为前提；本轮没有加载其权重。

来源：[VisionEncoder、ResBlock2d、Downsample2d](../../../cma_vae/models/architectures/vision_utils.py) · [HRNet 包装器](../../../cma_vae/models/architectures/hrnet_wrapper.py)。

## F Motion Encoder · 纯动作路径逐层

完整路径是 `encode_motion`；其中 `motion_encoder` 仅指 9 层 Transformer。
这里接收动作特征，当前没有图像或 bbox 输入。

```flow TB
features\nB × 20 × 269 -> 补齐时间长度\nT_pad 为 4 的倍数
补齐时间长度\nT_pad 为 4 的倍数 -> motion_embedding\nLinear 269 → 256
motion_embedding\nLinear 269 → 256 -> 时间前端第一组\nConv1d + GN + GELU · T 20 → 10
时间前端第一组\nConv1d + GN + GELU · T 20 → 10 -> 时间前端第二组\nConv1d + GN + GELU · T 10 → 5
时间前端第二组\nConv1d + GN + GELU · T 10 → 5 -> 转为 L × B × 256\n加可学习位置编码
转为 L × B × 256\n加可学习位置编码 -> motion_encoder\n9 层 SkipTransformer · d=256
motion_encoder\n9 层 SkipTransformer · d=256 -> motion_dist_layer\nLinear 256 → 512
motion_dist_layer\nLinear 256 → 512 -> mu\n5 × B × 256
motion_dist_layer\nLinear 256 → 512 -> logvar\n5 × B × 256
mu\n5 × B × 256 -> 重参数化采样\nz = mu + exp 0.5logvar × epsilon
logvar\n5 × B × 256 -> 重参数化采样\nz = mu + exp 0.5logvar × epsilon
重参数化采样\nz = mu + exp 0.5logvar × epsilon -> 清零 padding latent\n5 × B × 256
```

epsilon 来自标准正态分布。每个 latent 位置保留 256 个通道。
没有 CLS token 或全序列池化；eval 模式也仍调用 rsample。

| 阶段 | 输出形状 | 通道 / 长度 |
| --- | --- | --- |
| 输入 | [B,20,269] | D=269，T=20 |
| 输入 Linear | [B,20,256] | C=256，T 不变 |
| 两次时间卷积 | [B,5,256] | C=256，L=5 |
| 位置编码 | [5,B,256] | 加 [5,1,256]，来自容量 500 的参数 |
| 9 个 EncoderLayer 与末尾 LN | [5,B,256] | 每一层保持 256 |
| 后验 Linear | [5,B,512] | 拼接的 mu、logvar |
| 拆分与采样 | [5,B,256] | mu、logvar、z 各 256 |

`lengths` 生成 key padding mask；注意力未使用因果 mask。
有效位置之间可双向交互。
采样后清零无效 z，返回的分布参数本身不清零。

来源：[encode_motion](../../../cma_vae/models/architectures/mld_vae_kl_align.py) · [迁入配置](../../../cma_vae/configs/config_cma_vae.yaml)。

## G 两个 CMA-VAE Encoder · 9 层与长跳连

Motion Encoder 使用 d=256；Vision-Motion Encoder 使用 d=384。
它们结构相同，但权重独立。图中 E、M、O 全部是 EncoderLayer。

```flow TB
输入 L × B × d -> E1 · 输入块 0\n通道 d
E1 · 输入块 0\n通道 d -> E2 · 输入块 1\n通道 d
E2 · 输入块 1\n通道 d -> E3 · 输入块 2\n通道 d
E3 · 输入块 2\n通道 d -> E4 · 输入块 3\n通道 d
E4 · 输入块 3\n通道 d -> M · 中间块\n通道 d
M · 中间块\n通道 d -> 拼接 M 与 E4\n2d → Linear → d
拼接 M 与 E4\n2d → Linear → d -> O1 · 输出块 0\n通道 d
O1 · 输出块 0\n通道 d -> 拼接 O1 与 E3\n2d → Linear → d
拼接 O1 与 E3\n2d → Linear → d -> O2 · 输出块 1\n通道 d
O2 · 输出块 1\n通道 d -> 拼接 O2 与 E2\n2d → Linear → d
拼接 O2 与 E2\n2d → Linear → d -> O3 · 输出块 2\n通道 d
O3 · 输出块 2\n通道 d -> 拼接 O3 与 E1\n2d → Linear → d
拼接 O3 与 E1\n2d → Linear → d -> O4 · 输出块 3\n通道 d
O4 · 输出块 3\n通道 d -> 末尾 LayerNorm d\nL × B × d
E4 · 输入块 3\n通道 d --> 拼接 M 与 E4\n2d → Linear → d
E3 · 输入块 2\n通道 d --> 拼接 O1 与 E3\n2d → Linear → d
E2 · 输入块 1\n通道 d --> 拼接 O2 与 E2\n2d → Linear → d
E1 · 输入块 0\n通道 d --> 拼接 O3 与 E1\n2d → Linear → d
```

| 深度 | 源码模块 | Motion 通道变化 | Vision-Motion 通道变化 |
| --- | --- | --- | --- |
| 1 · E1 | input_blocks.0 | 256 → 256，保存输出 | 384 → 384，保存输出 |
| 2 · E2 | input_blocks.1 | 256 → 256，保存输出 | 384 → 384，保存输出 |
| 3 · E3 | input_blocks.2 | 256 → 256，保存输出 | 384 → 384，保存输出 |
| 4 · E4 | input_blocks.3 | 256 → 256，保存输出 | 384 → 384，保存输出 |
| 5 · M | middle_block | 256 → 256 | 384 → 384 |
| O1 前 | concat E4 → linear_blocks.0 | 256+256 → 512 → 256 | 384+384 → 768 → 384 |
| 6 · O1 | output_blocks.0 | 256 → 256 | 384 → 384 |
| O2 前 | concat E3 → linear_blocks.1 | 256+256 → 512 → 256 | 384+384 → 768 → 384 |
| 7 · O2 | output_blocks.1 | 256 → 256 | 384 → 384 |
| O3 前 | concat E2 → linear_blocks.2 | 256+256 → 512 → 256 | 384+384 → 768 → 384 |
| 8 · O3 | output_blocks.2 | 256 → 256 | 384 → 384 |
| O4 前 | concat E1 → linear_blocks.3 | 256+256 → 512 → 256 | 384+384 → 768 → 384 |
| 9 · O4 | output_blocks.3 | 256 → 256 | 384 → 384 |
| 堆栈出口 | norm | LayerNorm 256 | LayerNorm 384 |

长跳连使用 concat + Linear，块内残差使用 Add。
四个跳连 Linear 带 bias，彼此独立，后面直接接 EncoderLayer。
整个堆栈的时间长度 L 保持不变。

来源：[SkipTransformerEncoder](../../../cma_vae/models/operator/cross_attention.py)。

## H CMA-VAE 单个 EncoderLayer · 内部逐层对照

这张图适用于 G 中全部 9 层。
d=256 对应 Motion；d=384 对应 Vision-Motion。
两者都采用 Post-LN、4 个注意力头、1024 维 FFN。

```flow TB
块输入 x\nL × B × d -> QKV 投影\nd → 3d
QKV 投影\nd → 3d -> 分为 Q K V\n每个 B × 4 × L × d/4
分为 Q K V\n每个 B × 4 × L × d/4 -> 时间自注意力\nL × L · attention dropout 0.1
时间自注意力\nL × L · attention dropout 0.1 -> 合并 heads 与 out_proj\nLinear d → d
合并 heads 与 out_proj\nLinear d → d -> dropout1\np=0.1
dropout1\np=0.1 -> 残差 Add 一\nL × B × d
块输入 x\nL × B × d --> 残差 Add 一\nL × B × d
残差 Add 一\nL × B × d -> norm1\nLayerNorm d
norm1\nLayerNorm d -> linear1\nLinear d → 1024
linear1\nLinear d → 1024 -> GELU 与 FFN Dropout\np=0.1
GELU 与 FFN Dropout\np=0.1 -> linear2\nLinear 1024 → d
linear2\nLinear 1024 → d -> dropout2\np=0.1
dropout2\np=0.1 -> 残差 Add 二\nL × B × d
norm1\nLayerNorm d --> 残差 Add 二\nL × B × d
残差 Add 二\nL × B × d -> norm2\nLayerNorm d
```

| 内部层 | Motion Encoder | Vision-Motion Encoder |
| --- | --- | --- |
| QKV 合并投影 | 256 → 768 | 384 → 1152 |
| 每个 Q / K / V | [B,4,L,64] | [B,4,L,96] |
| 逻辑注意力矩阵 | [B,4,L,L] | [B,4,L,L] |
| 合并 heads | 4×64=256 | 4×96=384 |
| self_attn.out_proj | Linear 256 → 256 | Linear 384 → 384 |
| 第一次残差 Add | 两个 [L,B,256] 相加 | 两个 [L,B,384] 相加 |
| norm1 | LayerNorm 256 | LayerNorm 384 |
| linear1 | Linear 256 → 1024 | Linear 384 → 1024 |
| activation → dropout | GELU → Dropout 0.1 | GELU → Dropout 0.1 |
| linear2 | Linear 1024 → 256 | Linear 1024 → 384 |
| 第二次残差 Add | 两个 [L,B,256] 相加 | 两个 [L,B,384] 相加 |
| norm2 | LayerNorm 256 | LayerNorm 384 |

QKV 是 nn.MultiheadAttention 内部的合并投影。
这里写出它的逻辑展开，不代表源码另有三个显式 Linear。
所有 Linear 均带 bias；两个分支都没有 LayerScale。
Dropout 0.1 在训练时生效，eval 时关闭。

### 四个模块的通道结论

| 模块 | 输入投影 | 主干宽度 | FFN 隐藏通道 | 出口 |
| --- | --- | --- | --- | --- |
| DA3 CameraEnc · Large | 9 → 512 → 1024 | 1024，4 层 | 4096 | 1024 维条件 token |
| CMA-VAE 最早 Linear | 269 → 256 | 单层 | 无 FFN | 256 维逐帧特征 |
| CMA-VAE Vision-Motion | 共享 269 → 256；视觉 256 → 128 | 拼接后 384，9 层 | 1024 | 384 → 512 → mu/logvar 各 256 |
| CMA-VAE Motion | 共享 269 → 256 | 256，9 层 | 1024 | 256 → 512 → mu/logvar 各 256 |

来源：[TransformerEncoderLayer](../../../cma_vae/models/operator/cross_attention.py) · [配置](../../../cma_vae/configs/config_cma_vae.yaml) · [源码快照](sources.json)。

[返回模块图册](index.html) · [相机 9D/11D 双向转换](camera-pose-codec.html)
