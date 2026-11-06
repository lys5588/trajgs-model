---
title: Depth Anything 3 · Camera Encoder
subtitle: 从已知相机参数到主干条件 token
lang: zh
template: doc
theme: shadcn
mode: light
file: da3-camera-encoder
---
CameraEnc 将已知相机参数编码成条件 token。其内部使用 MLP 与 4 层 Transformer，不预测 VAE 高斯后验。

[模块目录与对比](index.html) · [CMA-VAE Motion Encoder](motion-encoder.html)

## A 模块职责与分析边界

本页分析 `model/cam_enc.py` 中的 `CameraEnc`。
外参已知时，DA3 调用它生成每个视图的相机条件 token。
图像编码和相机参数预测属于其他模块。

| 项目 | CameraEnc 当前实现 |
| --- | --- |
| forward 输入 | ext、ixt、image_size |
| 转换后的参数维度 | 9 |
| 输出 | `[B,S,d]`，每个视图一个条件 token |
| Transformer 层数 | 4 |
| 注意力 heads | 16 |
| FFN 扩展比 | 4 |
| 残差分支 | Pre-LN＋LayerScale，γ 初始化为 0.01 |
| 采样与高斯参数 | 无 |

源码 docstring 提到 CameraHead 与迭代预测，但实际 forward 是参数编码。
`target_dim` 被保存，却没有参与该 forward。

## B 9D 参数如何构造

ext 使用 world-to-camera 约定。
编码前调用 `affine_inverse`，得到 camera-to-world。
ixt 是相机内参矩阵，image_size 为处理后的 H、W。

```flow TB
外参 ext B × S × 4 × 4 ->affine_inverse
affine_inverse -> c2w 平移与旋转
c2w 平移与旋转 -> 平移 T 3 维
c2w 平移与旋转 -> 旋转矩阵转四元数 4 维
内参 ixt 与图像 H W -> 计算竖直与水平 FOV 2 维
平移 T 3 维 -> 拼接为 B × S × 9
旋转矩阵转四元数 4 维 -> 拼接为 B × S × 9
计算竖直与水平 FOV 2 维 -> 拼接为 B × S × 9
```

| 索引 | 含义 |
| --- | --- |
| 0–2 | c2w 平移 tx、ty、tz |
| 3–6 | 旋转四元数，XYZW，实部最后 |
| 7 | 竖直视场角 fov_h |
| 8 | 水平视场角 fov_w |

$$
\mathrm{fov}_h=2\arctan\frac{H}{2f_y},\qquad\mathrm{fov}_w=2\arctan\frac{W}{2f_x}
$$

这条 9D 路径没有编码内参中的 cx、cy 或 skew。
它只从内参读取 fx、fy，结合图像尺寸得到 FOV。
我们的 11D 特征含义仍由相机模块定义，不与这里的 9D 等同。

## C 完整网络与尺寸

以 DA3-Large 的 d=1024、S=20 为例。
窄屏可横向滑动图表，或点击右上角展开。

```flow TB
9D 相机参数 B × 20 × 9 -> Linear 9 到 512
Linear 9 到 512 -> GELU
GELU -> Linear 512 到 1024
Linear 512 到 1024 -> token_norm LayerNorm
 token_norm LayerNorm -> Transformer Block × 4
Transformer Block × 4 -> trunk_norm LayerNorm
trunk_norm LayerNorm -> 条件 token B × 20 × 1024
```

MLP 中的两个 Dropout 均为 0。
Transformer 的 attention dropout 和 projection dropout 也为 0。
所有层保留视图数量 S，没有时间下采样或池化。

| 官方配置 | d | 前端 MLP | Block FFN | 单 head 维度 |
| --- | --- | --- | --- | --- |
| Small | 384 | 9 → 192 → 384 | 384 → 1536 → 384 | 24 |
| Base | 768 | 9 → 384 → 768 | 768 → 3072 → 768 | 48 |
| Large | 1024 | 9 → 512 → 1024 | 1024 → 4096 → 1024 | 64 |
| Giant | 1536 | 9 → 768 → 1536 | 1536 → 6144 → 1536 | 96 |

这些配置只覆盖主 DA3 系列的 CameraEnc。
不把 Large 的 1024 维泛化为所有版本。

## D 单个 Block：Pre-LN 与 LayerScale

```flow TB
块输入 x -> LayerNorm 一
LayerNorm 一 -> 16 头全视图自注意力
16 头全视图自注意力 -> LayerScale 一
LayerScale 一 -> 残差相加一
块输入 x --> 残差相加一
残差相加一 -> LayerNorm 二
LayerNorm 二 -> MLP d 到 4d 到 d
MLP d 到 4d 到 d -> LayerScale 二
LayerScale 二 -> 残差相加二
残差相加一 --> 残差相加二
```

每个 LayerScale 都有长度为 d 的可学习向量 γ。
两条残差分支的 γ 分别初始化为 0.01。

$$
x_1=x+\mathrm{gamma}_1\odot\mathrm{Attention}(\mathrm{LN}_1(x))
$$

$$
x_2=x_1+\mathrm{gamma}_2\odot\mathrm{MLP}(\mathrm{LN}_2(x_1))
$$

注意力先用 Linear d → 3d 产生 Q、K、V。
Large 中每组形状为 `[B,16,S,64]`。
注意力沿 S 个视图计算，而不是沿 9 个参数维度计算。

该模块默认 `qk_norm=false`、`rope=None`。
没有长跳连；四个 Block 按顺序串联。

## E 它如何处理视图顺序

CameraEnc 内部没有时间位置编码，也没有 view-index embedding。
trunk 接收 `[B,S,d]`，在 S 个相机之间计算自注意力。
`forward` 没有接收或传入 padding mask、因果 mask。

因此，它可整合相机集合的关系，但没有显式标注第几帧或时间间隔。
相机参数中的位置与姿态属于几何信息，不是序列时间编码。

**由结构可推导：** 单独看 CameraEnc，同步重排输入视图会相应重排输出。
这一置换等变性描述模块本身，不代表 DA3 完整流程。

对于有先后顺序的相机轨迹，我们仍需明确时间位置编码。
不能仅因 CameraEnc 使用 Transformer，就认为它已建模轨迹时序。

## F 条件 token 在 DA3 中如何使用

```flow TB
已知 ext 与 ixt -> CameraEnc
CameraEnc -> 相机条件 B × S × d
RGB 图像 B × S × 3 × H × W -> 图像 patch 与特殊 token
图像 patch 与特殊 token -> 前部图像主干层
前部图像主干层 -> 替换每视图第 0 个 token
相机条件 B × S × d -> 替换每视图第 0 个 token
替换每视图第 0 个 token -> 后续局部与全局注意力
```

代码在 `i == alt_start` 时执行 `x[:,:,0] = cam_token`。
Large 的 alt_start=8，是从 0 开始计数的第 8 层索引。
CameraEnc 不处理 RGB，也不在内部提取 bbox 区域特征。

未提供外参时，CameraEnc 不被调用。
主干改用学习到的参考视图 token 和其他视图 token。

高层 API 还会预先对外参作归一化。
它以首个相机为参考，并按相机距离中位数缩放平移。
这发生在 CameraEnc 外部，不能画成它内部的网络层。

## G 对我们相机编码方案的参考价值

| 可以参考的部分 | 接入时仍要定义的部分 |
| --- | --- |
| 用 MLP 映射低维相机参数 | 我们的 11D 含义与输入归一化 |
| 保留每个相机对应的 token | 20 步时间位置编码 |
| 多头注意力整合相机之间的关系 | 逐帧有效性 mask |
| Pre-LN 与 LayerScale 残差结构 | 是否采用该块，而非现有 SkipTransformer |
| 不下采样的序列处理 | 与 bbox 视觉特征的逐时刻融合 |

若仅把构造参数 `dim_in` 改成 11，原 forward 仍会生成 9D。
需要同时设计输入路径；本轮不执行这种修改。

它没有 μ/logvar 头，不能直接替代完整 VAE 编码器。
CMA-VAE 负责轨迹后验，DA3 CameraEnc 负责已知相机条件编码。

## H 源码证据与复核入口

```python src=submodules/Depth-Anything-3/src/depth_anything_3/model/cam_enc.py lines=70-80
```

| 源码 | 核对位置 |
| --- | --- |
| [CameraEnc](../../../submodules/Depth-Anything-3/src/depth_anything_3/model/cam_enc.py) | 30：结构；64：forward |
| [参数转换](../../../submodules/Depth-Anything-3/src/depth_anything_3/model/utils/transform.py) | 19：9D 拼接；100：四元数转换 |
| [Block](../../../submodules/Depth-Anything-3/src/depth_anything_3/model/utils/block.py) | 22：Pre-LN 与残差 |
| [Attention / Mlp / LayerScale](../../../submodules/Depth-Anything-3/src/depth_anything_3/model/utils/attention.py) | 23 / 85 / 70 |
| [DA3 网络调用](../../../submodules/Depth-Anything-3/src/depth_anything_3/model/da3.py) | 125：相机条件路径 |
| [主干注入](../../../submodules/Depth-Anything-3/src/depth_anything_3/model/dinov2/vision_transformer.py) | 323：替换第 0 个 token |
| [高层 API](../../../submodules/Depth-Anything-3/src/depth_anything_3/api.py) | 327：外参归一化 |
| [Large 配置](../../../submodules/Depth-Anything-3/src/depth_anything_3/configs/da3-large.yaml) | cam_enc.dim_out 与 net.alt_start |

依据本地 DA3 checkout 静态核对，未加载预训练权重或执行 forward。
