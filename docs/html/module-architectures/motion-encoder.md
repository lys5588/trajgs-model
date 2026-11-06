---
title: CMA-VAE · Motion Encoder
subtitle: 从动作序列到逐位置高斯后验
lang: zh
template: doc
theme: shadcn
mode: light
file: motion-encoder
---
当前实现把动作序列编码成高斯后验，再用重参数化采样得到 latent。本页分析迁入代码，尚未执行相机适配。

[模块目录与对比](index.html) · [DA3 Camera Encoder](da3-camera-encoder.html)

## A 分析范围与默认配置

这里分析 `CMAVAE.encode_motion()`。
它包含输入投影、时间处理、时序编码和后验预测。
`self.motion_encoder` 仅指其中的 SkipTransformer。

| 参数 | 迁入配置 |
| --- | --- |
| 输入特征 D | 269，人体动作 |
| latent 通道 d | 256 |
| 时间下采样 r | 4 |
| Transformer 层数 | 9，按 4＋1＋4 组织 |
| 注意力 heads | 4，每头 64 维 |
| FFN | 256 → 1024 → 256，GELU |
| 归一化 | Post-LN；整个 SkipTransformer 末尾再加 LayerNorm |
| Dropout | 0.1 |
| 位置编码 | 可学习绝对位置，容量 500 |

这些参数取自迁入的 `config_cma_vae.yaml`。
`latent_dim: [1,256]` 中的 1 不会把输出固定成一个 token。
这里实际使用末项 256，latent 数量由输入长度和 r 决定。

## B 完整 forward 与张量形状

记 B 为 batch 大小，T 为输入时间长度。
令 T_pad 为补齐到 r 的整数倍后的长度，L=T_pad/r。
窄屏可横向滑动图表，或点击右上角展开。

```flow TB
输入 B × T × D -> 时间补零到 T_pad
时间补零到 T_pad -> Linear D 到 256
Linear D 到 256 -> TemporalDownsample
TemporalDownsample -> 换轴为 L × B × 256
换轴为 L × B × 256 -> 加可学习位置编码
加可学习位置编码 -> 9 层 SkipTransformer
9 层 SkipTransformer -> Linear 256 到 512
Linear 256 到 512 -> 拆分均值与 logvar
拆分均值与 logvar -> Normal 后验与重参数化采样
Normal 后验与重参数化采样 -> 清零 padding 位置的 latent
```

| 节点 | 当前 T=20、D=269、r=4 的形状 |
| --- | --- |
| 输入 | `[B,20,269]` |
| 输入投影 | `[B,20,256]` |
| 两次时间卷积后 | `[B,5,256]` |
| Transformer 输入与输出 | `[5,B,256]` |
| 后验参数头 | `[5,B,512]` |
| μ、logvar、采样 z | 各为 `[5,B,256]` |

每个 latent 位置有自己的 256 维对角高斯。
没有全局池化，也没有专门的 CLS 汇聚 token。

## C 时间前端：下采样发生在哪里

```flow LR
投影特征 B × T_pad × 256 -> 转为 B × 256 × T_pad
转为 B × 256 × T_pad -> Conv1d 第一层
Conv1d 第一层 -> GroupNorm 与 GELU 第一组
GroupNorm 与 GELU 第一组 -> Conv1d 第二层
Conv1d 第二层 -> GroupNorm 与 GELU 第二组
GroupNorm 与 GELU 第二组 -> 转回 B × L × 256
```

r=4 时，两层 Conv1d 均为 256 → 256。
两层均采用 kernel=4、stride=2、padding=1。
每次卷积后的 GroupNorm 都使用 8 组。

| r | TemporalDownsample 的行为 |
| --- | --- |
| 1 | forward 直接返回输入 |
| 2 | 一组卷积、归一化、GELU |
| 4 | 两组，当前配置 |
| 8 | 三组 |

我们的下一步已确定 r=1，因此跳过整个时间卷积前端。
这不会关闭 Transformer 内部的跨时刻注意力。

## D SkipTransformer：9 层与四条长跳连

E1–E4 是四个输入块，M 是中间块。
D1–D4 是四个输出块；它们仍然是 EncoderLayer。
这里没有 cross-attention，也没有单独的 decoder query。

```flow TB
输入特征 -> E1 -> E2 -> E3 -> E4 -> 中间块 M
中间块 M -> 拼接 E4 并投影 -> 输出块 D1
输出块 D1 -> 拼接 E3 并投影 -> 输出块 D2
输出块 D2 -> 拼接 E2 并投影 -> 输出块 D3
输出块 D3 -> 拼接 E1 并投影 -> 输出块 D4
输出块 D4 -> 末尾 LayerNorm
E4 --> 拼接 E4 并投影
E3 --> 拼接 E3 并投影
E2 --> 拼接 E2 并投影
E1 --> 拼接 E1 并投影
```

每次长跳连先沿通道拼接，再用 Linear 512 → 256 融合。
跳连保存的是各输入块的输出，按逆序取回。
这些融合层不会改变时间长度 L。

$$
u_j=D_j\bigl(W_j[\,u_{j-1}\Vert e_{5-j}\,]+b_j\bigr),\quad j=1,2,3,4
$$

其中 u₀ 是中间块输出。四个 W_j 各自独立学习。
长跳连之外，每个 Transformer 块内部还有残差相加。

## E 单个 EncoderLayer：Post-LN

```flow TB
块输入 x -> 全序列自注意力 MHA
全序列自注意力 MHA -> Dropout 第一支
Dropout 第一支 -> 残差相加一
块输入 x --> 残差相加一
残差相加一 -> LayerNorm 一
LayerNorm 一 -> Linear 256 到 1024
Linear 256 到 1024 -> GELU 与内部 Dropout
GELU 与内部 Dropout -> Linear 1024 到 256
Linear 1024 到 256 -> Dropout 第二支
Dropout 第二支 -> 残差相加二
LayerNorm 一 --> 残差相加二
残差相加二 -> LayerNorm 二
```

`normalize_before=false` 选择该 Post-LN 路径。
注意力有 4 个 head，单个 head 的通道维度为 64。
`encode_motion` 未传入因果 mask，因此有效位置之间双向交互。

位置编码已在进入整个堆栈前加到输入。
本调用没有再给每层传入独立 `pos` 参数。

## F 位置、mask 与序列顺序

可学习位置参数的形状为 `[500,1,256]`。
forward 取前 L 个位置，与 `[L,B,256]` 输入相加。
它表达序列索引，不是 diffusion timestep 或真实时间戳。

`lengths` 描述每条样本的有效长度。
有效 latent 长度为 `ceil(length/r)`。
`~latent_mask` 传给注意力，屏蔽 padding key/value。

仅使用 key padding mask 不会自动清零所有 query 输出。
因此采样后还会将 padding 位置的 z 置零。
返回的 `dist` 参数本身没有同步清零。

已确定的相机首版为完整 20 步、r=1。
这时 L=20，且完整样本的 mask 全部有效。

## G 后验头与重参数化采样

256 → 512 的线性层同时预测 μ 和 logvar。
它不是把 latent 通道扩展为 512 后交给下游。
拆分后，两者的最后一维仍然都是 256。

$$
\mathrm{sigma}=\exp(\tfrac{1}{2}\,\mathrm{logvar}),\qquad z=\mathrm{mu}+\mathrm{sigma}\odot\mathrm{epsilon},\qquad \mathrm{epsilon}\sim\mathcal N(0,I)
$$

`encode_motion` 总是调用 `rsample()`，eval 模式也不会自动改取均值。
需要确定性编码时，调用者应明确使用返回分布的 `loc`。

```python src=cma_vae/models/architectures/mld_vae_kl_align.py lines=548-560
```

## H 对 20×11D 相机输入意味着什么

这是下一步的接口推导，代码尚未改动。

| 步骤 | 拟采用的形状 |
| --- | --- |
| 输入 | `[B,20,11]` |
| 相机投影 11 → 256 | `[B,20,256]` |
| r=1，不下采样 | `[B,20,256]` |
| 位置编码与 SkipTransformer | `[20,B,256]` |
| 高斯参数头 | `[20,B,512]` |
| μ、logvar、z | 各为 `[20,B,256]` |

纯 Motion Encoder 不消费图像或 bbox。
逐帧区域特征属于另一条 Vision-Motion Encoder 路径。
当前关注两条编码路径，解码头保持原状。

| 源码 | 核对位置 |
| --- | --- |
| [CMAVAE](../../../cma_vae/models/architectures/mld_vae_kl_align.py) | 73：时间处理；241：编码器定义；497：forward |
| [SkipTransformer 与 EncoderLayer](../../../cma_vae/models/operator/cross_attention.py) | 18：长跳连；236：单块 |
| [位置编码](../../../cma_vae/models/operator/position_encoding.py) | 138：Learned1D；163：构造函数 |
| [模型配置](../../../cma_vae/configs/config_cma_vae.yaml) | model 与 vae_config |
| [配置装配](../../../cma_vae/models/modeltype/mld_kl_align.py) | 43：CMAVAE 实例化 |

依据当前本地源码静态核对，未执行网络 forward。
