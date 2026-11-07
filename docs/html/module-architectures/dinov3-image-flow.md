---
title: 一张图怎样变成 DINOv3 的 768 维特征？
subtitle: 从 224×224 像素到 196 个 patch，再到我们选择的 CLS
lang: zh
template: doc
theme: shadcn
mode: light
file: dinov3-image-flow
---
最终每张裁剪图使用一个 768 维向量，这是我们选择的 CLS 输出。
PatchEmbed 先产生 196 个 768 维向量，Transformer 再让它们交换信息。

以下使用当前 ViT-B/16 配置，先省略 batch 维。
这里的“整张图”指 bbox 裁剪并拉伸后的目标图像。

## A 一张图的完整形状变化

```flow TB
bbox 目标区域 -> 拉伸与标准化\n3 × 224 × 224
拉伸与标准化\n3 × 224 × 224 -> PatchEmbed Conv2d\n768 × 14 × 14
PatchEmbed Conv2d\n768 × 14 × 14 -> 把网格排列为 token 序列\n196 × 768
把网格排列为 token 序列\n196 × 768 -> 加入 CLS 和 4 个辅助 token\n201 × 768
加入 CLS 和 4 个辅助 token\n201 × 768 -> 12 层 Transformer 与末尾 LayerNorm\n201 × 768
12 层 Transformer 与末尾 LayerNorm\n201 × 768 -> CLS 输出\n1 × 768
12 层 Transformer 与末尾 LayerNorm\n201 × 768 -> Patch 输出\n196 × 768
12 层 Transformer 与末尾 LayerNorm\n201 × 768 -> Storage 输出\n4 × 768
CLS 输出\n1 × 768 -> 当前项目取出 CLS\n每张图一个 768 维向量
```

| 数字 | 对应含义 |
| --- | --- |
| 3 | RGB 三个颜色通道 |
| 224×224 | 拉伸后图像的像素网格 |
| 16×16 | 每个 patch 的像素尺寸 |
| 14×14=196 | 图像中的 patch 数量 |
| 768 | 每个 token 的特征宽度 |
| 201 | 196 patch + 1 CLS + 4 storage |
| 12 | DINOv3 ViT-B/16 的 Transformer 层数 |

768、196、201 描述不同维度，不能互相替代。

## B PatchEmbed · 一个 patch 如何变成向量

你的理解中，“通道从 3 变成 768、宽高减少”是正确的。
需要补充：每个输出位置覆盖原图的一个 16×16 区域。

实际卷积为 `Conv2d(3,768,kernel_size=16,stride=16)`。
它使用 768 个卷积核，每个核覆盖三个颜色通道和 16×16 像素。

```flow LR
一个局部 patch\n3 × 16 × 16 -> 展平成局部像素向量\n768 个数
展平成局部像素向量\n768 个数 -> 共享可学习投影\nW 为 768 × 768
共享可学习投影\nW 为 768 × 768 -> 一个 patch token\n768 个特征值
```

上图的展平是数学等价解释。源码直接通过 Conv2d 完成计算。

| 计算 | 含义 |
| --- | --- |
| 单个输出通道 | 对整个 3×16×16 patch 加权求和，再加 bias |
| 所有输出通道 | 同一个 patch 得到 768 个不同的加权结果 |
| 卷积权重 | [768,3,16,16]；bias 为 [768] |
| stride=16 | 窗口每次移动 16 像素，初始 patch 不重叠 |
| 输出空间 | 每边 224÷16=14 个位置 |
| 权重共享 | 196 个位置使用同一组卷积权重 |

它包含空间像素混合，不是对每个像素单独做 3→768 的 1×1 投影。
也不是先把图片插值缩小成 14×14，再扩展颜色通道。

这里 3×16×16 恰好等于 768。
输出宽度 768 仍由模型配置决定，不是由这个乘法强制决定。
因此本层虽然减少空间网格，但本配置下输出元素总数没有减少。

源码中的 PatchEmbed 还返回 [N,14,14,768]，保留二维网格。
prepare_tokens 随后将它展平为 [N,196,768]。
这一布局转换不汇聚 patch，也不改变元素数。

## C CLS 是什么 · 为什么会有 201 个 token

进入 Transformer 前，序列按以下顺序组成：

```flow LR
CLS\n可学习初始向量 1×768 -> Storage\n4 个辅助向量
Storage\n4 个辅助向量 -> Patch tokens\n196 个局部图像向量
```

上图表示序列排列顺序，不表示 token 之间的计算流程。

| token 类型 | 从哪里来 | 进入 Transformer 后 |
| --- | --- | --- |
| CLS | 一个可学习参数，扩展到每张图 | 通过注意力从当前图像汇聚信息 |
| 4 个 storage | 四个可学习辅助参数 | 与其他 token 一起参与计算 |
| 196 个 patch | PatchEmbed 的图像投影 | 在局部内容上加入其他区域的上下文 |

CLS 初始值不是这张图的像素平均值。
不同图像从同一个 CLS 参数开始，但注意力读取的图像内容不同。
经过 Transformer 后，各图像的 CLS 才变成各自的表示。

storage token 不对应四块图像，也不是四个相机时刻。
本项目不把它们作为视觉输出。

## D Transformer · 从局部像素到图像上下文

PatchEmbed 的单个 token 起初只读取自己的 patch。
Self-attention 让同一张图的所有 201 个 token 交换信息。
CLS 也参与其中，能够从图像 token 中读取内容。

```flow TB
输入所有 token\n201 × 768 -> norm1 与 self-attention\n12 heads · 每头 64 维
norm1 与 self-attention\n12 heads · 每头 64 维 -> LayerScale 与第一次残差相加\n201 × 768
输入所有 token\n201 × 768 --> LayerScale 与第一次残差相加\n201 × 768
LayerScale 与第一次残差相加\n201 × 768 -> norm2 与 FFN\n768 → 3072 → 768
norm2 与 FFN\n768 → 3072 → 768 -> LayerScale 与第二次残差相加\n201 × 768
LayerScale 与第一次残差相加\n201 × 768 --> LayerScale 与第二次残差相加\n201 × 768
LayerScale 与第二次残差相加\n201 × 768 -> 下一个 Transformer Block\n总共 12 层
```

| 运算 | 作用 |
| --- | --- |
| 2D RoPE | 为 patch 的二维空间关系提供位置信息 |
| QKV 投影 | 每个 token 由 768 映射到 2304，再拆出 Q、K、V |
| 注意力 | 每个 head 在 201 个 token 之间计算内容相关的权重 |
| CLS 的注意力输出 | 根据其注意力权重组合其他 token 的信息 |
| FFN | 对每个 token 的通道做非线性变换 |
| 残差连接 | 将分支计算结果加回原来的 token 表示 |

经过每个 Block，token 数量仍是 201，宽度仍是 768。
Transformer 没有在这里把 196 个 patch 逐层减少成 1 个。

加载预训练权重后，这些学习到的变换用于提取图像表示。
768 只是向量宽度；并不表示 768 个类别或预先命名的属性。

## E 为什么我们最后只有一个 768 维向量

DINOv3 的 forward_features 同时返回多种表示。
当前代码只读取 `x_norm_clstoken`。

| 返回项 | 单图形状 | 当前项目 |
| --- | --- | --- |
| x_norm_clstoken | [768] | 取用 |
| x_norm_patchtokens | [196,768] | 本版不传给相机联合编码器 |
| x_storage_tokens | [4,768] | 不取用 |

这个过程是“经过交互后的 CLS → 取出第一个 token”。
没有对 196 个 patch 做 mean pooling。
也没有把 196×768 展平后用一个巨大 Linear 压成 768。

```python src=cma_vae/models/architectures/camera_vision.py lines=134-144
```

这里的 norm 指模型中的 LayerNorm。
当前没有额外做 L2 单位向量归一化。

CLS 是整张目标裁剪图的聚合表示。
Patch tokens 保留 14×14 的空间排列，并已包含上下文。
选择 CLS 后，下游不再直接看到这张 14×14 特征网格。
冻结骨干仅停止参数更新，不会停止其注意力计算。

## F 回到 20 帧相机序列

令 N=B×20。这个 N 是图像 batch 数，不是 Transformer 的 token 数。

```flow TB
完整裁剪序列\nB × 20 × 3 × 224 × 224 -> 展平图像 batch\nN × 3 × 224 × 224
展平图像 batch\nN × 3 × 224 × 224 -> DINOv3 内部\nN × 201 × 768
DINOv3 内部\nN × 201 × 768 -> 每张图选择 CLS\nN × 768
每张图选择 CLS\nN × 768 -> 恢复时间轴\nB × 20 × 768
恢复时间轴\nB × 20 × 768 -> 视觉 Linear 768 → 256\nB × 20 × 256
相机 Linear 与时间 PE\nB × 20 × 256 -> 同一时刻通道拼接\nB × 20 × 512
视觉 Linear 768 → 256\nB × 20 × 256 -> 同一时刻通道拼接\nB × 20 × 512
同一时刻通道拼接\nB × 20 × 512 -> 联合时序 Transformer\n20 个时间 token
```

| 两个 Transformer 阶段 | 交互范围 |
| --- | --- |
| DINOv3 | 同一张裁剪图内的 201 个 token；不同帧独立 |
| Vision-Camera Encoder | 同一轨迹的 20 个融合 token；跨时刻交互 |

当前 B=2 时有 40 张裁剪图。
DINOv3 分别编码这 40 张图，不把它们串成一个大注意力序列。
每张图得到 768 维 CLS，再投影成相机网络使用的 256 维视觉特征。

来源为本地固定版本源码：
[PatchEmbed](../../../submodules/DINOv3/dinov3/layers/patch_embed.py)、
[token 组织与输出](../../../submodules/DINOv3/dinov3/models/vision_transformer.py)、
[ViT-B/16 配置](../../../submodules/DINOv3/dinov3/hub/backbones.py)、
[项目视觉包装器](../../../cma_vae/models/architectures/camera_vision.py)。

本页只解释当前计算流程，不修改模型。
[返回相机双编码器](camera-encoders.html) · [返回图册](index.html)
