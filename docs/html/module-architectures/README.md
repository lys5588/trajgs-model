# 模块架构图册

此目录长期收录 TrajGS 使用的模块和用于对比的参考模块。入口为 [index.html](index.html)。原版分析保留作为基线；新增相机 9D/11D 几何转换与双编码器实现。相机架构 forward/backward 已验证，预训练权重验证等待用户下载。

## 当前条目

| 模块 | 角色 | 页面 | 内容源 |
| --- | --- | --- | --- |
| CLS 与 Patch 融合分析 | 候选分析 | [cls-vs-patch-analysis.html](cls-vs-patch-analysis.html) | [cls-vs-patch-analysis.md](cls-vs-patch-analysis.md) |
| DINOv3 图像处理 | 计算流程详解 | [dinov3-image-flow.html](dinov3-image-flow.html) | [dinov3-image-flow.md](dinov3-image-flow.md) |
| 相机双编码器 | 已实现 | [camera-encoders.html](camera-encoders.html) | [camera-encoders.md](camera-encoders.md) |
| 四个编码模块逐层对照 | 当前源码详解 | [encoder-layer-comparison.html](encoder-layer-comparison.html) | [encoder-layer-comparison.md](encoder-layer-comparison.md) |
| CMA-VAE Motion Encoder | 已迁入基线 | [motion-encoder.html](motion-encoder.html) | [motion-encoder.md](motion-encoder.md) |
| DA3 CameraEnc | 参考模块 | [da3-camera-encoder.html](da3-camera-encoder.html) | [da3-camera-encoder.md](da3-camera-encoder.md) |
| 相机 9D/11D codec | 已实现工具模块 | [camera-pose-codec.html](camera-pose-codec.html) | [camera-pose-codec.md](camera-pose-codec.md) |
| 总览与对比 | 导航 | [index.html](index.html) | [index.md](index.md) |

## 统一记号

- B：batch 大小；T：动作/相机轨迹时间步；S：DA3 输入视图数。
- D：原始输入特征维度；d：DA3 token 通道维度。
- r：时间压缩率；L：CMA-VAE latent 序列长度。
- DA3 模块内没有时间位置编码，因此不能直接把视图索引 S 当成已经编码的时间。
- 相机新实现采用 T=20、D=9/11、r=1；原版分析页面保留 D=269、r=4。

## 后续模块条目

原版 Vision-Motion Encoder 在四模块逐层对照页中展开。新 bbox/DINOv3 与 Camera Encoder 已在相机双编码器页收录。后续模块按实际实现继续增加。

每个条目包含：模块职责、输入输出、层级结构、残差与注意力、位置与 mask、默认参数与版本差异、源码证据和实施状态。网络架构与方案建议明确区分。

## 重建

从项目根目录执行。am 路径是当前开发机的技能安装路径。

```bash
node /root/.codex/skills/answer-me-with-html/scripts/am.mjs render docs/html/module-architectures/index.md --no-open -o docs/html/module-architectures/index.html
node /root/.codex/skills/answer-me-with-html/scripts/am.mjs render docs/html/module-architectures/motion-encoder.md --no-open -o docs/html/module-architectures/motion-encoder.html
node /root/.codex/skills/answer-me-with-html/scripts/am.mjs render docs/html/module-architectures/da3-camera-encoder.md --no-open -o docs/html/module-architectures/da3-camera-encoder.html
```

页面使用既有 shadcn/doc 主题。修改 Markdown 后重新生成，不直接修改 HTML/CSS/SVG。源码摘录由 am 从工作区文件读取；页面包含公开上游代码片段。

来源版本与 SHA-256 见 [sources.json](sources.json)。页面描述的是这个源码快照；后续改动模型时需同步重读源码并更新图册。

## 初版图册验证

- 三页均由 am 成功生成，STE 警告为 0。
- 已检查桌面 1440×1000 与手机 390×844，共六个页面组合。
- 无页面脚本错误、无整页横向溢出；图表展开与 Escape 关闭通过。
- 已目视检查主流程、四条长跳连、Post-LN 与 Pre-LN 块。
- 页面与 Markdown 的本地链接通过存在性检查。
- 54 个迁入文件与原始迁入清单一致，分析所引用的 DA3 文件哈希也保持不变。
- 此验证只覆盖说明页和源码未改动，不代表网络 forward 已跑通。

## 相机 codec 增量

`cma_vae/utils/camera.py` 已实现两种几何表示的双向转换；数值测试在 `tests/test_camera_codec.py`。图册原有的源码分析验证记录仍指前三张页面。

重建新增页面：

```bash
node /root/.codex/skills/answer-me-with-html/scripts/am.mjs render docs/html/module-architectures/camera-pose-codec.md --no-open -o docs/html/module-architectures/camera-pose-codec.html
```

## 四模块逐层对照增量

新增页按当前源码展开 DA3 CameraEnc、CMA-VAE 输入 Linear、Vision-Motion Encoder 和 Motion Encoder。标明了 QKV 投影、head 维度、FFN、归一化、残差、九层长跳连、视觉卷积与后验头。使用 T=20 / S=20 作为形状示例；CMA-VAE 保留当前 D=269、r=4，未来相机接口单独标注。

单图视觉分支的实际输入为 [B,1,H,W,3]，不是逐帧 bbox 特征。没有修改任何模型层、配置或解码头。

重建新增页面：

```bash
node /root/.codex/skills/answer-me-with-html/scripts/am.mjs render docs/html/module-architectures/encoder-layer-comparison.md --no-open -o docs/html/module-architectures/encoder-layer-comparison.html
```

此页及更新后的图册入口通过桌面 1440×1000 和手机 390×844 检查：无脚本错误或整页横向溢出，8 张层级图均可展开和关闭。已目视核对全部 8 个面板；54 个迁入文件与 19 个源码快照哈希一致。验证记录见 [encoder-layer-verification.json](encoder-layer-verification.json)。


## 相机双编码器增量

相机实现与数值验证见 [camera-encoder-verification.json](../../camera-encoder-verification.json)。29 项测试与 9D/11D 官方 DINOv3 架构的 CUDA 验证通过；使用显式随机权重，预训练验证待用户下载文件。新增页面保留相同主题与渲染器。

```bash
node /root/.codex/skills/answer-me-with-html/scripts/am.mjs render docs/html/module-architectures/camera-encoders.md --no-open -o docs/html/module-architectures/camera-encoders.html
```


## DINOv3 图像处理详解

[图像处理详解](dinov3-image-flow.html) 展开 PatchEmbed 的局部投影、196 个 patch、CLS/storage token、Transformer 交互与项目取用的 768 维特征。不修改模型或特征汇聚方式。

```bash
node /root/.codex/skills/answer-me-with-html/scripts/am.mjs render docs/html/module-architectures/dinov3-image-flow.md --no-open -o docs/html/module-architectures/dinov3-image-flow.html
```


## CLS 与 Patch 融合分析

[分析页](cls-vs-patch-analysis.html) 区分裁剪内部布局、原图构图和三维几何；比较当前 CLS 通道拼接、完整时空 token 及相机条件 patch 汇聚。候选设计均未实施，本次不修改模型或配置。

```bash
node /root/.codex/skills/answer-me-with-html/scripts/am.mjs render docs/html/module-architectures/cls-vs-patch-analysis.md --no-open -o docs/html/module-architectures/cls-vs-patch-analysis.html
```
