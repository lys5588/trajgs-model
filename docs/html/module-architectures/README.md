# 模块架构图册

此目录长期收录 TrajGS 使用的模块和用于对比的参考模块。入口为 [index.html](index.html)。网络架构页记录源码分析；新增相机 9D/11D 几何转换模块。当前未调整模型通道、配置、数据读取或解码头，也未运行网络 forward。

## 当前条目

| 模块 | 角色 | 页面 | 内容源 |
| --- | --- | --- | --- |
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
- 本轮用户已确定后续相机实现采用 T=20、D=11、r=1；当前源码仍使用原版参数。

## 后续模块条目

Vision-Motion Encoder 已在四模块逐层对照页中展开。bbox 区域视觉编码器和最终采用的 Camera Encoder 在分析或实现时各自新增页面。尚未分析的模块不使用占位网页冒充已完成条目。

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

## 本次验证

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
