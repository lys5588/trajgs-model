# CMA-VAE 相机框架对比页

本任务目录保存内容源文件与生成结果。入口为 `index.html`，内容源为 `explanation.md`，研究计划为 `../../plan.md`。

页面模式为 Read：先确认交付状态，再阅读原版与目标两张架构图，最后检查差异、训练关系与后续生成路径。使用 answer-me-with-html 的现成 shadcn/doc 视觉系统；impeccable 用于信息层级、状态真实性和桌面/窄屏可读性检查。本任务未建立独立品牌或自定义 CSS。

设计约定：

- THESIS：原版保留，目标方案清晰标为待实现。
- OWN-WORLD：沿用 am 的 shadcn 文档主题，不引入新视觉系统。
- STORY：输入 → 原版 → 目标 → 差异 → 损失 → 生成边界 → 文件。
- FIRST VIEWPORT：交付状态、任务输入与形状。
- FORM：用户指定的 Markdown 编译输出，两个独立架构图；属于明确限定的文档任务。

在项目根目录重建：

```bash
node /root/.codex/skills/answer-me-with-html/scripts/am.mjs render docs/html/cma-vae-camera-framework/explanation.md --no-open -o docs/html/cma-vae-camera-framework/index.html
```

生成器路径为当前开发机安装路径，其他机器请替换为技能安装位置。不要直接修改 `index.html`；修改内容源后重新生成。

## 本次验证

- 54 个迁入原文件的 SHA-256 与来源一致；49 个 Python 文件语法解析通过。
- 上游工作目录保持 clean；运行缓存未迁入。
- plan、迁入说明和页面源稿中的本地文档链接均存在。
- am 渲染成功：7 个内容区、4 张流程图、1 个目录图；STE 警告为 0。
- Chromium 实测 1440×1000 与 390×844，均无脚本错误和整页横向溢出。
- 手机端图表展开与 Escape 关闭操作通过；宽图可在图表容器内滚动。
- 未执行模型 forward 或训练；上述检查不代表相机模型已经实现。

视觉检查使用临时浏览器依赖，未修改模型 requirements 或系统软件安装。
