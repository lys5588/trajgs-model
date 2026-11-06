# 框架说明页上下文

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

用户指定 answer-me-with-html：扩展 Markdown 经 am CLI 生成单页 HTML。布局、字体、色彩和图坐标由编译器提供，禁止手写 HTML/CSS/SVG。

## Users

当前研究项目成员，在代码改造前理解原版 CMA-VAE 与相机轨迹目标方案。

## Product Purpose

分别展示两个架构，并区分已迁入代码、拟修改部分和后续生成任务。

## Capabilities and Constraints

文本指令、20 个 11D 相机时间步、逐帧图像及 bbox；相机解码头下一步处理。原版代码按目录结构迁入，未完成相机适配。

## Evidence on Hand

仓库 cma_vae/ 原版代码、docs/plan.md、迁入 SHA-256 清单。无相机训练结果。
