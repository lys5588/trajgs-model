# TrajGS

根据拍摄任务指令与场景参考图像生成相机轨迹。当前包含原样迁入的 CMA-VAE 基线、相机任务设计文档和框架对比页。

- [三模块计划](docs/plan.md)：数据读取、模型、训练。
- [CMA-VAE 迁入说明](cma_vae/IMPORT.md)：来源版本及后续适配边界。
- [原版与相机目标框架对比](docs/html/cma-vae-camera-framework/index.html)：下载后用浏览器打开；GitHub 文件页显示 HTML 源码。

## 获取代码与查看页面

```bash
git clone https://github.com/lys5588/trajgs-model.git
cd trajgs-model
```

用浏览器打开 `docs/html/cma-vae-camera-framework/index.html` 即可查看对比页，无需模型环境或服务器。

已有这个仓库的本地副本时，在其目录下执行：

```bash
git pull --ff-only origin main
```

## 参考仓库

三个参考仓库登记为 Git 子模块，主仓库只记录地址和固定提交。仅查看文档和 HTML 不需要下载子模块；需要参考代码时执行：

```bash
git submodule update --init --recursive
```

也可以首次直接使用 `git clone --recurse-submodules https://github.com/lys5588/trajgs-model.git`。

| 子模块 | 来源 | 说明 |
| --- | --- | --- |
| `submodules/Unimotion` | 本仓库的 `reference/unimotion-bootstrap` 分支 | 保留从官方 UniMotion 初始化后的环境配置提交；固定版本本身没有递归子模块 |
| `submodules/G2VLM` | `InternRobotics/G2VLM` | 固定在当前参考版本 |
| `submodules/Depth-Anything-3` | `ByteDance-Seed/Depth-Anything-3` | 固定在当前参考版本 |

具体 SHA 由 Git gitlink 记录，可用 `git submodule status` 查看。根目录 `cma_vae/` 是独立迁入的基线，用于后续相机适配。
