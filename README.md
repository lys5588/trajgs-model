# TrajGS

根据拍摄任务指令与场景参考图像生成相机轨迹。当前包含原样迁入的 CMA-VAE 基线、9D/11D 相机双编码器、相机几何 codec 和架构图册。指令条件生成与训练 loss 尚未接入。

- [相机 9D / 11D 转换](docs/html/module-architectures/camera-pose-codec.html)：原始相机参数与两种表示的双向转换、接口和测试。
- [相机双编码器](docs/html/module-architectures/camera-encoders.html)：逐帧 bbox、冻结 DINOv3、9D/11D 通道与验证状态。
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

四个上游仓库登记为 Git 子模块，主仓库只记录地址和固定提交。仅查看文档和 HTML 不需要下载子模块；需要上游代码时执行：

```bash
git submodule update --init --recursive
```

也可以首次直接使用 `git clone --recurse-submodules https://github.com/lys5588/trajgs-model.git`。

| 子模块 | 来源 | 说明 |
| --- | --- | --- |
| `submodules/Unimotion` | 本仓库的 `reference/unimotion-bootstrap` 分支 | 保留从官方 UniMotion 初始化后的环境配置提交；固定版本本身没有递归子模块 |
| `submodules/G2VLM` | `InternRobotics/G2VLM` | 固定在当前参考版本 |
| `submodules/Depth-Anything-3` | `ByteDance-Seed/Depth-Anything-3` | 固定在当前参考版本 |
| `submodules/DINOv3` | `facebookresearch/dinov3` | 相机视觉路径的运行依赖，ViT-B/16 |

具体 SHA 由 Git gitlink 记录，可用 `git submodule status` 查看。根目录 `cma_vae/` 是独立迁入的基线，用于后续相机适配。

## 相机双编码器 forward

从仓库根目录运行，使用已有 trajgs Python 环境（或包含 torch、numpy、omegaconf 的环境）。

```bash
git submodule update --init submodules/DINOv3
.local/miniforge3/envs/trajgs/bin/python -m cma_vae.smoke_camera_encoders --camera_dim 9
.local/miniforge3/envs/trajgs/bin/python -m cma_vae.smoke_camera_encoders --camera_dim 11
```

默认配置为 `cma_vae/configs/config_camera_vae.yaml`。上述命令要求官方 **DINOv3 ViT-B/16 LVD-1689M PyTorch .pth** 权重位于：

```text
/mnt/d/model/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

可用 `--dinov3_weights /实际路径/checkpoint.pth` 覆盖。下载入口为 [Meta 官方 DINOv3](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)。模型不会隐式下载或回退到随机权重。当前权重由用户下载，预训练验证待完成；9D/11D 已用官方真实架构的显式随机权重通过 CUDA forward/backward 和解码形状检查。

如只验证架构，可显式添加 `--random_backbone`；报告会标记 `pretrained_validation: NOT_RUN`。用 `--output /tmp/camera-smoke.json` 保存报告。

```bash
.local/miniforge3/envs/trajgs/bin/python -m unittest discover -s tests -p 'test_camera*.py' -v
```

模型入口为 `CameraCMAVAE(camera_dim=9|11)`。`encode_camera`、`encode_joint` 返回序列优先的 latent 与 Normal 后验；`forward` 只编码，`decode` 显式调用。完整接口、缺图规则与测试记录见 [计划文档](docs/plan.md) 和 [验证记录](docs/camera-encoder-verification.json)。
