---
title: 相机表示 · 9D / 11D 双向转换
subtitle: 已实现的几何模块，尚未接入网络通道
lang: zh
template: doc
theme: shadcn
mode: light
file: camera-pose-codec
---
两种表示都保存相机中心和两个 FOV。9D 使用 XYZW 四元数；11D 使用旋转矩阵的前两列。

[模块图册](index.html) · [转换代码](../../../cma_vae/utils/camera.py) · [数值测试](../../../tests/test_camera_codec.py)

## A 编码内容与索引

| 内容 | quaternion_9d | rotation_6d_11d |
| --- | --- | --- |
| 相机中心 c | `[0:3]`，C2W 平移 | `[0:3]`，C2W 平移 |
| 旋转 | `[3:7]`，qx、qy、qz、qw | `[3:6]` 第一列；`[6:9]` 第二列 |
| 垂直 FOV | `[7]` | `[9]` |
| 水平 FOV | `[8]` | `[10]` |
| 总维度 | 3＋4＋2＝9 | 3＋6＋2＝11 |

四元数之外的五个数是相机中心 3D 和 FOV 2D。
它们不是五个内参。6D 也只表示旋转，不包含外参平移。

## B 接口与张量约定

| 项目 | 本项目接口 |
| --- | --- |
| 编码函数 | `extri_intri_to_pose_encoding` |
| 解码函数 | `pose_encoding_to_extri_intri` |
| 原始 extrinsics | `[B,S,3,4]` 或 `[B,S,4,4]` |
| 原始 intrinsics | `[B,S,3,3]`，像素单位 |
| image_size_hw | `(H,W)`、`[2]`、`[B,2]` 或 `[B,S,2]` |
| pose_encoding | `[B,S,9]` 或 `[B,S,11]` |
| 逆向返回 | 二元组：extrinsics `[B,S,3,4]`、intrinsics `[B,S,3,3]` |
| 默认外参约定 | 输入和返回均为 W2C |
| 几何与输出类型 | torch.float32，与输入设备一致 |
| 时间轴 | 保留 S；当前任务 S=20 |

`extrinsics_type="c2w"` 可显式切换输入或返回约定。
两种表示内部始终使用 C2W 的中心与旋转。

DA3 的同名底层函数接受 C2W。
DA3 的 CameraEnc 在调用它前先将 W2C 求逆。
本接口把这一步合并到默认 W2C 路径，避免调用者漏掉求逆。

## C 正向计算

```flow TB
原始 W2C 外参 -> 刚体求逆得到 C2W
刚体求逆得到 C2W -> 相机中心 c
刚体求逆得到 C2W -> 旋转矩阵 Q
旋转矩阵 Q -> 四元数 XYZW
旋转矩阵 Q -> Q 的前两列按列展开
像素内参与有效 H W -> 垂直与水平 FOV
相机中心 c -> 拼接 9D
四元数 XYZW -> 拼接 9D
垂直与水平 FOV -> 拼接 9D
相机中心 c -> 拼接 11D
Q 的前两列按列展开 -> 拼接 11D
垂直与水平 FOV -> 拼接 11D
```

```text
C2W rotation = R_w2c.T
camera center = -R_w2c.T @ t_w2c
fov_h = 2 * atan((H / 2) / fy)
fov_w = 2 * atan((W / 2) / fx)
```

使用图像有效尺寸，不包含额外 padding。
resize/crop 后必须传入同步更新的像素内参。
模块不会自动归一化首帧、尺度或坐标系。

## D 逆向构造

```flow TB
9D 的四元数 -> 归一化旋转表示 -> C2W 旋转矩阵
11D 的两根轴 -> Gram-Schmidt 正交化 -> C2W 旋转矩阵
相机中心 -> 构造 C2W 外参
C2W 旋转矩阵 -> 构造 C2W 外参
构造 C2W 外参 -> 默认转回 W2C
两个 FOV 与有效 H W -> 恢复 fx 与 fy
恢复 fx 与 fy -> 构造中心主点内参 K
```

11D 先固定第一根轴，再去除第二根轴的平行分量。
第三根轴由叉积得到，三根轴作为旋转矩阵的列。
实现对第二根输入轴预归一化，并清理剩余正交误差。

零向量、共线向量和数值退化输入会明确报错。
有效非正交预测可构造成右手正交旋转。

```text
fy = (H / 2) / tan(fov_h / 2)
fx = (W / 2) / tan(fov_w / 2)
cx = W / 2
cy = H / 2
skew = 0
```

FOV 须已是弧度值，严格位于 (0,pi)。
此模块不把 logits 当成 FOV，也不隐式执行 sigmoid。

## E 如何调用

在项目根目录使用以下接口。
`extrinsics`、`intrinsics` 是已读取的 batch。

```python title="9D 与 11D · 调用示例"
from cma_vae.utils.camera import (
    extri_intri_to_pose_encoding,
    pose_encoding_to_extri_intri,
)

pose9 = extri_intri_to_pose_encoding(
    extrinsics, intrinsics, (480, 640),
    pose_encoding_type="quaternion_9d",
)
pose11 = extri_intri_to_pose_encoding(
    extrinsics, intrinsics, (480, 640),
    pose_encoding_type="rotation_6d_11d",
)

# 按末维 9 / 11 自动识别表示；默认返回 W2C。
extrinsics9, intrinsics9 = pose_encoding_to_extri_intri(
    pose9, (480, 640),
)
extrinsics11, intrinsics11 = pose_encoding_to_extri_intri(
    pose11, (480, 640),
)

# 若下游需要 DA3 底层函数的 C2W 输出约定：
c2w, intrinsics_out = pose_encoding_to_extri_intri(
    pose11, (480, 640), extrinsics_type="c2w",
)
```

编码默认选择 9D；解码默认按末维识别。
也可在解码时显式传入 `pose_encoding_type`，检查维度是否匹配。

## F 往返边界与验证

| 往返对象 | 可以恢复什么 |
| --- | --- |
| 合法位姿 | 恢复旋转与平移，容许 FP32 数值误差 |
| fx、fy | 在相同有效 H、W 下恢复 |
| 任意主点与 skew | 不保存；逆向构造中心主点与零 skew |
| 4×4 输入 | 返回 3×4；最后的齐次行由调用方补齐 |
| 已归一化世界坐标 | 保持传入的坐标系；不会恢复未保存的尺度或首帧变换 |
| 任意 6D 预测 | 正交化是多对一映射，不恢复原始六个数 |

验证覆盖附件示例、DA3 数值对照与 B=2、S=20 的往返。
还覆盖单位旋转、180 度旋转、无效输入和近共线旋转。
17 项测试全部通过，包含 CPU/CUDA 的 autocast 与反向梯度。

构造的内参可反向传播，FOV 可获得梯度。
这只证明转换可微，不代表未来训练损失已经接入 FOV 监督。

```bash title="运行数值测试"
.local/miniforge3/envs/trajgs/bin/python -m unittest discover -s tests -p 'test_camera_codec.py' -v
```

下一步再调整网络输入通道和相机特征的接入位置。
本轮未修改任何已有 encoder、decoder 或模型配置。
