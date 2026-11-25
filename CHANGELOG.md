# 更新日志

## 0.2.0 - 2026-11-25

### 新增

- 增加八类语义材质和真机、CAD 两套外观，支持通过 Python API 为同一场景的不同夹爪分别选色。
- 提供五张预览和坐标系、运动学、质量预算三张 300 dpi PNG 出版图，以及查询配置、构建和出图的命令行接口。
- 增加跨格式、物理基线、扩展模组与出图回归，接入 Ruff、Pyright、pre-commit 和双环境 CI。

### 变更

- 按真机照片校正配色和曲面法线，改善闭合序列、触觉局部轴及出版图图例。
- 统一 Python 与 ROS 包元数据，重排上手及集成文档。

### 修复

- 隔离宿主与模组材质命名，正确处理继承视觉组和扩展模组的 CAD 配色。
- 物理基线覆盖重复分区；固定出图环境时间，使出版图可逐字节复现。

## 0.1.0 - 2026-11-24

- 从 Onshape `Adaptive_Tactile_Gripper`（`B1` 分支）初始化独立夹爪描述仓库。
- 完成 `ament_cmake`（ROS 2 Jazzy）与 `hatchling`（Python wheel）双打包结构。
- 将夹爪底盘（DM-J4310P-2EC + 曲柄滑块 + Franka 法兰转接板）、指尖传感器（`pts_2_0`、`flat_pad`）与侧挂相机（`realsense_d435i`）解耦为可替换模块。
- 使用实测 686 g 整机总质量约束，在打印件均匀填充假设下积分估计各刚体惯量，并输出固定基座、解析闭环构型的 Franka Desk 负载表。
- 对齐 `realsense2_description` / `mujoco_menagerie` 的 `D435i` 标准外参 TF 树与双相机模型，以及 `PTS 2.0` 在 MJCF/URDF 中的 9 球体碰撞几何。
- 修复 `onshape-to-robot` 因左连杆 `CYLINDRICAL` 配合缺失的闭环约束，验证 MuJoCo 闭环与理论运动学误差小于 `0.5 mm`，以及双侧 3×3 触觉法向力正方向投影。
- 从 MJCF 位姿生成 URDF，修复右指、传感器和电机网格坐标。为转子与连杆保留独立刚体，统一跨格式质量属性与运动学。
- 修正 D435i 主惯量轴向，增加 accel 与 gyro 本体及光学帧，并支持真机驱动发布设备标定外参。
- 恢复整机 MJCF 的 `freejoint` 写法，移除被动关节附加惯量，补齐非触觉结构碰撞，区分协议角度、机构极限和模组接触界限。
- 修复 SRDF 被动关节层级和末端成员，增加单自由度规划组，并补充网格摘要、触觉布局接口与跨格式回归测试。
- 将 Python 依赖收紧为 `mujoco>=3.10.0,<5`，以支持模块装配使用的 `MjSpec` 接口。
- 同步 Onshape `B1`（微版本 `63ba637a284b7af6e2491519`）相机支架改版：将 RealSense D435i 沿 `+X` 平移 `+12.5 mm`（RGB 光心移至 `X = -20.0 mm`，Depth 光心移至 `X = -5.0 mm`），并将 `tools/build_description.py` 升级为从 STL 顶点动态生成 `camera_mount_collision` 包围盒与从主文档 `.part` 自动读取 `microversion_id`。
