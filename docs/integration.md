# 下游接入与自定义传感器扩展指南

本指南说明如何将 `adaptive_tactile_gripper_description` 接入 `multipanda_ros2`（真机 Panda + MoveIt 2）、`tactile_grasp`（MuJoCo 触觉控制仿真），以及如何扩展新的可替换指尖传感器模组。

---

## 1. 接入 `multipanda_ros2`（ROS 2 Jazzy + MoveIt 2 + 真机 Panda）

### 1.1 工作区构建

将本仓库放入 `multipanda_ros2` 所在 ROS 2 工作区的 `src/`（或通过 `.repos` 拉取），随后构建：

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select adaptive_tactile_gripper_description \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

> 若终端激活了额外的 Python 虚拟环境（缺少 `catkin_pkg`），加上 `-DPython3_EXECUTABLE=/usr/bin/python3` 可确保 `ament_cmake` 使用系统 ROS Python。

独立预览测试 Xacro 展开：

```bash
xacro "$(ros2 pkg prefix adaptive_tactile_gripper_description)/share/adaptive_tactile_gripper_description/urdf/gripper_standalone.urdf.xacro" \
  left_sensor:=pts_2_0 right_sensor:=pts_2_0 with_camera:=true safety_distance:=0.02
```

### 1.2 在 Panda URDF 与 SRDF 中挂载

在 `franka_description/robots/real/panda_arm.urdf.xacro`（或仿真 Xacro）中，将夹爪挂载到 `$(arg arm_id)_link8`：

```xml
<xacro:arg name="left_sensor" default="pts_2_0"/>
<xacro:arg name="right_sensor" default="pts_2_0"/>
<xacro:arg name="with_camera" default="true"/>
<xacro:arg name="use_nominal_extrinsics" default="false"/>

<xacro:include filename="$(find adaptive_tactile_gripper_description)/urdf/gripper.xacro"/>
<xacro:adaptive_tactile_gripper
    ns="$(arg arm_id)_gripper"
    connected_to="$(arg arm_id)_link8"
    xyz="0 0 0"
    rpy="0 0 0"
    left_sensor="$(arg left_sensor)"
    right_sensor="$(arg right_sensor)"
    with_camera="$(arg with_camera)"
    use_nominal_extrinsics="$(arg use_nominal_extrinsics)"
    safety_distance="0.02"/>
```

> **安装偏航角提示**：原厂 Franka 手爪在 `panda_link8` 上常使用 `rpy="0 0 ${-pi/4}"` 对齐定位销。上机装配后，请对照真机上侧挂 RealSense D435i 与双指开合方向在 RViz2 中核对一次 `rpy` 的绕 Z 旋转角。

在对应的 MoveIt 2 SRDF（如 `franka_moveit_config/srdf/panda_arm.srdf.xacro`）中引入免碰撞与末端规划组：

```xml
<xacro:include filename="$(find adaptive_tactile_gripper_description)/srdf/gripper.srdf.xacro"/>
<xacro:adaptive_tactile_gripper_srdf
    arm_id="$(arg arm_id)"
    ns="$(arg arm_id)_gripper"
    with_camera="$(arg with_camera)"
    use_nominal_extrinsics="$(arg use_nominal_extrinsics)"
    left_sensor="$(arg left_sensor)"
    right_sensor="$(arg right_sensor)"/>
```

夹爪的 `panda_gripper_tcp` 位于法兰底面前方 145.55 mm。相机本体 `d435i_link` 及各非光学帧采用 **+X 前、+Y 左、+Z 上**，所有 `*_optical_frame` 采用 **+Z 前、+X 右、+Y 下**。`use_nominal_extrinsics:=true` 会生成 depth、color、infra1、infra2、accel、gyro 的名义本体帧和光学帧。MJCF 使用同一套名义帧。

真机示例将 `use_nominal_extrinsics` 设为 `false`，URDF 只定义相机的安装螺孔帧和本体帧，标定外参由 RealSense 驱动发布。URDF 与 SRDF 使用相同的开关值。将驱动相机名称设为与本体帧匹配的 `panda_gripper_d435i`，并核对实际消息的 `frame_id`。双发布者使用同名帧会导致标定 TF 跳变。名义 IMU 帧定义见 [RealSense ROS 模型](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/urdf/_d435i_imu_modules.urdf.xacro)。

### 1.3 开合状态同步与 MoveIt 规划

`panda_gripper_left_finger_slide` 是控制和规划输入，单位为米。右滑块通过 URDF mimic 跟随。用 `get_joint_positions(q_drive_rad)` 将真机电机角转为单指位移和左右销轴角。有效曲柄角为驱动角加回差角，回差未知时使用默认零值。将左滑块、`gripper_drive` 及两个 `*_link_crank_pin` 一并发布到 `/joint_states`，各关节名加 `panda_gripper_` 前缀。`gripper_drive` 的 URDF 值取实际曲柄角，两个销轴值直接取函数返回值。

`panda_gripper_motion` 是单自由度规划组，完整的 `panda_gripper` 组包含被动连杆、指尖和安装帧，可用于末端定义与状态显示。URDF 以开链表达刚体，三个被动转动关节依赖上述闭环状态同步。执行侧用滑块位移或电机角控制机构，并将各被动关节状态由闭环关系重算。

电机协议与机构有效曲柄角范围均为 `[0, 1.7] rad`，曲柄到 `1.7 rad`（约 97.4°）时单指闭合 37.553 mm（离曲柄轴套与对侧滑块碰撞极限仍有 1.75 mm 安全余量）。双侧均装 `pts_2_0` 或 `flat_pad`（含混装）时，名义首次接触提前至 **1.450404 rad、33.15 mm**；双侧装 `paxini_m2824`（3.5 mm 转接垫块）时，名义首次接触为 **1.610581 rad、36.05 mm**。用 `get_stroke_limits(left_sensor, right_sensor)` 获取任意组合的无接触上界。SRDF 的 `close` 单指闭合 32 mm（双 PTS/平垫对置净距 2.3 mm，双 `paxini_m2824` 对置净距 8.1 mm）。该净距在 `safety_distance=0.02` 时保持一致，因为余量只加到外部避障包络，指背及传感器壳体/转接座沿闭合轴仅向后扩展。两侧传感器之间始终参与碰撞检查。

### 1.4 Franka Desk 负载核对

无相机双 PTS 整机实测 **686.0 g**。该总质量约束打印件的均匀填充模型，用于估计各部件的质量分布和惯量。含 D435i、支架和相机紧固件约 **777.0 g**。四颗法兰 M6×12 螺钉另计约 **20.5 g**。

`config/specs.yaml` 的 `franka_desk_payload` 记录固定基座、回差角为零的解析闭环构型及残差。质心以法兰基准表达，惯量绕合质心且轴向平行于该法兰基准。若真机安装有平移 `p` 和旋转 `R`，输入 Desk 前将质心换算为 `p + R c`，惯量换算为 `R I Rᵀ`，并将额外螺钉、线缆等质量并入实际总成。

---

## 2. 接入 `tactile_grasp`（MuJoCo 触觉抓取平台）

### 2.1 锁定版本依赖

本描述包声明 `mujoco>=3.10.0,<5`，与当前下游锁定的 3.10.0 有交集。下游双 PTS 配置可以将 `model` 指向任一预生成整机 MJCF。载入 XML 时按原有流程删除 `base_freejoint` 后再挂载到载体或 FR3 法兰。

在 `tactile_grasp/pyproject.toml` 中与 `papillarray-pts` 同样以固定 Git 引用引入（并经 `uv lock` 锁定哈希）：

```toml
dependencies = [
    # 其余已有依赖
    # 仅需 pts_2_0 / flat_pad 及外观预设时，可固定到 v0.2.0 标签：
    "adaptive-tactile-gripper-description @ git+https://github.com/langxin11/adaptive_tactile_gripper_description.git@v0.2.0",
    # 如需使用 paxini_m2824、1.7 rad 行程及下文 add_tactile_contact_pairs / TactileForceReader，
    # 请改用包含本特性的主干提交或后续发布标签（并经 uv lock 锁定提交哈希）：
    # "adaptive-tactile-gripper-description @ git+https://github.com/langxin11/adaptive_tactile_gripper_description.git@main",
]
```

### 2.2 `B1` 新版与旧版 `dm4310` 参数对照

切换到 `B1` 版夹爪时，同步将 `tactile_grasp/configs/control/force_tracking.yaml` 与 `configs/simulation/fr3_dm4310.yaml` 更新为 `config/specs.yaml` 中的 B1 机构几何：

| 参数 | 旧版 (`dm4310`) | 新版 (`B1`) | 单位 |
| --- | --- | --- | --- |
| `geometry_crank_radius_m` | `0.03` | `0.025` | `m` |
| `geometry_link_length_m` | `0.04` | `0.04353553390593274` | `m` |
| `geometry_offset_m` | `0.021213203435596423` | `0.017677669529663688` | `m` |
| `geometry_theta0_rad` | `0.7853981633974483` | `0.7853981633974483` | `rad` |
| `tool_center_in_flange_m` | `[0.0, 0.0, 0.13955]` | `[0.0, 0.0, 0.14555]` | `m` |

双指总闭合雅可比在全开时从旧版的 `0.0424264 m/rad` 变为 B1 的 `0.0353553 m/rad`。力矩到夹持力的换算应使用当前实际曲柄角对应的 `get_closure_jacobian(q_rad)`，同时更新 FR3 的 TCP 偏置（沿法兰 +Z 增加 6 mm）。

### 2.3 动态组合仿真场景

> 下述示例包含 `paxini_m2824` 模组与 `add_tactile_contact_pairs` / `TactileForceReader` 接口，需使用包含该特性的仓库提交或后续发布标签（见 2.1 节注释）；若仍锁定 `v0.2.0` 标签，请仅使用 `pts_2_0` / `flat_pad` 及 `build_gripper_spec`、`get_stroke_limits`、`get_tactile_layout`。

```python
from adaptive_tactile_gripper_description import (
    TactileForceReader,
    add_tactile_contact_pairs,
    build_gripper_spec,
    get_stroke_limits,
    get_tactile_layout,
)

gripper_spec = build_gripper_spec(
    left_sensor="pts_2_0",
    right_sensor="paxini_m2824",
    with_camera=False,
    fixed_base=True,
    backlash_rad=0.005,
    slide_frictionloss_n=0.1,
    drive_frictionloss_nm=0.04,
)
limits = get_stroke_limits("pts_2_0", "paxini_m2824")
layout = get_tactile_layout("pts_2_0", "paxini_m2824", prefix="gripper/")

# 挂载夹爪与被抓物体后，统一注册接触对并用 TactileForceReader 读取三轴阵列力：
add_tactile_contact_pairs(
    scene,
    "cube/target_cube_geom",
    left_sensor="pts_2_0",
    right_sensor="paxini_m2824",
    prefix="gripper/",
)
model = scene.compile()
reader = TactileForceReader(
    model,
    left_sensor="pts_2_0",
    right_sensor="paxini_m2824",
    prefix="gripper/",
    target_geoms="cube/target_cube_geom",
)
```

现有 `tactile_grasp` 的场景生成和触觉读取器要求双侧 3×3，直接接入时使用双 PTS，可通过 `reader.read_grid(data, side)` 直接获取 `(3, rows, cols)` 数组；支持异构模组（含 135 测点 `paxini_m2824`）或手写读取逻辑时，按以下规则接入：
- **`add_tactile_contact_pairs` / `TactileForceReader`（推荐）** 根据 `layout[side]["mode"]` 处理配对与读取。为 `contact_geom`（`pts_2_0`）逐球注册 `<pair>` 并读取各球的三维力。为 `pad_sites`（`paxini_m2824`）仅对整块平垫和目标几何体注册一条 `<pair>`，并把接触力分配到 135 个测点。`read(data)` 按 `layout[side]["taxels"]` 顺序返回各测点局部坐标系中的 `[Fx, Fy, Fz]`，无触觉侧返回 `None`。`read_grid(data, side)` 按测点声明的 `row` 和 `col` 排列。`target_geoms` 接受单个几何体名称、ID 或两者的序列。
- **手动查询 `layout`**：`contact_geom` 为每个 `taxel["geom_name"]` 注册 `<pair>`；`pad_sites` 整块平垫共用顶层 `layout[side]["pad_geom_name"]`，**仅可对该碰撞体注册一次 `<pair>`**（MuJoCo 不会对重复 `<pair>` 去重，按测点循环注册会成倍放大接触数量）。用挂载后的 `gripper/tcp` 站点放置场景对象，避免依赖两侧中心 taxel。自定义环境碰撞掩码时，壳体使用类型位 `4`，指间接触面使用类型位 `8`，二者均与默认环境类型位 `1` 交互。

对 `pad_sites`，读取器基于垫块碰撞体自身坐标帧仅统计作用于前感应面的接触（自动排除垫块侧面与背面接触），以高斯权重（默认 `pad_sigma_m=0.0025` m）为先验，在前感应面切平面内通过局部非负指数倾斜校正权重质心。每个前表面接触的权重非负且和为一，将各测点力转换到同一坐标系后求和，前感应面三维合力守恒至浮点精度。投影接触点位于测点凸包内时，重构平面压力中心及平面力臂产生的一阶力矩。投影接触点位于凸包外时，按最近的凸包边界点重构压力中心及相应力矩。

使用 `condim=4` 或 `6` 时，接触法向扭矩在垫面法向上的分量由高斯加权的零合力切向力偶重构。加权力臂方差不超过 1e-16 m² 时省略该力偶。`condim=6` 的两个滚动力矩分量省略。对测点法向一致的平垫，纯压缩时各测点的 `Fz` 均非负。

真机的 `*_sensor_link` 与安装系同向，`taxel_*` 受力站点相对安装系绕局部 X 轴旋转 **Rx(π)**，各原点在相应球心。桥接 `hub_*/sensor_*` 的六轴 wrench 时，先核对 `frame_id` 和参考原点。变换力及力矩时使用同一旋转，移动力矩参考点时添加相应的力臂叉乘项，并将协议的 N·mm 转为 N·m。当前仿真读取器输出逐触点三轴力。


### 2.4 外观与场景挂载

外观参数 `appearance` 与查询接口 `list_appearances()` 自 `v0.2.0` 标签起可用。

`build_gripper_spec(appearance="realistic")` 和 `build_gripper_spec(appearance="cad")` 在内存中为底盘、两侧指尖与相机设置材质。省略参数时，接口沿用资产自带的预设。通过 `list_appearances()` 查询可用名称。MuJoCo 挂载模组时为材质名添加 `left_`、`right_` 或 `cam_` 前缀，因此混装时也能定位材质类别。

同一类别下的零件可以有不同颜色，例如真机指根的橙色加强片，以及 CAD 中不同颜色的支架。读取 MJCF 时须同时考虑 geom 的 `rgba` 和关联材质。Xacro 已写入每件的最终 RGBA，RViz 随所安装资产显示。要在 ROS 2 中切换整套配色，在描述包源码目录运行 `uv run python tools/build_description.py --appearance cad` 并重新构建、加载该包。默认构建命令会生成真机配色。

天空盒、网格地面、布光与离屏采样由预览脚本创建。下游场景可以直接 `attach` 夹爪，并由宿主模型设置全局渲染参数。新增模组视觉网格时，应在材质库登记规范网格名和类别，构建会检查缺失映射。[外观配置](../config/appearance.yaml)

---

## 3. 新增自定义指尖传感器模组（如 `eflesh` 或新款触觉垫）

所有传感器模组均装配在左右手指统一的 `sensor_mount` 局部坐标系下：

- **原点**：位于左右指传感器安装面中心（`PTS 2.0` 的 `Housing` 装配参考原点）；
- **`+Z` 轴**：垂直指腹指向被抓物体方向（即传感器向外突出方向）；
- **`+X` 轴**：沿指尖远端向上（对应夹爪坐标系 `+Z`）；
- **`+Y` 轴**：由右手定则 `Z × X` 确定。

- **`paxini_m2824`（内置模块）**：PaXini `PX6AX-GEN4-GR-M2824-Gripper`（28.75 × 24.02 × 6.0 mm）配 3.5 mm 转接垫块，前接触面位于 `thickness_to_tip_m: 0.0037`（双侧全开净开口 72.1 mm，首次接触 1.610581 rad，距 1.7 rad 上限保留 1.50 mm/指压入余量）；MJCF/URDF 采用纯盒式碰撞（`pad_collision` + `sensor_housing_collision`，不使用接触球），并在 `pad_sites` 模式下暴露 135 个测点坐标帧（`taxel_001`–`taxel_135`）。

### 新增模组 3 步清单

假设要新增名为 `my_sensor` 的模组（将网格放到 `meshes/my_sensor/`）：

1. **添加 MuJoCo 模组** `mjcf/sensors/my_sensor.xml`：
   - 根体下设 `<body name="sensor_body" pos="0 0 0">` 与惯量；
   - 视觉网格设 `contype="0" conaffinity="0" group="2"`，接触球/块使用 `class="pillar_collision"`（`group="3"`）；
   - 如为触觉传感器，确保各 taxel `site` 局部 **`+Z` 指向 `-Z_mount`（受压方向）**，使受压时 `Fz > 0`。
2. **添加 ROS 2 Xacro 模组** `urdf/sensors/my_sensor.xacro`：
   - 定义宏 `my_sensor_sensor(prefix, parent, safety_distance:=0.0)`；
   - 创建 `${prefix}_sensor_link`（含视觉、随 `safety_distance` 外扩的 MoveIt 碰撞盒）及末端 `${prefix}_fingertip`；
   - 在 `urdf/gripper.xacro` 中引入并在 `left_sensor` / `right_sensor` 分支增加对应 `<xacro:if>`（同步更新 `srdf/gripper.srdf.xacro` 如需附加连杆免碰撞）。
3. **添加规格文件** `config/sensors/my_sensor.yaml`：记录质量 `mass_kg`、指尖偏置厚度 `thickness_to_tip_m` 与阵列描述；若两侧自定模组接触面正对重叠并希望自动挂载防越心限位肌腱 `opposing_tip_stop`，可显式声明 `opposing_tip_stop: true`。完成后在测试中补充相应编译、展开放置以及可选触觉布局的回归检查。
