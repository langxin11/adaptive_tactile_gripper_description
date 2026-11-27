"""夹爪描述资产路径定位、规格读取与模块化 MuJoCo MjSpec 组装。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import yaml

from .appearance import Appearance, read_appearance
from .kinematics import (
    POSITION_MAX_RAD,
    POSITION_MIN_RAD,
    PROTOCOL_RANGE_RAD,
    SINGLE_FINGER_STROKE_M,
    drive_angle_for_closure,
)

_mj: Any = mujoco
if TYPE_CHECKING:
    type MjSpec = Any
else:
    MjSpec = mujoco.MjSpec


def get_data_root() -> Path:
    """返回描述包资产根目录（优先包内 `_data`，否则回退到源码仓库根目录）。"""
    pkg_data = Path(__file__).resolve().parent / "_data"
    if (pkg_data / "mjcf" / "gripper_chassis.xml").is_file():
        return pkg_data
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "mjcf" / "gripper_chassis.xml").is_file():
        return repo_root
    raise FileNotFoundError("未找到夹爪描述资产目录（mjcf/gripper_chassis.xml 不存在）")


def list_appearances() -> list[str]:
    """返回夹爪材质库提供的外观预设名称。"""
    return list(read_appearance(get_data_root() / "config" / "appearance.yaml")["presets"])


def list_sensors() -> list[str]:
    """返回可用的指尖传感器模组名称列表。"""
    sensors_dir = get_data_root() / "mjcf" / "sensors"
    return sorted(p.stem for p in sensors_dir.glob("*.xml"))


def list_accessories() -> list[str]:
    """返回可用的外设附件模组名称列表。"""
    acc_dir = get_data_root() / "mjcf" / "accessories"
    return sorted(p.stem for p in acc_dir.glob("*.xml"))


def get_specs() -> dict[str, Any]:
    """读取夹爪机构与几何单一真值 `config/specs.yaml`。"""
    with open(get_data_root() / "config" / "specs.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_sensor_spec(name: str) -> dict[str, Any]:
    """读取指定传感器模组的规格配置 `config/sensors/<name>.yaml`。"""
    path = get_data_root() / "config" / "sensors" / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"未知传感器模组 {name!r}，可选：{list_sensors()}")
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_stroke_limits(
    left_sensor: str | None = "pts_2_0",
    right_sensor: str | None = "pts_2_0",
) -> dict[str, Any]:
    """返回协议角度、机构行程及所选模组首次接触前的工作行程。

    模组行程按名义刚体几何计算，角度为包含回差的实际曲柄角。
    在机构行程内无模组接触时，首次接触字段为 None。
    """
    sensors = [
        get_sensor_spec(name) if name is not None and name != "none" else None
        for name in (left_sensor, right_sensor)
    ]
    mount_gap = get_specs()["geometry"]["nominal_full_open_sensor_mount_gap_m"]
    tip_gap = mount_gap - sum(
        sensor["thickness_to_tip_m"] for sensor in sensors if sensor is not None
    )
    contact_slide = tip_gap / 2
    has_contact = (
        all(sensor is not None for sensor in sensors) and contact_slide <= SINGLE_FINGER_STROKE_M
    )
    contact_q = drive_angle_for_closure(contact_slide) if has_contact else None
    return {
        "protocol_range_rad": PROTOCOL_RANGE_RAD,
        "mechanism_range_rad": (POSITION_MIN_RAD, POSITION_MAX_RAD),
        "mechanism_single_finger_stroke_m": SINGLE_FINGER_STROKE_M,
        "first_contact_q_rad": contact_q,
        "first_contact_single_finger_closure_m": contact_slide if has_contact else None,
        "no_contact_range_rad": (POSITION_MIN_RAD, contact_q if has_contact else POSITION_MAX_RAD),
        "no_contact_single_finger_stroke_m": contact_slide
        if has_contact
        else SINGLE_FINGER_STROKE_M,
    }


def get_tactile_layout(
    left_sensor: str | None = "pts_2_0",
    right_sensor: str | None = "pts_2_0",
    *,
    prefix: str = "",
) -> dict[str, dict[str, Any] | None]:
    """按侧返回触觉阵列及 MJCF geom/site 名称，无触觉的一侧返回 None。

    `prefix` 可传入挂载前缀（如 `gripper/`），各测点位置以 `sensor_mount` 表达：
    - `mode == "contact_geom"`（如 `pts_2_0`）：每个 `taxels` 元素含独立碰撞体 `geom_name` 与 `site_name`；
    - `mode == "pad_sites"`（如 `paxini_m2824`）：整块平垫共用顶层 `pad_geom_name`（显式 `<pair>` 仅可对该
      `pad_geom_name` 注册一次，切勿按测点重复注册），每个 `taxels` 元素含测点 `id`、`pos_m` 与 `site_name`；
    - `mode == "pad_wrench"`（如 `soft_dome`）：单个柔顺球冠/垫块碰撞体 `pad_geom_name` 配接触面参考系
      `site_name`（及可选底座量测中心 `base_site_name`），用于六轴接触力/抗扭力矩（Wrench）读取。
    """
    layout = {}
    for side, name in (("left", left_sensor), ("right", right_sensor)):
        sensor = get_sensor_spec(name) if name is not None and name != "none" else None
        tactile = sensor["tactile"] if sensor is not None else None
        if sensor is None or tactile is None:
            layout[side] = None
            continue
        if tactile["mode"] == "pad_wrench":
            base_key = f"{side}_base_site"
            layout[side] = {
                "mode": "pad_wrench",
                "pad_geom_name": prefix + tactile[f"{side}_pad_geom"],
                "site_name": prefix + tactile[f"{side}_site"],
                "base_site_name": (prefix + tactile[base_key] if tactile.get(base_key) else None),
                "site_quat_in_sensor_mount_wxyz": tactile["site_quat_in_sensor_mount_wxyz"],
            }
            continue
        site_prefix = prefix + tactile[f"{side}_site_prefix"]
        if tactile["mode"] == "pad_sites":
            pad_geom = prefix + tactile[f"{side}_pad_geom"]
            taxels = [
                {
                    **taxel,
                    "site_name": f"{site_prefix}{taxel['id']:03d}",
                }
                for taxel in sensor["taxels_in_sensor_mount"]
            ]
            layout[side] = {
                "mode": "pad_sites",
                "count": tactile["count"],
                "pad_geom_name": pad_geom,
                "site_quat_in_sensor_mount_wxyz": tactile["site_quat_in_sensor_mount_wxyz"],
                "taxels": taxels,
            }
            continue
        geom_prefix = prefix + tactile[f"{side}_prefix"]
        taxels = []
        for taxel in sensor["taxels_in_sensor_mount"]:
            index = f"{taxel['row']}{taxel['col']}"
            taxels.append(
                {**taxel, "geom_name": geom_prefix + index, "site_name": site_prefix + index}
            )
        layout[side] = {
            "mode": tactile["mode"],
            "rows": tactile["rows"],
            "cols": tactile["cols"],
            "site_quat_in_sensor_mount_wxyz": tactile["site_quat_in_sensor_mount_wxyz"],
            "taxels": taxels,
        }
    return layout


def get_mjcf_path(variant: str = "gripper") -> Path:
    """返回预生成单文件 MJCF 路径（`gripper`、`gripper_no_camera` 或 `gripper_chassis`）。"""
    path = get_data_root() / "mjcf" / f"{variant}.xml"
    if not path.is_file():
        raise FileNotFoundError(f"未找到 MJCF 文件: {path}")
    return path


def _load_chassis_spec(
    data_root: Path,
    *,
    fixed_base: bool,
    backlash_rad: float | None,
    slide_frictionloss_n: float | None,
    drive_frictionloss_nm: float | None,
    appearance: Appearance | None,
) -> MjSpec:
    """加载底盘 XML 并按需覆盖机构参数与固定基座设置。"""
    chassis_path = data_root / "mjcf" / "gripper_chassis.xml"
    root = ET.parse(chassis_path).getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str((data_root / "meshes").resolve()))

    if fixed_base:
        base = root.find(".//body[@name='base']")
        assert base is not None
        for freejoint in base.findall("freejoint"):
            base.remove(freejoint)

    if slide_frictionloss_n is not None:
        for jnt_name in ("left_finger_slide", "right_finger_slide"):
            jnt = root.find(f".//joint[@name='{jnt_name}']")
            if jnt is not None:
                jnt.set("frictionloss", str(slide_frictionloss_n))

    if drive_frictionloss_nm is not None:
        drive_jnt = root.find(".//joint[@name='gripper_drive']")
        if drive_jnt is not None:
            drive_jnt.set("frictionloss", str(drive_frictionloss_nm))

    if backlash_rad is not None:
        backlash = root.find(".//joint[@name='gripper_backlash']")
        if backlash is not None:
            if backlash_rad > 0.0:
                backlash.set("range", f"{-backlash_rad / 2.0} {backlash_rad / 2.0}")
            else:
                for parent in root.iter():
                    if backlash in list(parent):
                        parent.remove(backlash)
                        break

    return _parse_spec(root, appearance)


def _load_module_spec(xml_path: Path, meshes_dir: Path, appearance: Appearance | None) -> MjSpec:
    """加载传感器或相机模组 XML 并解析其绝对网格目录。"""
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        compiler.set("meshdir", str(meshes_dir.resolve()))
    return _parse_spec(root, appearance)


def _parse_spec(root: ET.Element, appearance: Appearance | None) -> MjSpec:
    spec = _mj.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    if appearance is not None:
        appearance.apply(spec)
    return spec


def _add_module_contact_exclusions(
    spec: MjSpec,
    left: bool,
    right: bool,
    with_camera: bool,
) -> None:
    """按挂载情况排除底盘装配界面处的内部接触。"""
    for side, mounted in (("left", left), ("right", right)):
        if mounted:
            for adjacent in ("base", f"{side}_finger"):
                spec.add_exclude(bodyname1=adjacent, bodyname2=f"{side}_sensor_body")
    if with_camera:
        spec.add_exclude(bodyname1="base", bodyname2="cam_camera_assembly")


_OVERLAPPING_TIP_SENSORS = frozenset({"pts_2_0", "flat_pad", "paxini_m2824", "soft_dome"})


def _contact_slide_if_configured(
    data_root: Path, left_sensor: str | None, right_sensor: str | None
) -> float | None:
    if left_sensor in (None, "none") or right_sensor in (None, "none"):
        return None
    for name in (left_sensor, right_sensor):
        yaml_path = data_root / "config" / "sensors" / f"{name}.yaml"
        if not yaml_path.is_file():
            return None
        spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        thickness = spec.get("thickness_to_tip_m") if isinstance(spec, dict) else None
        if isinstance(thickness, bool) or not isinstance(thickness, (int, float)):
            return None
        if name not in _OVERLAPPING_TIP_SENSORS and spec.get("opposing_tip_stop") is not True:
            return None
    return get_stroke_limits(left_sensor, right_sensor)["first_contact_single_finger_closure_m"]


def _add_opposing_tip_stop(spec: MjSpec, contact_slide_m: float | None) -> None:
    """当两侧传感器在行程内可相遇时，添加单调限位肌腱防止大扭矩/大步长下对置指尖越心反转。"""
    if contact_slide_m is None:
        return
    # 允许双侧指尖在首次接触后合计留出 3.0 mm（单侧 1.5 mm）压入行程，既确保 1–3 mm 薄物体
    # 满扭矩抓取不被肌腱分流夹持力，又严格早于 5.6–6.0 mm 球心/垫块中面越界点，避免法向反转卡死。
    max_closure = min(2.0 * float(contact_slide_m) + 3e-3, 2.0 * SINGLE_FINGER_STROKE_M)
    stop = spec.add_tendon(
        name="opposing_tip_stop",
        limited=True,
        range=[-1.0, max_closure],
        solref_limit=[0.002, 1.0],
        solimp_limit=[0.99, 0.999, 0.0001, 0.5, 2.0],
    )
    stop.wrap_joint("left_finger_slide", 1.0)
    stop.wrap_joint("right_finger_slide", 1.0)


def build_gripper_spec(
    *,
    left_sensor: str | None = "pts_2_0",
    right_sensor: str | None = "pts_2_0",
    with_camera: bool = True,
    fixed_base: bool = False,
    backlash_rad: float | None = None,
    slide_frictionloss_n: float | None = None,
    drive_frictionloss_nm: float | None = None,
    appearance: str | None = None,
) -> MjSpec:
    """按参数模块化组装夹爪 MuJoCo MjSpec。

    参数
        left_sensor: 左指传感器模组名（如 ``pts_2_0``、``flat_pad``、``paxini_m2824``、``soft_dome``），``None`` 或 ``"none"`` 不挂传感器。
        right_sensor: 右指传感器模组名，语义同 ``left_sensor``。
        appearance: 外观预设名；省略时沿用资产自带的材质。
        with_camera: 是否挂载侧置 RealSense D435i 支架与相机。
        fixed_base: 为 True 时移除 ``base_freejoint``，便于直接挂载到地面或机械臂法兰。
        backlash_rad: 可选覆盖 ``gripper_backlash`` 回差总宽度 (rad)；<=0 时移除回差关节。
        slide_frictionloss_n: 可选覆盖左右滑块 ``frictionloss`` (N)。
        drive_frictionloss_nm: 可选覆盖 ``gripper_drive`` 摩擦 (N·m)。

    返回
        可直接 ``compile()`` 或被上层场景 ``attach()`` 的 ``mujoco.MjSpec``。
    """
    data_root = get_data_root()
    meshes_dir = data_root / "meshes"
    palette = None
    if appearance is not None:
        palette = Appearance.from_library(
            read_appearance(data_root / "config" / "appearance.yaml"), appearance
        )
    spec = _load_chassis_spec(
        data_root,
        fixed_base=fixed_base,
        appearance=palette,
        backlash_rad=backlash_rad,
        slide_frictionloss_n=slide_frictionloss_n,
        drive_frictionloss_nm=drive_frictionloss_nm,
    )

    if with_camera:
        cam_xml = data_root / "mjcf" / "accessories" / "realsense_d435i.xml"
        cam_spec = _load_module_spec(cam_xml, meshes_dir, palette)
        spec.attach(cam_spec, prefix="cam_", frame=spec.frame("camera_mount_frame"))

    available = set(list_sensors())
    for side, sensor_name in (("left", left_sensor), ("right", right_sensor)):
        if sensor_name is None or sensor_name == "none":
            continue
        if sensor_name not in available:
            raise ValueError(f"未知传感器模组 {sensor_name!r}，可选：{sorted(available)}")
        sensor_xml = data_root / "mjcf" / "sensors" / f"{sensor_name}.xml"
        sensor_spec = _load_module_spec(sensor_xml, meshes_dir, palette)
        spec.attach(sensor_spec, prefix=f"{side}_", frame=spec.frame(f"{side}_sensor_mount"))

    _add_module_contact_exclusions(
        spec,
        left_sensor not in (None, "none"),
        right_sensor not in (None, "none"),
        with_camera,
    )
    _add_opposing_tip_stop(spec, _contact_slide_if_configured(data_root, left_sensor, right_sensor))
    return spec
