#!/usr/bin/env python3
"""从 Onshape 导出的原始 MJCF 自动生成模块化 MJCF、URDF/Xacro、SRDF 与规格配置。

包含基于二进制 STL 多面体体积矩积分的刚体惯量计算、以实测整机总质量 686 g 约束的
均匀填充惯量估计（手指 PLA + 结构件 PLA-CF）、标准 DM-J4310P-2EC 与 RealSense D435i
模型参数对齐，以及 PTS 2.0 球体碰撞几何定义。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_tactile_gripper_description.appearance import Appearance, is_visual, read_appearance
from adaptive_tactile_gripper_description.assembly import (
    _add_module_contact_exclusions,
    _add_opposing_tip_stop,
)
from adaptive_tactile_gripper_description.component_mass import (
    BASE_FASTENERS_MASS_KG,
    FINGER_FASTENERS_MASS_KG,
    MF84_BEARING_MASS_KG,
    MGN9_RAIL_150MM_MASS_KG,
    MGN9C_BLOCK_MASS_KG,
    MOTOR_FASTENERS_MASS_KG,
    ROD_SPACER_HARDWARE_MASS_KG,
)
from adaptive_tactile_gripper_description.kinematics import (
    CRANK_RADIUS_M,
    LINK_LENGTH_M,
    OFFSET_M,
    POSITION_MAX_RAD,
    POSITION_MIN_RAD,
    PROTOCOL_RANGE_RAD,
    SINGLE_FINGER_STROKE_M,
    THETA0_RAD,
    TOTAL_STROKE_M,
    drive_angle_for_closure,
    get_joint_positions,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT_DIR = ROOT / "tools" / "onshape_export"
DEFAULT_PAXINI_EXPORT_DIR = ROOT / "tools" / "paxini_export"
PACKAGE_NAME = "adaptive_tactile_gripper_description"

FULL_OPEN_MOUNT_GAP_M = 0.0795
TIP_PROTRUSION_M = 0.0066
PAXINI_M2824_TIP_PROTRUSION_M = 0.0037
SOFT_DOME_TIP_PROTRUSION_M = 0.0037
SOFT_DOME_BASE_FRONT_Z_M = -0.0038
SOFT_DOME_BACK_Z_M = -0.0058
SOFT_DOME_BASE_RADIUS_M = 0.0120
SOFT_DOME_RADIUS_M = 0.0120
SOFT_DOME_SPHERE_CENTER_Z_M = round(SOFT_DOME_TIP_PROTRUSION_M - SOFT_DOME_RADIUS_M, 9)
SOFT_DOME_TRANSDUCER_MASS_KG = 0.0060
SOFT_DOME_SILICONE_DENSITY_KG_M3 = 1070.0
SOFT_DOME_HOUSING_POS = [0.0, 0.0, -0.0048]
SOFT_DOME_HOUSING_SIZE = [0.012001, 0.001001]
SRDF_CLOSE_M = 0.032

# 原始网格名 -> 仓库规范 ASCII 小写网格名（不含 .stl 后缀）
MESH_NAME_MAP: dict[str, str] = {
    "8x4x2_flanged_bearing": "bearing_8x4x2",
    "8x4x2_flanged_bearing__2": "bearing_8x4x2_2",
    "8x4x2_flanged_bearing__3": "bearing_8x4x2_3",
    "8x4x2_flanged_bearing__4": "bearing_8x4x2_4",
    "bracket_2": "bracket_2",
    "camera_mount": "camera_mount",
    "crank": "crank",
    "d435i_solid": "d435i_solid",
    "franka_adapter": "franka_adapter",
    "housing": "housing",
    "link": "link",
    "ls__mgn9_block__6faf12bd810aab5c7f7b6a160ef20d75": "mgn9_block",
    "ls__mgn9_rail__6faf12bd810aab5c7f7b6a160ef20d75": "mgn9_rail",
    "motor": "motor",
    "motor_mount": "motor_mount",
    "part_1": "finger_mount",
    "part_1__配置": "link_spacer",
    "part_2": "finger_brace",
    "part_4": "finger_body",
    "pillars_0": "pillars_0",
    "pillars_1": "pillars_1",
    "pillars_2": "pillars_2",
    "pillars_3": "pillars_3",
    "pillars_4": "pillars_4",
    "pillars_5": "pillars_5",
    "pillars_6": "pillars_6",
    "pillars_7": "pillars_7",
    "pillars_8": "pillars_8",
    "shim": "shim",
    "stator": "stator",
}

# 左右手指中心与法兰 TCP 坐标（base 原点位于 franka_adapter 下端面中心）
TCP_IN_FLANGE_M = [0.0, 0.0, 0.14555]
LEFT_SENSOR_MOUNT_IN_FINGER_POS = [0.0, 0.0615, 0.03525]
LEFT_SENSOR_MOUNT_IN_FINGER_QUAT = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]
RIGHT_SENSOR_MOUNT_IN_FINGER_POS = [0.0, -0.0615, 0.03525]
RIGHT_SENSOR_MOUNT_IN_FINGER_QUAT = [0.7071067811865476, 0.0, 0.0, -0.7071067811865476]

# ---------- 参考与实测质量规格（单位：kg） ----------
# 电子秤实测整机总重（双侧 PTS 2.0，不含 D435i 相机模组，不含装机 M6x12 螺钉）
MEASURED_TOTAL_NO_CAMERA_KG = 0.686000

# 达妙 DM-J4310P-2EC 官方手册标称整机 0.330 kg，定子与转子拆分为建模估计
DM4310P_STATOR_MASS_KG = 0.2820
DM4310P_ROTOR_OUTPUT_MASS_KG = 0.0480

# HIWIN MGN9R (150mm, 0.38 kg/m) 与 MGN9C (28.9mm 短滑块)

# 连杆五金（单根）：4x NSK MF84 (0.44 g) + 2x link_spacer/销轴/垫片 (4.76 g)

# 机内 A2-70 不锈钢紧固件（ISO 4762 / ISO 4032）

# Contactile PTS 2.0 单只 3x3 传感头（中空铝壳 + Shore-A40 硅胶膜柱 + 内置相机/LED PCB）
PTS_2_0_MASS_KG = 0.0150
FLAT_PAD_MASS_KG = 0.0100
# PaXini PX6AX-GEN4-GR-M2824-Gripper 裸传感器 (8.5 g) + 6x M1.6x4 不锈钢螺钉 (0.32 g)
PAXINI_M2824_SENSOR_MASS_KG = 0.00882
PAXINI_M2824_HOUSING_POS = [-0.00205, 0.0, -0.004051]
PAXINI_M2824_HOUSING_SIZE = [0.01236, 0.012013, 0.001751]
PAXINI_M2824_PAD_POS = [-6.8e-05, 0.0, 0.00069875]
PAXINI_M2824_PAD_SIZE = [0.014378, 0.012013, 0.00300125]

# Intel RealSense D435i 官方裸机 72.0 g + 支架五金 (2x M4x8 + 2x M4x16 + 2x M4螺母 + 1/4-20螺钉)
D435I_BARE_MASS_KG = 0.0720
CAMERA_FASTENERS_MASS_KG = 2 * 0.00167 + 2 * 0.00227 + 2 * 0.00082 + 0.00220  # 11.72g
# 由 MuJoCo Menagerie 的 D435i 模型得到的质心、主惯量和主轴方向，表达基准为 STL 系
D435I_COM_IN_STL_M = np.array([0.00145113, -0.00031933, -0.01354119], dtype=float)
D435I_DIAGINERTIA_LOCAL = np.array([3.7994e-05, 3.7055e-05, 4.5547e-06], dtype=float)
D435I_IQUAT_IN_STL_WXYZ = [0.02901893, 0.70831212, 0.02754758, 0.70476448]

# 3D 打印丝材原丝密度 (kg/m^3)：手指为 PLA，其余结构件为 PLA-CF
RAW_DENSITY_PLA_KG_M3 = 1240.0
RAW_DENSITY_PLACF_KG_M3 = 1220.0


@dataclass(frozen=True)
class InertialPart:
    """单个刚体组成部分的质量属性（在父 body 局部坐标系下）。"""

    name: str
    mass_kg: float
    com_m: np.ndarray
    inertia_com_kg_m2: np.ndarray


def _quat_wxyz_to_mat(quat: list[float]) -> np.ndarray:
    """将 wxyz 四元数转为 3x3 旋转矩阵。"""
    q = np.asarray(quat, dtype=float)
    q = q / np.linalg.norm(q)
    mat = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)


def _mat_to_quat_wxyz(mat: np.ndarray) -> list[float]:
    """将 3x3 旋转矩阵转为 wxyz 四元数。"""
    q = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=float).ravel())
    if q[0] < 0:
        q = -q
    return [round(float(x), 8) for x in q]


def _mat_to_rpy(mat: np.ndarray) -> list[float]:
    """将 3x3 旋转矩阵转为 URDF 外旋 XYZ（等价内旋 ZYX）欧拉角。"""
    horizontal = math.hypot(mat[0, 0], mat[1, 0])
    pitch = math.atan2(-mat[2, 0], horizontal)
    if horizontal > 1e-12:
        roll = math.atan2(mat[2, 1], mat[2, 2])
        yaw = math.atan2(mat[1, 0], mat[0, 0])
    else:
        yaw = 0.0
        if pitch > 0.0:
            roll = math.atan2(mat[0, 1], mat[0, 2])
        else:
            roll = math.atan2(-mat[0, 1], -mat[0, 2])
    return [roll, pitch, yaw]


def _parse_vec(text: str | None, default: list[float]) -> list[float]:
    """解析空格分隔浮点字符串。"""
    if not text:
        return list(default)
    return [float(x) for x in text.split()]


def _fmt_vec(vals: list[float]) -> str:
    """格式化向量为精简可读字符串。"""
    out = []
    for v in vals:
        if abs(v) < 1e-12:
            v = 0.0
        out.append(f"{v:.8g}")
    return " ".join(out)


def _pose_to_mat(pos: list[float], quat: list[float]) -> np.ndarray:
    """由 pos 和 wxyz 四元数构造 4x4 齐次变换矩阵。"""
    T = np.eye(4, dtype=float)
    T[:3, :3] = _quat_wxyz_to_mat(quat)
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def _stl_triangles(stl_path: Path) -> np.ndarray:
    """读取二进制 STL 的三角形顶点，单位为米。"""
    data = stl_path.read_bytes()
    ntri = struct.unpack("<I", data[80:84])[0]
    triangles = struct.iter_unpack("<12x9f2x", data[84 : 84 + 50 * ntri])
    return np.array(list(triangles), dtype=float).reshape(ntri, 3, 3)


def _stl_polyhedral_props(stl_path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """四面体有向体积积分计算闭合二进制 STL 的体积(m^3)、体心(m)与单位密度绕体心惯量矩阵。"""
    vol = 0.0
    m1 = np.zeros(3, dtype=float)
    cov = np.zeros((3, 3), dtype=float)
    for a, b, c in _stl_triangles(stl_path):
        det = float(np.dot(a, np.cross(b, c)))
        vol += det / 6.0
        m1 += (det / 24.0) * (a + b + c)
        for i in range(3):
            for j in range(3):
                cov[i, j] += (det / 120.0) * (
                    2.0 * (a[i] * a[j] + b[i] * b[j] + c[i] * c[j])
                    + (a[i] * b[j] + b[i] * a[j])
                    + (a[i] * c[j] + c[i] * a[j])
                    + (b[i] * c[j] + c[i] * b[j])
                )
    if vol < 0.0:
        vol, m1, cov = -vol, -m1, -cov
    if vol <= 1e-15:
        raise ValueError(f"STL 体积为零或未闭合: {stl_path}")
    com = m1 / vol
    cov_com = cov - vol * np.outer(com, com)
    inertia_unit = float(np.trace(cov_com)) * np.eye(3) - cov_com
    inertia_unit = 0.5 * (inertia_unit + inertia_unit.T)
    eigs = np.linalg.eigvalsh(inertia_unit)
    if np.any(eigs <= 0.0):
        raise ValueError(f"STL 单位密度惯量矩阵非正定 ({eigs}): {stl_path}")
    return float(vol), com, inertia_unit


def _part_from_mesh(
    name: str,
    stl_path: Path,
    pos: list[float],
    quat: list[float],
    *,
    mass_kg: float | None = None,
    density_kg_m3: float | None = None,
) -> InertialPart:
    """按 STL 几何积分构造位于父 body 坐标系中的 InertialPart。"""
    vol, com_local, i_unit_local = _stl_polyhedral_props(stl_path)
    if mass_kg is None:
        if density_kg_m3 is None:
            raise ValueError(f"{name}: 必须指定 mass_kg 或 density_kg_m3")
        mass_kg = vol * density_kg_m3
    scale = mass_kg / vol
    i_local = i_unit_local * scale
    R = _quat_wxyz_to_mat(quat)
    p = np.asarray(pos, dtype=float)
    com_parent = p + R @ com_local
    i_parent = R @ i_local @ R.T
    return InertialPart(
        name=name, mass_kg=float(mass_kg), com_m=com_parent, inertia_com_kg_m2=i_parent
    )


def _point_part(
    name: str, mass_kg: float, pos: list[float], diag_inertia: list[float] | None = None
) -> InertialPart:
    """构造紧固件等小零件的等效惯量块。"""
    i_mat = np.diag(diag_inertia if diag_inertia is not None else [1e-8, 1e-8, 1e-8]).astype(float)
    return InertialPart(
        name=name,
        mass_kg=float(mass_kg),
        com_m=np.asarray(pos, dtype=float),
        inertia_com_kg_m2=i_mat,
    )


def _combine_inertials(parts: list[InertialPart]) -> tuple[float, np.ndarray, np.ndarray]:
    """用平行轴定理合并多个刚体子件的总质量、合质心与合惯量矩阵。"""
    total_mass = sum(p.mass_kg for p in parts)
    if total_mass <= 0.0:
        raise ValueError("刚体合并总质量必须为正")
    com = sum(p.mass_kg * p.com_m for p in parts) / total_mass
    I_total = np.zeros((3, 3), dtype=float)
    for p in parts:
        d = p.com_m - com
        I_total += p.inertia_com_kg_m2 + p.mass_kg * (
            float(np.dot(d, d)) * np.eye(3) - np.outer(d, d)
        )
    I_total = 0.5 * (I_total + I_total.T)
    eigs = np.linalg.eigvalsh(I_total)
    if np.any(eigs <= 0.0):
        raise ValueError(f"合并后惯量矩阵非正定: {eigs}")
    return float(total_mass), com, I_total


def _fullinertia_str(I_mat: np.ndarray) -> str:
    """转为 MuJoCo fullinertia 属性格式 [Ixx Iyy Izz Ixy Ixz Iyz]。"""
    return _fmt_vec(
        [
            float(I_mat[0, 0]),
            float(I_mat[1, 1]),
            float(I_mat[2, 2]),
            float(I_mat[0, 1]),
            float(I_mat[0, 2]),
            float(I_mat[1, 2]),
        ]
    )


def _set_inertial_element(
    body_el: ET.Element, mass_kg: float, com: np.ndarray, I_mat: np.ndarray
) -> None:
    """替换或创建 body 下的 <inertial> 标签。"""
    old = body_el.find("inertial")
    if old is not None:
        body_el.remove(old)
    el = ET.Element(
        "inertial",
        {
            "pos": _fmt_vec(com.tolist()),
            "mass": f"{mass_kg:.8g}",
            "fullinertia": _fullinertia_str(I_mat),
        },
    )
    # 紧跟在 freejoint 或 joint 之后，保持 XML 可读顺序
    insert_idx = 0
    for idx, child in enumerate(list(body_el)):
        if child.tag in ("freejoint", "joint"):
            insert_idx = idx + 1
    body_el.insert(insert_idx, el)


def _solve_print_fill_ratio(raw_root: ET.Element, meshes_dir: Path) -> tuple[float, float, float]:
    """根据无相机标准件总重与实测 686 g，反解 3D 打印结构件的统一有效填充系数。"""
    # 100% 实心参考打印质量（单只）
    pla_meshes = [("finger_body", 2)]
    placf_meshes = [
        ("franka_adapter", 1),
        ("motor_mount", 1),
        ("bracket_2", 1),
        ("crank", 1),
        ("link", 2),
        ("finger_mount", 2),
        ("finger_brace", 2),
    ]
    solid_pla_kg = sum(
        _stl_polyhedral_props(meshes_dir / f"{m}.stl")[0] * RAW_DENSITY_PLA_KG_M3 * n
        for m, n in pla_meshes
    )
    solid_placf_kg = sum(
        _stl_polyhedral_props(meshes_dir / f"{m}.stl")[0] * RAW_DENSITY_PLACF_KG_M3 * n
        for m, n in placf_meshes
    )

    standard_no_cam_kg = (
        DM4310P_STATOR_MASS_KG
        + DM4310P_ROTOR_OUTPUT_MASS_KG
        + MGN9_RAIL_150MM_MASS_KG
        + 2.0 * MGN9C_BLOCK_MASS_KG
        + 2.0 * (4.0 * MF84_BEARING_MASS_KG + ROD_SPACER_HARDWARE_MASS_KG)
        + BASE_FASTENERS_MASS_KG
        + MOTOR_FASTENERS_MASS_KG
        + 2.0 * FINGER_FASTENERS_MASS_KG
        + 2.0 * PTS_2_0_MASS_KG
    )
    target_print_kg = MEASURED_TOTAL_NO_CAMERA_KG - standard_no_cam_kg
    phi = target_print_kg / (solid_pla_kg + solid_placf_kg)
    if not (0.45 <= phi <= 1.05):
        raise ValueError(f"求解打印有效密度比例异常 phi={phi:.4f}，请核对零件列表与实测重量")
    rho_pla = RAW_DENSITY_PLA_KG_M3 * phi
    rho_placf = RAW_DENSITY_PLACF_KG_M3 * phi
    return phi, rho_pla, rho_placf


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """原地美化 XML 缩进。"""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def _write_xml(root: ET.Element, path: Path) -> None:
    """将 ElementTree 根节点格式化写入 UTF-8 XML 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    _indent_xml(root)
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    path.write_bytes(xml_bytes + b"\n")


def _copy_and_rename_meshes(
    raw_root: ET.Element, raw_assets_dir: Path, out_meshes_dir: Path
) -> set[str]:
    """仅复制被引用且映射的网格文件到 meshes/ 并返回规范网格名集合。"""
    out_meshes_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_meshes_dir.glob("*.stl"):
        stale.unlink()

    referenced: set[str] = set()
    for mesh_el in raw_root.findall("./asset/mesh"):
        file_attr = mesh_el.get("file", "")
        stem = Path(file_attr).stem
        if stem not in MESH_NAME_MAP:
            raise KeyError(f"未在 MESH_NAME_MAP 中注册的网格: {file_attr}")
        src = raw_assets_dir / Path(file_attr).name
        if not src.is_file():
            raise FileNotFoundError(f"缺少导出网格文件: {src}")
        canonical = MESH_NAME_MAP[stem]
        shutil.copy2(src, out_meshes_dir / f"{canonical}.stl")
        referenced.add(canonical)
    return referenced


def _load_paxini_taxels(path: Path) -> list[dict[str, object]]:
    """读取并验证 PaXini 测点导出，保留原始顺序。"""
    taxels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(taxels, list) or len(taxels) != 135:
        raise ValueError(f"{path} 必须包含 135 个测点的列表")

    ids: set[int] = set()
    for index, item in enumerate(taxels):
        if not isinstance(item, dict):
            raise ValueError(f"{path} 第 {index + 1} 项必须为对象")
        taxel_id = item.get("id")
        if type(taxel_id) is not int or not 1 <= taxel_id <= 135 or taxel_id in ids:
            raise ValueError(f"{path} 第 {index + 1} 项的 id 必须为 1..135 内的唯一整数")
        ids.add(taxel_id)

        pos = item.get("pos_m")
        if (
            not isinstance(pos, list)
            or len(pos) != 3
            or not all(
                type(value) in (int, float) and abs(value) <= sys.float_info.max for value in pos
            )
        ):
            raise ValueError(f"{path} 中 id={taxel_id} 的 pos_m 必须包含 3 个有限数值")
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        if (
            abs(x - PAXINI_M2824_PAD_POS[0]) > PAXINI_M2824_PAD_SIZE[0] + 1e-4
            or abs(y - PAXINI_M2824_PAD_POS[1]) > PAXINI_M2824_PAD_SIZE[1] + 1e-4
            or abs(z - PAXINI_M2824_TIP_PROTRUSION_M) > 5e-4
        ):
            raise ValueError(f"{path} 中 id={taxel_id} 的 pos_m 超出感应面范围：{pos}")
    return taxels


def _assert_closed_manifold_stl(stl_path: Path) -> None:
    """校验二进制 STL 为闭合且面片定向一致的流形网格（每条有向边恰好配对一次反向边）。"""
    tris = _stl_triangles(stl_path)
    if len(tris) < 4:
        raise ValueError(f"{stl_path.name} 三角面片不足，无法构成闭合刚体")
    edges: dict[tuple[tuple[float, float, float], tuple[float, float, float]], int] = {}
    for tri in tris:
        v0, v1, v2 = (tuple(np.round(v, 9).tolist()) for v in tri)
        if v0 == v1 or v1 == v2 or v2 == v0:
            raise ValueError(f"{stl_path.name} 包含退化三角面片")
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            edges[(a, b)] = edges.get((a, b), 0) + 1
    for (a, b), count in edges.items():
        if count != 1 or edges.get((b, a), 0) != 1:
            raise ValueError(f"{stl_path.name} 非闭合或面片定向不一致，无法可靠积分惯量")


def _clip_polygon_z(
    poly: list[np.ndarray], z_plane: float, *, keep_above: bool
) -> list[np.ndarray]:
    """用平面 z = z_plane 裁剪凸多边形，返回保留侧的新顶点列表。"""
    if not poly:
        return []

    def inside(pt: np.ndarray) -> bool:
        return bool(pt[2] >= z_plane) if keep_above else bool(pt[2] <= z_plane)

    out: list[np.ndarray] = []
    prev = poly[-1]
    prev_in = inside(prev)
    for curr in poly:
        curr_in = inside(curr)
        if curr_in != prev_in:
            dz = float(curr[2] - prev[2])
            if abs(dz) > 1e-15:
                t = (z_plane - float(prev[2])) / dz
                out.append(prev + t * (curr - prev))
        if curr_in:
            out.append(curr)
        prev, prev_in = curr, curr_in
    return out


def _validate_paxini_meshes(paxini_export_dir: Path) -> None:
    """在修改输出目录前核对 PaXini 源 STL 的存在性、闭合惯量、整面片碰撞包络与 MuJoCo 编译有效性。"""
    sensor_stl = paxini_export_dir / "paxini_m2824.stl"
    adapter_stl = paxini_export_dir / "paxini_m2824_adapter.stl"
    for src in (sensor_stl, adapter_stl):
        if not src.is_file():
            raise FileNotFoundError(f"缺少 PaXini 导出网格文件: {src}")
    _assert_closed_manifold_stl(adapter_stl)
    _stl_polyhedral_props(adapter_stl)
    c_h, s_h = np.asarray(PAXINI_M2824_HOUSING_POS), np.asarray(PAXINI_M2824_HOUSING_SIZE)
    c_p, s_p = np.asarray(PAXINI_M2824_PAD_POS), np.asarray(PAXINI_M2824_PAD_SIZE)
    z_split = float(c_p[2] - s_p[2])
    for src in (adapter_stl, sensor_stl):
        tris = _stl_triangles(src)
        if tris.size == 0:
            raise ValueError(f"{src.name} 不包含任何三角面片")
        verts = tris.reshape(-1, 3)
        in_h = np.all(np.abs(verts - c_h) <= s_h + 1e-8, axis=1).reshape(-1, 3)
        in_p = np.all(np.abs(verts - c_p) <= s_p + 1e-8, axis=1).reshape(-1, 3)
        if not bool(np.all(in_h | in_p)):
            raise ValueError(f"{src.name} 存在超出碰撞盒包络的顶点")
        # 跨壳体与垫块交界面的三角形须按 z = z_split 裁剪并核对两侧子多边形分别落入对应凸盒。
        spanning = ~(np.all(in_h, axis=1) | np.all(in_p, axis=1))
        for tri in tris[spanning]:
            poly = [tri[0], tri[1], tri[2]]
            below = _clip_polygon_z(poly, z_split, keep_above=False)
            above = _clip_polygon_z(poly, z_split, keep_above=True)
            if below and not bool(np.all(np.abs(np.asarray(below) - c_h) <= s_h + 1e-8)):
                raise ValueError(f"{src.name} 存在跨交界面突出于碰撞盒包络的三角面片")
            if above and not bool(np.all(np.abs(np.asarray(above) - c_p) <= s_p + 1e-8)):
                raise ValueError(f"{src.name} 存在跨交界面突出于碰撞盒包络的三角面片")

    probe_root = ET.Element("mujoco")
    ET.SubElement(probe_root, "compiler", {"meshdir": str(paxini_export_dir.resolve())})
    asset = ET.SubElement(probe_root, "asset")
    worldbody = ET.SubElement(probe_root, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": "sensor_body"})
    for mesh_name, fname in (
        ("paxini_m2824", "paxini_m2824.stl"),
        ("paxini_m2824_adapter", "paxini_m2824_adapter.stl"),
    ):
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": fname})
        ET.SubElement(
            body,
            "geom",
            {
                "type": "mesh",
                "mesh": mesh_name,
                "contype": "0",
                "conaffinity": "0",
                "group": "2",
            },
        )
    mujoco.MjSpec.from_string(ET.tostring(probe_root, encoding="unicode")).compile()


def _copy_paxini_meshes(paxini_export_dir: Path, out_meshes_dir: Path) -> None:
    """将 PaXini M2824 传感器与转接垫块 STL 复制到 meshes/paxini_m2824/ 子目录。"""
    sub_dir = out_meshes_dir / "paxini_m2824"
    sub_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("paxini_m2824.stl", "paxini_m2824_adapter.stl"):
        src = paxini_export_dir / fname
        if not src.is_file():
            raise FileNotFoundError(f"缺少 PaXini 导出网格文件: {src}")
        shutil.copy2(src, sub_dir / fname)


def _write_indexed_stl(
    stl_path: Path,
    header_text: str,
    vertices: np.ndarray,
    faces: list[tuple[int, int, int]],
) -> None:
    """按共享顶点索引确定性写出闭合二进制 STL，确保相邻三角边精确共享同一 float32 坐标。"""
    verts_f32 = np.asarray(vertices, dtype=np.float32)
    header = header_text.encode("ascii")[:80].ljust(80, b"\0")
    chunks = [struct.pack("<80sI", header, len(faces))]
    for i0, i1, i2 in faces:
        v0, v1, v2 = verts_f32[i0], verts_f32[i1], verts_f32[i2]
        normal = np.cross(
            v1.astype(np.float64) - v0.astype(np.float64),
            v2.astype(np.float64) - v0.astype(np.float64),
        )
        norm = float(np.linalg.norm(normal))
        if norm > 0.0:
            normal /= norm
        chunks.append(
            struct.pack(
                "<12fH",
                float(normal[0]),
                float(normal[1]),
                float(normal[2]),
                float(v0[0]),
                float(v0[1]),
                float(v0[2]),
                float(v1[0]),
                float(v1[1]),
                float(v1[2]),
                float(v2[0]),
                float(v2[1]),
                float(v2[2]),
                0,
            )
        )
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    stl_path.write_bytes(b"".join(chunks))


def _generate_soft_dome_meshes(out_meshes_dir: Path) -> tuple[Path, Path]:
    """程序化生成 soft_dome 薄圆柱转接底座与扁球冠（半球为 h=R 特例）闭合流形二进制 STL。"""
    sub_dir = out_meshes_dir / "soft_dome"
    sub_dir.mkdir(parents=True, exist_ok=True)
    base_stl = sub_dir / "soft_dome_base.stl"
    dome_stl = sub_dir / "soft_dome.stl"

    # 1. 薄圆柱转接底座：直径 24 mm（半径 12 mm），z 从指腹安装面 -5.8 mm 延伸至 -3.8 mm（厚 2.0 mm）
    n_lon = 48
    ring_xy = [
        (
            SOFT_DOME_BASE_RADIUS_M * math.cos(2.0 * math.pi * k / n_lon),
            SOFT_DOME_BASE_RADIUS_M * math.sin(2.0 * math.pi * k / n_lon),
        )
        for k in range(n_lon)
    ]
    z0, z1 = SOFT_DOME_BACK_Z_M, SOFT_DOME_BASE_FRONT_Z_M
    base_verts = (
        [[x, y, z0] for x, y in ring_xy]
        + [[x, y, z1] for x, y in ring_xy]
        + [[0.0, 0.0, z0], [0.0, 0.0, z1]]
    )
    n_ring = len(ring_xy)
    bot_center, top_center = 2 * n_ring, 2 * n_ring + 1
    base_faces: list[tuple[int, int, int]] = []
    for k in range(n_ring):
        k1 = (k + 1) % n_ring
        base_faces.append((bot_center, k1, k))
        base_faces.append((top_center, n_ring + k, n_ring + k1))
        base_faces.append((k, k1, n_ring + k1))
        base_faces.append((k, n_ring + k1, n_ring + k))
    _write_indexed_stl(
        base_stl, "adaptive_tactile_gripper soft_dome_base", np.array(base_verts), base_faces
    )

    # 2. 硅胶扁球冠：球心 (0, 0, z_c)，曲率半径 R = 12 mm，截取 z in [z1, tip]（h_cap = R 时退化为半球）
    cap_height = SOFT_DOME_TIP_PROTRUSION_M - z1
    if not (0.0 < cap_height <= SOFT_DOME_RADIUS_M):
        raise ValueError(
            f"soft_dome 冠高 {cap_height} m 必须满足 0 < h <= R ({SOFT_DOME_RADIUS_M} m)"
        )
    cos_max = (z1 - SOFT_DOME_SPHERE_CENTER_Z_M) / SOFT_DOME_RADIUS_M
    theta_max = math.acos(max(-1.0, min(1.0, cos_max)))
    n_lat = 18
    dome_verts: list[list[float]] = [[0.0, 0.0, SOFT_DOME_TIP_PROTRUSION_M]]
    for r_idx in range(1, n_lat + 1):
        if r_idx == n_lat:
            z_ring = z1
            rad_ring = math.sqrt(
                max(0.0, SOFT_DOME_RADIUS_M**2 - (z1 - SOFT_DOME_SPHERE_CENTER_Z_M) ** 2)
            )
        else:
            theta = theta_max * (r_idx / n_lat)
            z_ring = SOFT_DOME_SPHERE_CENTER_Z_M + SOFT_DOME_RADIUS_M * math.cos(theta)
            rad_ring = SOFT_DOME_RADIUS_M * math.sin(theta)
        for j in range(n_lon):
            phi = 2.0 * math.pi * j / n_lon
            dome_verts.append([rad_ring * math.cos(phi), rad_ring * math.sin(phi), z_ring])
    dome_bot_center = len(dome_verts)
    dome_verts.append([0.0, 0.0, z1])

    dome_faces: list[tuple[int, int, int]] = []
    for j in range(n_lon):
        j1 = (j + 1) % n_lon
        dome_faces.append((0, 1 + j, 1 + j1))
    for r_idx in range(1, n_lat):
        b0 = 1 + (r_idx - 1) * n_lon
        b1 = 1 + r_idx * n_lon
        for j in range(n_lon):
            j1 = (j + 1) % n_lon
            dome_faces.append((b0 + j, b1 + j, b1 + j1))
            dome_faces.append((b0 + j, b1 + j1, b0 + j1))
    bn = 1 + (n_lat - 1) * n_lon
    for j in range(n_lon):
        j1 = (j + 1) % n_lon
        dome_faces.append((dome_bot_center, bn + j1, bn + j))
    _write_indexed_stl(
        dome_stl, "adaptive_tactile_gripper soft_dome", np.array(dome_verts), dome_faces
    )

    for stl_path in (base_stl, dome_stl):
        _assert_closed_manifold_stl(stl_path)
        _stl_polyhedral_props(stl_path)
    return base_stl, dome_stl


def _rename_mesh_refs(elem: ET.Element) -> None:
    """将元素树内的旧网格名称映射为纯 ASCII 规范名。"""
    for node in elem.iter():
        mesh = node.get("mesh")
        if mesh and mesh in MESH_NAME_MAP:
            node.set("mesh", MESH_NAME_MAP[mesh])


def _build_default_section(for_chassis: bool = True) -> ET.Element:
    """构造统一的 MuJoCo <default> 配置块。"""
    default = ET.Element("default")
    pg = ET.SubElement(default, "default", {"class": "parallel_gripper"})
    if for_chassis:
        ET.SubElement(pg, "joint", {"frictionloss": "0.1"})
    vis = ET.SubElement(pg, "default", {"class": "visual"})
    ET.SubElement(vis, "geom", {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "2"})
    col = ET.SubElement(pg, "default", {"class": "collision"})
    ET.SubElement(col, "geom", {"group": "3"})
    pcol = ET.SubElement(pg, "default", {"class": "pillar_collision"})
    ET.SubElement(
        pcol,
        "geom",
        {
            "group": "3",
            "contype": "8",
            "conaffinity": "9",
            "friction": "0.8 0.02 0.001",
            "solref": "-1200 -10",
            "solimp": "0.75 0.95 0.0025 0.5 2",
        },
    )
    return default


def _add_collision_box(
    body: ET.Element,
    name: str,
    pos: list[float],
    size: list[float],
    quat: list[float] | None = None,
) -> ET.Element:
    """在指定刚体下添加碰撞盒，尺寸为半长。"""
    geom = ET.SubElement(
        body,
        "geom",
        {
            "name": name,
            "class": "collision",
            "type": "box",
            "pos": _fmt_vec(pos),
            "size": _fmt_vec(size),
        },
    )
    if quat is not None:
        geom.set("quat", _fmt_vec(quat))
    return geom


def _add_bounding_box(
    body: ET.Element,
    name: str,
    vertices: np.ndarray,
    *,
    pos: np.ndarray | None = None,
    quat: list[float] | None = None,
) -> None:
    """在顶点所在坐标系拟合盒子，再将其写入父刚体系。"""
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = (lower + upper) / 2
    if quat is not None:
        center = _quat_wxyz_to_mat(quat) @ center
    if pos is not None:
        center += pos
    # 留 1 µm 给网格编译与 XML 序列化的数值舍入。
    _add_collision_box(body, name, center, (upper - lower) / 2 + 1e-6, quat)


def _build_appearance_library(raw_root: ET.Element, out_root: Path) -> dict:
    """按原始视觉几何实际引用的材质采集 CAD 色，合并内置默认映射并保留用户扩展。"""
    bundled = read_appearance(ROOT / "config" / "appearance.yaml")
    path = out_root / "config" / "appearance.yaml"
    if path.is_file():
        existing = read_appearance(path)
        presets = {}
        for name in dict.fromkeys([*bundled["presets"], *existing.get("presets", ())]):
            base_preset = bundled["presets"].get(name, {})
            user_preset = existing.get("presets", {}).get(name, {})
            merged_preset = {**base_preset, **user_preset}
            if "mesh_rgba" in base_preset or "mesh_rgba" in user_preset:
                merged_preset["mesh_rgba"] = {
                    **base_preset.get("mesh_rgba", {}),
                    **user_preset.get("mesh_rgba", {}),
                }
            presets[name] = merged_preset
        library = {
            **bundled,
            **existing,
            "materials": {**bundled["materials"], **existing.get("materials", {})},
            "mesh_classes": {**bundled["mesh_classes"], **existing.get("mesh_classes", {})},
            "smooth_meshes": list(existing["smooth_meshes"])
            if "smooth_meshes" in existing
            else list(bundled["smooth_meshes"]),
            "presets": presets,
        }
    else:
        library = bundled
    raw_colors = {
        mat.attrib["name"]: mat.attrib["rgba"] for mat in raw_root.findall("asset/material")
    }
    colors: dict[str, list[float]] = {}
    for geom in raw_root.findall(".//worldbody//geom[@class='visual']"):
        mesh = MESH_NAME_MAP[geom.attrib["mesh"]]
        if mesh not in library["mesh_classes"]:
            raise ValueError(f"视觉网格 {mesh!r} 缺少材质类别映射")
        color = [float(x) for x in raw_colors[geom.attrib["material"]].split()]
        if colors.setdefault(mesh, color) != color:
            raise ValueError(f"网格 {mesh!r} 的多个实例使用了不同的 CAD 色")
    library["presets"]["cad"]["mesh_rgba"] = dict(
        sorted((library["presets"]["cad"]["mesh_rgba"] | colors).items())
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    # 映射逐行输出，颜色向量保持在一行，方便核对导出差异。
    class LibraryDumper(yaml.SafeDumper):
        """以行内列表保存材质向量。"""

    LibraryDumper.add_representer(
        list,
        lambda dumper, seq: dumper.represent_sequence(
            "tag:yaml.org,2002:seq",
            seq,
            flow_style=True,
        ),
    )
    path.write_text(yaml.dump(library, Dumper=LibraryDumper, sort_keys=False), encoding="utf-8")
    return library


def _add_visual_assets(
    root: ET.Element, appearance: Appearance, mesh_files: dict[str, str] | None = None
) -> None:
    """为模组添加所引用的网格、曲面法线和语义材质。"""
    asset = ET.SubElement(root, "asset")
    meshes = {geom.attrib["mesh"] for geom in root.findall(".//worldbody//geom[@mesh]")}
    for mesh in sorted(meshes):
        file_rel = (mesh_files or {}).get(mesh, f"{mesh}.stl")
        element = ET.SubElement(asset, "mesh", name=mesh, file=file_rel)
        if mesh in appearance.smooth_meshes:
            element.set("smoothnormal", "true")
    # 内置模组在生成时显式标记视觉几何，按配置原始精度写入交付 XML。
    used = set()
    for geom in root.findall(".//worldbody//geom"):
        if geom.get("class") != "visual" and geom.get("group") != "2":
            continue
        mesh = geom.get("mesh") or geom.attrib["name"]
        category = appearance.category(mesh)
        used.add(category)
        geom.set("material", category)
        geom.attrib.pop("rgba", None)
        if mesh in appearance.mesh_rgba:
            geom.set("rgba", " ".join(map(str, appearance.mesh_rgba[mesh])))
    for category in sorted(used):
        material = ET.SubElement(asset, "material", name=category)
        for key, value in appearance.material_attributes(category).items():
            material.set(key, " ".join(map(str, value)) if key == "rgba" else str(value))


def _extract_sensor_mount_world_mats(raw_root: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    """计算左、右 sensor_mount 在导出初始零位下的世界变换矩阵。"""
    left_body = raw_root.find(".//body[@name='ls__mgn9_block__6faf12bd810aab5c7f7b6a160ef20d75']")
    right_body = raw_root.find(
        ".//body[@name='ls__mgn9_block__6faf12bd810aab5c7f7b6a160ef20d75_2']"
    )
    if left_body is None or right_body is None:
        raise ValueError("raw_gripper.xml 缺少左右滑块 body")

    T_left_body = _pose_to_mat(
        _parse_vec(left_body.get("pos"), [0, 0, 0]),
        _parse_vec(left_body.get("quat"), [1, 0, 0, 0]),
    )
    T_right_body = _pose_to_mat(
        _parse_vec(right_body.get("pos"), [0, 0, 0]),
        _parse_vec(right_body.get("quat"), [1, 0, 0, 0]),
    )
    T_left_mount = T_left_body @ _pose_to_mat(
        LEFT_SENSOR_MOUNT_IN_FINGER_POS, LEFT_SENSOR_MOUNT_IN_FINGER_QUAT
    )
    T_right_mount = T_right_body @ _pose_to_mat(
        RIGHT_SENSOR_MOUNT_IN_FINGER_POS, RIGHT_SENSOR_MOUNT_IN_FINGER_QUAT
    )
    return T_left_mount, T_right_mount


def _build_pts_2_0_mjcf(
    raw_root: ET.Element, appearance: Appearance, meshes_dir: Path, out_path: Path
) -> list[dict[str, object]]:
    """生成模块化传感器 mjcf/sensors/pts_2_0.xml（视觉保留网格，碰撞优先用 9 球体）。"""
    T_left_mount, _ = _extract_sensor_mount_world_mats(raw_root)
    T_mount_inv = np.linalg.inv(T_left_mount)

    left_body = raw_root.find(".//body[@name='ls__mgn9_block__6faf12bd810aab5c7f7b6a160ef20d75']")
    T_left_body = _pose_to_mat(
        _parse_vec(left_body.get("pos"), [0, 0, 0]),
        _parse_vec(left_body.get("quat"), [1, 0, 0, 0]),
    )

    root = ET.Element("mujoco", {"model": "pts_2_0"})
    ET.SubElement(
        root, "compiler", {"angle": "radian", "meshdir": "../../meshes", "autolimits": "true"}
    )
    root.append(_build_default_section(for_chassis=False))
    worldbody = ET.SubElement(root, "worldbody")
    sensor_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "sensor_body",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "childclass": "parallel_gripper",
        },
    )

    inertial_parts: list[InertialPart] = []
    housing_part: InertialPart | None = None
    for geom in left_body.findall("geom"):
        if geom.get("class") != "visual":
            continue
        mesh = geom.get("mesh", "")
        if mesh != "housing" and not mesh.startswith("pillars_"):
            continue
        T_geom_local = _pose_to_mat(
            _parse_vec(geom.get("pos"), [0, 0, 0]),
            _parse_vec(geom.get("quat"), [1, 0, 0, 0]),
        )
        T_rel = T_mount_inv @ (T_left_body @ T_geom_local)
        rel_pos_list = T_rel[:3, 3].tolist()
        rel_quat_list = _mat_to_quat_wxyz(T_rel[:3, :3])
        ET.SubElement(
            sensor_body,
            "geom",
            {
                "type": "mesh",
                "class": "visual",
                "pos": _fmt_vec(rel_pos_list),
                "quat": _fmt_vec(rel_quat_list),
                "mesh": mesh,
            },
        )
        if mesh == "housing":
            # Contactile PTS 2.0 单只传感头总重 15.0 g（外壳+PCB ~13.0 g，硅胶柱阵列 ~2.0 g）
            housing_part = _part_from_mesh(
                "pts_housing_and_pcb",
                meshes_dir / "housing.stl",
                rel_pos_list,
                rel_quat_list,
                mass_kg=0.0130,
            )
            inertial_parts.append(housing_part)

    housing = _add_collision_box(
        sensor_body,
        "sensor_housing_collision",
        [-0.0033, 0, -0.0015],
        [0.0153, 0.012, 0.0043],
    )
    # 保留与默认环境几何的接触，避免越过指间接触端点后壳体与对侧柱头或平垫发生附加碰撞。
    housing.set("contype", "4")
    housing.set("conaffinity", "1")

    # 抽取 9 个触觉站点并在同一坐标下生成 r=2.8mm 球形碰撞体
    taxels_spec: list[dict[str, object]] = []
    silicone_mass_each = (PTS_2_0_MASS_KG - 0.0130) / 9.0
    for row in range(3):
        for col in range(3):
            idx = f"{row}{col}"
            site = left_body.find(f"site[@name='left_taxel_{idx}']")
            if site is None:
                raise ValueError(f"缺少站点 left_taxel_{idx}")
            T_site_local = _pose_to_mat(
                _parse_vec(site.get("pos"), [0, 0, 0]),
                _parse_vec(site.get("quat"), [1, 0, 0, 0]),
            )
            T_rel = T_mount_inv @ (T_left_body @ T_site_local)
            pos_list = [round(float(v), 7) for v in T_rel[:3, 3]]
            pos_str = _fmt_vec(pos_list)
            ET.SubElement(
                sensor_body,
                "geom",
                {
                    "type": "sphere",
                    "class": "pillar_collision",
                    "name": f"taxel_geom_{idx}",
                    "pos": pos_str,
                    "size": "0.0028",
                },
            )
            ET.SubElement(
                sensor_body,
                "site",
                {
                    "group": "3",
                    "name": f"taxel_{idx}",
                    "pos": pos_str,
                    "quat": "0 1 0 0",
                },
            )
            r = 0.0028
            i_sph = 0.4 * silicone_mass_each * r * r
            inertial_parts.append(
                _point_part(f"pillar_{idx}", silicone_mass_each, pos_list, [i_sph, i_sph, i_sph])
            )
            taxels_spec.append({"row": row, "col": col, "pos_m": pos_list, "radius_m": 0.0028})

    ET.SubElement(
        sensor_body,
        "site",
        {
            "name": "fingertip",
            "group": "4",
            "pos": _fmt_vec([0, 0, TIP_PROTRUSION_M]),
        },
    )
    mass_pts, com_pts, I_pts = _combine_inertials(inertial_parts)
    _set_inertial_element(sensor_body, mass_pts, com_pts, I_pts)

    _add_visual_assets(root, appearance)
    _write_xml(root, out_path)
    return taxels_spec


def _build_flat_pad_mjcf(appearance: Appearance, out_path: Path) -> None:
    """生成无触觉平垫占位模组 mjcf/sensors/flat_pad.xml。"""
    root = ET.Element("mujoco", {"model": "flat_pad"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    root.append(_build_default_section(for_chassis=False))
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "sensor_body",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "childclass": "parallel_gripper",
        },
    )
    m = FLAT_PAD_MASS_KG
    sx, sy, sz = 0.024, 0.024, 0.0066
    ixx = m * (sy**2 + sz**2) / 12.0
    iyy = m * (sx**2 + sz**2) / 12.0
    izz = m * (sx**2 + sy**2) / 12.0
    ET.SubElement(
        body,
        "inertial",
        {"pos": "0 0 0.0033", "mass": f"{m:.6g}", "diaginertia": _fmt_vec([ixx, iyy, izz])},
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "pad_visual",
            "type": "box",
            "size": "0.012 0.012 0.0033",
            "pos": "0 0 0.0033",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "pad_collision",
            "type": "box",
            "class": "pillar_collision",
            "size": "0.012 0.012 0.0033",
            "pos": "0 0 0.0033",
        },
    )
    ET.SubElement(
        body,
        "site",
        {"group": "3", "name": "pad_center", "pos": "0 0 0.0066", "quat": "0 1 0 0"},
    )
    ET.SubElement(
        body,
        "site",
        {
            "name": "fingertip",
            "group": "4",
            "pos": _fmt_vec([0, 0, TIP_PROTRUSION_M]),
        },
    )
    _add_visual_assets(root, appearance)
    _write_xml(root, out_path)


def _build_paxini_m2824_mjcf(
    appearance: Appearance,
    meshes_dir: Path,
    rho_placf: float,
    taxels_raw: list[dict[str, object]],
    out_path: Path,
) -> float:
    """生成 PaXini PX6AX-GEN4-GR-M2824-Gripper（含转接垫块、纯盒式碰撞与 135 测点 site）MJCF。"""
    root = ET.Element("mujoco", {"model": "paxini_m2824"})
    ET.SubElement(
        root, "compiler", {"angle": "radian", "meshdir": "../../meshes", "autolimits": "true"}
    )
    root.append(_build_default_section(for_chassis=False))
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "sensor_body",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "childclass": "parallel_gripper",
        },
    )

    adapter_stl = meshes_dir / "paxini_m2824" / "paxini_m2824_adapter.stl"
    sensor_stl = meshes_dir / "paxini_m2824" / "paxini_m2824.stl"
    adapter_part = _part_from_mesh(
        "paxini_m2824_adapter",
        adapter_stl,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        density_kg_m3=rho_placf,
    )
    m_s = PAXINI_M2824_SENSOR_MASS_KG
    sx, sy, sz = 0.028752, 0.024021, 0.0060025
    sensor_part = _point_part(
        "paxini_m2824_sensor",
        m_s,
        [-6.8e-05, 0.0, 0.00069875],
        [
            m_s * (sy**2 + sz**2) / 12.0,
            m_s * (sx**2 + sz**2) / 12.0,
            m_s * (sx**2 + sy**2) / 12.0,
        ],
    )
    total_mass, com, I_mat = _combine_inertials([adapter_part, sensor_part])
    _set_inertial_element(body, total_mass, com, I_mat)

    for mesh_name in ("paxini_m2824_adapter", "paxini_m2824"):
        ET.SubElement(
            body,
            "geom",
            {
                "type": "mesh",
                "class": "visual",
                "pos": "0 0 0",
                "quat": "1 0 0 0",
                "mesh": mesh_name,
            },
        )

    housing_pos = PAXINI_M2824_HOUSING_POS
    housing_size = PAXINI_M2824_HOUSING_SIZE
    housing = _add_collision_box(body, "sensor_housing_collision", housing_pos, housing_size)
    housing.set("contype", "4")
    housing.set("conaffinity", "1")

    pad_pos = PAXINI_M2824_PAD_POS
    pad_size = PAXINI_M2824_PAD_SIZE
    ET.SubElement(
        body,
        "geom",
        {
            "name": "pad_collision",
            "type": "box",
            "class": "pillar_collision",
            "pos": _fmt_vec(pad_pos),
            "size": _fmt_vec(pad_size),
        },
    )

    for stl_path in (adapter_stl, sensor_stl):
        verts = _stl_triangles(stl_path).reshape(-1, 3)
        c_h, s_h = np.asarray(housing_pos), np.asarray(housing_size)
        c_p, s_p = np.asarray(pad_pos), np.asarray(pad_size)
        covered = np.all(np.abs(verts - c_h) <= s_h + 1e-8, axis=1) | np.all(
            np.abs(verts - c_p) <= s_p + 1e-8, axis=1
        )
        if not bool(covered.all()):
            raise ValueError(f"{stl_path.name} 存在超出碰撞盒包络的顶点")

    ET.SubElement(
        body,
        "site",
        {
            "group": "3",
            "name": "pad_center",
            "pos": _fmt_vec([0, 0, PAXINI_M2824_TIP_PROTRUSION_M]),
            "quat": "0 1 0 0",
        },
    )
    for item in taxels_raw:
        tid = int(item["id"])  # type: ignore[arg-type]
        pos_m = [float(v) for v in item["pos_m"]]  # type: ignore[attr-defined]
        ET.SubElement(
            body,
            "site",
            {
                "group": "3",
                "name": f"taxel_{tid:03d}",
                "pos": _fmt_vec(pos_m),
                "quat": "0 1 0 0",
                "size": "0.0006",
            },
        )
    ET.SubElement(
        body,
        "site",
        {
            "name": "fingertip",
            "group": "4",
            "pos": _fmt_vec([0, 0, PAXINI_M2824_TIP_PROTRUSION_M]),
        },
    )

    _add_visual_assets(
        root,
        appearance,
        {
            "paxini_m2824": "paxini_m2824/paxini_m2824.stl",
            "paxini_m2824_adapter": "paxini_m2824/paxini_m2824_adapter.stl",
        },
    )
    _write_xml(root, out_path)
    return total_mass


def _build_soft_dome_mjcf(
    appearance: Appearance,
    meshes_dir: Path,
    rho_placf: float,
    out_path: Path,
) -> float:
    """生成硅胶扁球冠六轴 F/T 模组 mjcf/sensors/soft_dome.xml（球面碰撞代理 + 扁球冠视觉网格）。"""
    root = ET.Element("mujoco", {"model": "soft_dome"})
    ET.SubElement(
        root, "compiler", {"angle": "radian", "meshdir": "../../meshes", "autolimits": "true"}
    )
    root.append(_build_default_section(for_chassis=False))
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "sensor_body",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "childclass": "parallel_gripper",
        },
    )

    base_stl = meshes_dir / "soft_dome" / "soft_dome_base.stl"
    dome_stl = meshes_dir / "soft_dome" / "soft_dome.stl"
    base_part = _part_from_mesh(
        "soft_dome_base",
        base_stl,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        density_kg_m3=rho_placf,
    )
    dome_part = _part_from_mesh(
        "soft_dome",
        dome_stl,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        density_kg_m3=SOFT_DOME_SILICONE_DENSITY_KG_M3,
    )
    m_ft = SOFT_DOME_TRANSDUCER_MASS_KG
    r_ft, h_ft = 0.009, 0.002
    transducer_part = _point_part(
        "soft_dome_ft_core",
        m_ft,
        [0.0, 0.0, -0.0048],
        [
            m_ft * (3.0 * r_ft**2 + h_ft**2) / 12.0,
            m_ft * (3.0 * r_ft**2 + h_ft**2) / 12.0,
            0.5 * m_ft * r_ft**2,
        ],
    )
    total_mass, com, I_mat = _combine_inertials([base_part, dome_part, transducer_part])
    _set_inertial_element(body, total_mass, com, I_mat)

    for mesh_name in ("soft_dome_base", "soft_dome"):
        ET.SubElement(
            body,
            "geom",
            {
                "type": "mesh",
                "class": "visual",
                "pos": "0 0 0",
                "quat": "1 0 0 0",
                "mesh": mesh_name,
            },
        )

    ET.SubElement(
        body,
        "geom",
        {
            "name": "sensor_housing_collision",
            "class": "collision",
            "type": "cylinder",
            "pos": _fmt_vec(SOFT_DOME_HOUSING_POS),
            "size": _fmt_vec(SOFT_DOME_HOUSING_SIZE),
            "contype": "4",
            "conaffinity": "1",
        },
    )

    sphere_center = [0.0, 0.0, SOFT_DOME_SPHERE_CENTER_Z_M]
    ET.SubElement(
        body,
        "geom",
        {
            "name": "pad_collision",
            "type": "sphere",
            "class": "pillar_collision",
            "pos": _fmt_vec(sphere_center),
            "size": _fmt_vec([SOFT_DOME_RADIUS_M]),
            "condim": "4",
            "friction": "0.8 0.0064 0.0001",
            "solref": "-4800 -24",
            "solimp": "0.75 0.97 0.002 0.5 2",
        },
    )

    c_h = np.asarray(SOFT_DOME_HOUSING_POS)
    r_h, hz_h = SOFT_DOME_HOUSING_SIZE
    c_s = np.asarray(sphere_center)
    base_verts = _stl_triangles(base_stl).reshape(-1, 3)
    base_radial = np.hypot(base_verts[:, 0] - c_h[0], base_verts[:, 1] - c_h[1])
    if not bool(
        np.all((base_radial <= r_h + 1e-8) & (np.abs(base_verts[:, 2] - c_h[2]) <= hz_h + 1e-8))
    ):
        raise ValueError("soft_dome_base.stl 存在超出圆柱底座碰撞包络的顶点")
    dome_verts = _stl_triangles(dome_stl).reshape(-1, 3)
    if not bool(np.all(np.linalg.norm(dome_verts - c_s, axis=1) <= SOFT_DOME_RADIUS_M + 1e-7)):
        raise ValueError("soft_dome.stl 存在超出球面碰撞体包络的顶点")

    ET.SubElement(
        body,
        "site",
        {
            "group": "3",
            "name": "pad_center",
            "pos": _fmt_vec([0, 0, SOFT_DOME_TIP_PROTRUSION_M]),
            "quat": "0 1 0 0",
        },
    )
    ET.SubElement(
        body,
        "site",
        {
            "group": "3",
            "name": "ft_sensor",
            "pos": _fmt_vec([0, 0, SOFT_DOME_BASE_FRONT_Z_M]),
            "quat": "0 1 0 0",
        },
    )
    ET.SubElement(
        body,
        "site",
        {
            "name": "fingertip",
            "group": "4",
            "pos": _fmt_vec([0, 0, SOFT_DOME_TIP_PROTRUSION_M]),
        },
    )

    _add_visual_assets(
        root,
        appearance,
        {
            "soft_dome": "soft_dome/soft_dome.stl",
            "soft_dome_base": "soft_dome/soft_dome_base.stl",
        },
    )
    _write_xml(root, out_path)
    return total_mass


def _compute_d435i_frames(
    cam_pos: list[float], cam_quat: list[float]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """计算与 Intel realsense2_description (_d435i.urdf.xacro) 一致的 D435i 标准坐标帧（位于 base 系下）。

    d435i_solid.stl 局部坐标原点位于前玻璃面板中心，+z_m 朝镜头外。
    ROS 相机本体坐标：X_ros = +z_m（视线前），Y_ros = +x_m（左，RGB 偏置 +15mm），Z_ros = +y_m（上）。
    """
    T_stl = _pose_to_mat(cam_pos, cam_quat)
    R_stl = T_stl[:3, :3]
    # 从 ROS 相机系 (X_ros, Y_ros, Z_ros) 映射到 STL 局部系的基底矩阵：
    # X_ros -> [0,0,1], Y_ros -> [1,0,0], Z_ros -> [0,1,0]
    R_ros_in_stl = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    R_ros_base = R_stl @ R_ros_in_stl

    # d435i_link 位于前玻璃面内缩 4.3mm、沿 +Y_ros 偏移 +17.5mm（左 IR / 深度光心）
    # 按 STL 坐标即 x_m=+0.0175, y_m=0.0, z_m=-0.0043
    T_link = np.eye(4, dtype=float)
    T_link[:3, :3] = R_ros_base
    T_link[:3, 3] = T_stl[:3, 3] + R_stl @ np.array([0.0175, 0.0, -0.0043], dtype=float)

    def _offset_ros(
        dx: float, dy: float, dz: float, optical: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        pos = T_link[:3, 3] + R_ros_base @ np.array([dx, dy, dz], dtype=float)
        if optical:
            # ROS 光学坐标系：+Z_opt = +X_ros, +X_opt = -Y_ros, +Y_opt = -Z_ros
            R_opt = np.column_stack([-R_ros_base[:, 1], -R_ros_base[:, 2], R_ros_base[:, 0]])
            return pos, R_opt
        return pos, R_ros_base

    return {
        "bottom_screw": (
            T_stl[:3, 3] + R_stl @ np.array([0.0, -0.0125, -0.0149], dtype=float),
            R_ros_base,
        ),
        "link": (T_link[:3, 3], R_ros_base),
        "depth": _offset_ros(0.0, 0.0, 0.0, optical=False),
        "depth_optical": _offset_ros(0.0, 0.0, 0.0, optical=True),
        "infra1": _offset_ros(0.0, 0.0, 0.0, optical=False),
        "infra1_optical": _offset_ros(0.0, 0.0, 0.0, optical=True),
        "infra2": _offset_ros(0.0, -0.050, 0.0, optical=False),
        "infra2_optical": _offset_ros(0.0, -0.050, 0.0, optical=True),
        "color": _offset_ros(0.0, 0.015, 0.0, optical=False),
        "color_optical": _offset_ros(0.0, 0.015, 0.0, optical=True),
        "imu": _offset_ros(-0.01174, -0.00552, 0.0051, optical=False),
        "accel": _offset_ros(-0.01174, -0.00552, 0.0051),
        "accel_optical": _offset_ros(-0.01174, -0.00552, 0.0051, optical=True),
        "gyro": _offset_ros(-0.01174, -0.00552, 0.0051),
        "gyro_optical": _offset_ros(-0.01174, -0.00552, 0.0051, optical=True),
    }


def _build_realsense_d435i_mjcf(
    raw_root: ET.Element,
    appearance: Appearance,
    meshes_dir: Path,
    rho_placf: float,
    out_path: Path,
) -> dict[str, object]:
    """生成对齐官方 D435i 参数的 mjcf/accessories/realsense_d435i.xml。"""
    raw_base = raw_root.find(".//body[@name='base_assembly']")
    root = ET.Element("mujoco", {"model": "realsense_d435i"})
    ET.SubElement(
        root, "compiler", {"angle": "radian", "meshdir": "../../meshes", "autolimits": "true"}
    )
    root.append(_build_default_section(for_chassis=False))
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "camera_assembly",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "childclass": "parallel_gripper",
        },
    )

    cam_mount_pos = [0.0, 0.0, 0.007]
    cam_mount_quat = [1.0, 0.0, 0.0, 0.0]
    cam_pos = [0.0, -0.0685273, 0.0279714]
    cam_quat = [0.0, 0.0, -0.121869, -0.992546]
    for geom in raw_base.findall("geom"):
        if geom.get("class") != "visual":
            continue
        mesh = geom.get("mesh", "")
        if mesh not in ("camera_mount", "d435i_solid"):
            continue
        pos = geom.get("pos", "0 0 0")
        quat = geom.get("quat", "1 0 0 0")
        if mesh == "camera_mount":
            cam_mount_pos = _parse_vec(pos, cam_mount_pos)
            cam_mount_quat = _parse_vec(quat, cam_mount_quat)
        elif mesh == "d435i_solid":
            cam_pos = _parse_vec(pos, cam_pos)
            cam_quat = _parse_vec(quat, cam_quat)
        ET.SubElement(
            body,
            "geom",
            {
                "type": "mesh",
                "class": "visual",
                "pos": pos,
                "quat": quat,
                "mesh": mesh,
            },
        )

    # 估计相机模组惯量（PLA-CF 支架、72 g 裸机和支架紧固件）
    R_stl = _quat_wxyz_to_mat(cam_quat)
    p_stl = np.asarray(cam_pos, dtype=float)
    d435i_com_base = p_stl + R_stl @ D435I_COM_IN_STL_M
    R_iquat = _quat_wxyz_to_mat(D435I_IQUAT_IN_STL_WXYZ)
    d435i_I_stl = R_iquat @ np.diag(D435I_DIAGINERTIA_LOCAL) @ R_iquat.T
    d435i_I_base = R_stl @ d435i_I_stl @ R_stl.T
    mount_vertices = _stl_triangles(meshes_dir / "camera_mount.stl").reshape(-1, 3)
    _add_bounding_box(
        body, "camera_mount_collision", mount_vertices, pos=cam_mount_pos, quat=cam_mount_quat
    )
    camera_vertices = _stl_triangles(meshes_dir / "d435i_solid.stl").reshape(-1, 3)
    nominal_corners = [[-0.045, -0.0125, -0.02505], [0.045, 0.0125, 0]]
    _add_bounding_box(
        body,
        "d435i_housing_collision",
        np.vstack((camera_vertices, nominal_corners)),
        pos=p_stl,
        quat=cam_quat,
    )
    cam_parts = [
        _part_from_mesh(
            "camera_mount",
            meshes_dir / "camera_mount.stl",
            cam_mount_pos,
            cam_mount_quat,
            density_kg_m3=rho_placf,
        ),
        InertialPart("d435i_bare", D435I_BARE_MASS_KG, d435i_com_base, d435i_I_base),
        _point_part("camera_fasteners", CAMERA_FASTENERS_MASS_KG, [0.0, -0.064, 0.002]),
    ]
    cam_mass, cam_com, cam_I = _combine_inertials(cam_parts)
    _set_inertial_element(body, cam_mass, cam_com, cam_I)

    frames = _compute_d435i_frames(cam_pos, cam_quat)
    # MuJoCo 相机沿 -Z_cam 视向、+Y_cam 朝图像上方、+X_cam 朝图像右方
    # 对应 ROS optical frame (+Z_opt 朝前, +X_opt 朝右, +Y_opt 朝下): x_cam = +X_opt, y_cam = -Y_opt
    for cam_name, frame_key, fovy in (
        ("d435i_color", "color_optical", "42.5"),
        ("d435i_depth", "depth_optical", "58.0"),
    ):
        pos_c, R_opt = frames[frame_key]
        x_cam = R_opt[:, 0]
        y_cam = -R_opt[:, 1]
        ET.SubElement(
            body,
            "camera",
            {
                "name": cam_name,
                "pos": _fmt_vec(pos_c.tolist()),
                "xyaxes": _fmt_vec(x_cam.tolist() + y_cam.tolist()),
                "fovy": fovy,
            },
        )

    # 以 group=4 记录关键参考站点（避免干扰 group=3 触觉检测）
    for key, (s_pos, s_rot) in frames.items():
        site_name = "d435i_link" if key == "link" else f"d435i_{key}_frame"
        ET.SubElement(
            body,
            "site",
            {
                "group": "4",
                "name": site_name,
                "pos": _fmt_vec(s_pos.tolist()),
                "quat": _fmt_vec(_mat_to_quat_wxyz(s_rot)),
            },
        )

    _add_visual_assets(root, appearance)
    _write_xml(root, out_path)
    return {"mass_kg": cam_mass}


def _build_chassis_mjcf(
    raw_root: ET.Element,
    appearance: Appearance,
    meshes_dir: Path,
    rho_pla: float,
    rho_placf: float,
    out_path: Path,
) -> dict[str, float]:
    """生成以实测整机总质量约束惯量估计的纯夹爪底盘 MJCF。"""
    root = ET.Element("mujoco", {"model": "parallel_gripper"})
    ET.SubElement(
        root, "compiler", {"angle": "radian", "meshdir": "../meshes", "autolimits": "true"}
    )
    root.append(_build_default_section(for_chassis=True))

    worldbody = ET.SubElement(root, "worldbody")
    raw_base = raw_root.find(".//body[@name='base_assembly']")
    base = ET.SubElement(
        worldbody,
        "body",
        {"name": "base", "pos": "0 0 0", "quat": "1 0 0 0", "childclass": "parallel_gripper"},
    )
    ET.SubElement(base, "freejoint", {"name": "base_freejoint"})

    base_parts: list[InertialPart] = []

    base_mesh_masses = {
        "stator": ("mass", DM4310P_STATOR_MASS_KG),
        "mgn9_rail": ("mass", MGN9_RAIL_150MM_MASS_KG),
        "franka_adapter": ("density", rho_placf),
        "motor_mount": ("density", rho_placf),
        "bracket_2": ("density", rho_placf),
    }

    for geom in raw_base.findall("geom"):
        mesh = geom.get("mesh", "")
        if mesh in ("camera_mount", "d435i_solid"):
            continue
        cls = geom.get("class", "visual")
        g_el = ET.fromstring(ET.tostring(geom))
        _rename_mesh_refs(g_el)
        if cls == "visual":
            base.append(g_el)
            canon = g_el.get("mesh", "")
            mode, val = base_mesh_masses[canon]
            p = _parse_vec(g_el.get("pos"), [0, 0, 0])
            q = _parse_vec(g_el.get("quat"), [1, 0, 0, 0])
            base_parts.append(
                _part_from_mesh(
                    canon,
                    meshes_dir / f"{canon}.stl",
                    p,
                    q,
                    mass_kg=val if mode == "mass" else None,
                    density_kg_m3=val if mode == "density" else None,
                )
            )

    base_radius = 0.045
    ET.SubElement(
        base,
        "geom",
        {
            "name": "base_collision",
            "class": "collision",
            "type": "cylinder",
            "pos": "0 0 0.03",
            "size": _fmt_vec([base_radius, 0.031]),
        },
    )
    _add_collision_box(base, "rail_bridge_collision", [0, 0, 0.071], [0.079, 0.038, 0.014])
    # 转接板凸耳超出底座圆柱。纳入整片跨界三角形，让盒子与圆柱连续搭接。
    adapter = base.find("geom[@mesh='franka_adapter']")
    adapter_rotation = _quat_wxyz_to_mat(_parse_vec(adapter.get("quat"), [1, 0, 0, 0]))
    adapter_triangles = _stl_triangles(meshes_dir / "franka_adapter.stl") @ adapter_rotation.T
    adapter_triangles += _parse_vec(adapter.get("pos"), [0, 0, 0])
    outside = np.linalg.norm(adapter_triangles[:, :, :2], axis=2) > base_radius
    tab_triangles = adapter_triangles[outside.any(axis=1)]
    if tab_triangles.size:
        _add_bounding_box(base, "adapter_tab_collision", tab_triangles.reshape(-1, 3))

    # 分摊底座内部紧固件质量（M4 螺钉螺母固定 adapter/motor_mount，M3x12 固定电机/导轨）
    base_parts.append(_point_part("base_fasteners", BASE_FASTENERS_MASS_KG, [0.0, 0.0, 0.042]))
    base_mass, base_com, base_I = _combine_inertials(base_parts)
    _set_inertial_element(base, base_mass, base_com, base_I)

    ET.SubElement(base, "site", {"name": "tcp", "group": "4", "pos": _fmt_vec(TCP_IN_FLANGE_M)})
    # 预留相机安装 frame
    ET.SubElement(base, "frame", {"name": "camera_mount_frame", "pos": "0 0 0", "quat": "1 0 0 0"})

    # 拷贝并改造 motor 与两根连杆子装配体
    raw_motor = raw_root.find(".//body[@name='motor']")
    motor = ET.fromstring(ET.tostring(raw_motor))
    _rename_mesh_refs(motor)

    motor_parts: list[InertialPart] = []
    for geom in list(motor.findall("geom")):
        if geom.get("class") != "visual":
            motor.remove(geom)
        else:
            canon = geom.get("mesh", "")
            p = _parse_vec(geom.get("pos"), [0, 0, 0])
            q = _parse_vec(geom.get("quat"), [1, 0, 0, 0])
            if canon == "motor":
                motor_parts.append(
                    _part_from_mesh(
                        "motor_output",
                        meshes_dir / "motor.stl",
                        p,
                        q,
                        mass_kg=DM4310P_ROTOR_OUTPUT_MASS_KG,
                    )
                )
            elif canon == "crank":
                motor_parts.append(
                    _part_from_mesh(
                        "crank", meshes_dir / "crank.stl", p, q, density_kg_m3=rho_placf
                    )
                )
    motor_parts.append(_point_part("motor_fasteners", MOTOR_FASTENERS_MASS_KG, [0.0, 0.0, 0.022]))
    motor_mass, motor_com, motor_I = _combine_inertials(motor_parts)
    _set_inertial_element(motor, motor_mass, motor_com, motor_I)

    # 估计连杆子装配体惯量，并移除内部连杆碰撞体。
    rod_masses: list[float] = []
    for rod_side, rod_name in (
        ("left", "rod_bearing_shim_subassembly"),
        ("right", "rod_bearing_shim_subassembly_2"),
    ):
        rod = motor.find(f"body[@name='{rod_name}']")
        rod_parts: list[InertialPart] = []
        spacer_idx = 0
        crank_spacer_found = False
        for geom in list(rod.findall("geom")):
            if geom.get("class") != "visual":
                rod.remove(geom)
            else:
                canon = geom.get("mesh", "")
                p = _parse_vec(geom.get("pos"), [0, 0, 0])
                q = _parse_vec(geom.get("quat"), [1, 0, 0, 0])
                if canon == "link":
                    rod_parts.append(
                        _part_from_mesh(
                            "link", meshes_dir / "link.stl", p, q, density_kg_m3=rho_placf
                        )
                    )
                elif canon.startswith("bearing_"):
                    rod_parts.append(
                        _part_from_mesh(
                            canon, meshes_dir / f"{canon}.stl", p, q, mass_kg=MF84_BEARING_MASS_KG
                        )
                    )
                elif canon == "link_spacer":
                    spacer_idx += 1
                    # 2 个销轴+垫片+M3x5 共 4.76g，均分到两个销轴位置
                    rod_parts.append(
                        _part_from_mesh(
                            f"pin_hw_{spacer_idx}",
                            meshes_dir / "link_spacer.stl",
                            p,
                            q,
                            mass_kg=ROD_SPACER_HARDWARE_MASS_KG / 2.0,
                        )
                    )
                    if float(np.hypot(p[0], p[1])) < 1e-3:
                        geom.set("name", f"{rod_side}_crank_spacer")
                        crank_spacer_found = True
        assert crank_spacer_found
        rm, rcom, rI = _combine_inertials(rod_parts)
        _set_inertial_element(rod, rm, rcom, rI)
        rod_masses.append(rm)

    # 配置 DM-J4310P-2EC 电机参数并紧随串联 gripper_backlash
    drive_jnt = motor.find("joint[@name='gripper_drive']")
    drive_jnt.set("range", f"0 {POSITION_MAX_RAD:.16g}")
    drive_jnt.set("frictionloss", "0.04")
    drive_jnt.set("damping", "0.02532329")
    drive_jnt.set("armature", "0.002074755")
    drive_jnt.set("solreflimit", "0.002 1")
    drive_jnt.set("solimplimit", "0.99 0.999 0.0001 0.5 2")
    backlash = ET.Element(
        "joint",
        {
            "axis": drive_jnt.get("axis", "0 0 1"),
            "name": "gripper_backlash",
            "type": "hinge",
            "range": "-0.0025 0.0025",
            "frictionloss": "0",
            "damping": "0",
            "armature": "0",
            "solreflimit": "0.002 1",
            "solimplimit": "0.99 0.999 0.0001 0.5 2",
        },
    )
    motor_children = list(motor)
    idx_drive = motor_children.index(drive_jnt)
    motor.insert(idx_drive + 1, backlash)

    for jnt in motor.findall(".//joint"):
        if jnt.get("name") in ("left_link_crank_pin", "right_link_crank_pin"):
            jnt.set("frictionloss", "0.0")
            jnt.set("damping", "0.01")
            jnt.set("armature", "0")

    # 修正左侧连杆闭环站点并补齐 _z 站点
    left_rod = motor.find(".//body[@name='rod_bearing_shim_subassembly']")
    s2 = left_rod.find("site[@name='closing_left_link_finger_pin_2']")
    s2.set("pos", "-0.0435355339 0 -0.0079")
    ET.SubElement(
        left_rod,
        "site",
        {
            "group": "3",
            "name": "closing_left_link_finger_pin_2_z",
            "pos": "-0.0435355339 0 -0.1079",
            "quat": "0 1 0 0",
        },
    )
    base.append(motor)

    # 拷贝并改造左右滑块指体（彻底剥离传感器质量，独立重算滑块+PLA指体+PLA-CF座惯量）
    finger_specs = [
        (
            "ls__mgn9_block__6faf12bd810aab5c7f7b6a160ef20d75",
            "left_finger",
            "left_sensor_mount",
            LEFT_SENSOR_MOUNT_IN_FINGER_POS,
            LEFT_SENSOR_MOUNT_IN_FINGER_QUAT,
            True,
            [0.0, 0.020, 0.014],
        ),
        (
            "ls__mgn9_block__6faf12bd810aab5c7f7b6a160ef20d75_2",
            "right_finger",
            "right_sensor_mount",
            RIGHT_SENSOR_MOUNT_IN_FINGER_POS,
            RIGHT_SENSOR_MOUNT_IN_FINGER_QUAT,
            False,
            [0.0, -0.020, 0.014],
        ),
    ]
    finger_masses: list[float] = []
    for raw_name, new_name, frame_name, f_pos, f_quat, is_left, fast_pos in finger_specs:
        raw_finger = raw_root.find(f".//body[@name='{raw_name}']")
        finger = ET.fromstring(ET.tostring(raw_finger))
        finger.set("name", new_name)
        _rename_mesh_refs(finger)

        jnt = finger.find("joint")
        jnt.set("frictionloss", "0.1")
        jnt.set("damping", "0.01")
        jnt.set("armature", "0")
        jnt.set("range", f"0 {SINGLE_FINGER_STROKE_M:.16g}")
        jnt.set("solreflimit", "0.002 1")
        jnt.set("solimplimit", "0.99 0.999 0.0001 0.5 2")

        finger_parts: list[InertialPart] = []
        for geom in list(finger.findall("geom")):
            mesh = geom.get("mesh", "")
            if geom.get("class") != "visual" or mesh == "housing" or mesh.startswith("pillars_"):
                finger.remove(geom)
            else:
                p = _parse_vec(geom.get("pos"), [0, 0, 0])
                q = _parse_vec(geom.get("quat"), [1, 0, 0, 0])
                if mesh == "mgn9_block":
                    finger_parts.append(
                        _part_from_mesh(
                            mesh, meshes_dir / f"{mesh}.stl", p, q, mass_kg=MGN9C_BLOCK_MASS_KG
                        )
                    )
                elif mesh == "finger_body":
                    finger_parts.append(
                        _part_from_mesh(
                            mesh, meshes_dir / f"{mesh}.stl", p, q, density_kg_m3=rho_pla
                        )
                    )
                else:
                    finger_parts.append(
                        _part_from_mesh(
                            mesh, meshes_dir / f"{mesh}.stl", p, q, density_kg_m3=rho_placf
                        )
                    )

        finger_parts.append(
            _point_part(f"{new_name}_fasteners", FINGER_FASTENERS_MASS_KG, fast_pos)
        )
        fm, fcom, fI = _combine_inertials(finger_parts)
        _set_inertial_element(finger, fm, fcom, fI)
        finger_masses.append(fm)

        for site in list(finger.findall("site")):
            sname = site.get("name", "")
            if "_taxel_" in sname:
                finger.remove(site)

        if is_left:
            ET.SubElement(
                finger,
                "site",
                {
                    "group": "3",
                    "name": "closing_left_link_finger_pin_1_z",
                    "pos": "-0.0176777 0.0955 0.0137868",
                    "quat": "0.5 -0.5 -0.5 -0.5",
                },
            )

        ET.SubElement(
            finger,
            "frame",
            {"name": frame_name, "pos": _fmt_vec(f_pos), "quat": _fmt_vec(f_quat)},
        )
        side = "left" if is_left else "right"
        for geom in finger.findall("geom"):
            mesh = geom.get("mesh")
            rotation = _quat_wxyz_to_mat(_parse_vec(geom.get("quat"), [1, 0, 0, 0]))
            vertices = _stl_triangles(meshes_dir / f"{mesh}.stl").reshape(-1, 3) @ rotation.T
            vertices += _parse_vec(geom.get("pos"), [0, 0, 0])
            # 指体系 +Z 朝向安装面，URDF 沿该轴只向外扩展碰撞盒。
            _add_bounding_box(finger, f"{side}_{mesh}_back_collision", vertices)
        base.append(finger)

    _add_visual_assets(root, appearance)

    actuator = ET.SubElement(root, "actuator")
    ET.SubElement(
        actuator,
        "motor",
        {
            "class": "parallel_gripper",
            "name": "gripper_drive",
            "joint": "gripper_drive",
            "gear": "1",
            "ctrlrange": "-4 4",
            "forcerange": "-4 4",
        },
    )

    contact = ET.SubElement(root, "contact")
    for side in ("left", "right"):
        ET.SubElement(contact, "exclude", {"body1": "base", "body2": f"{side}_finger"})
    for rod_side, opp_side in (("left", "right"), ("right", "left")):
        # 曲柄销轴套与对侧滑块碰撞盒的内部限位止档（q <= 1.7 rad 时净距 1.75 mm > margin 1.70 mm，无接触），
        # 在空载急闭过冲超出 1.7 rad 时阻止曲柄轴套撞入对侧滑块。
        ET.SubElement(
            contact,
            "pair",
            {
                "name": f"chassis_{rod_side}_spacer_stop",
                "geom1": f"{rod_side}_crank_spacer",
                "geom2": f"{opp_side}_mgn9_block_back_collision",
                "condim": "1",
                "margin": "0.0017",
                "solref": "0.002 1",
                "solimp": "0.99 0.999 0.0001 0.5 2",
            },
        )

    equality = ET.SubElement(root, "equality")
    # 使用刚体锚点定义闭环，同时保留站点用于核对闭环误差。
    for side, rod_name in (
        ("left", "rod_bearing_shim_subassembly"),
        ("right", "rod_bearing_shim_subassembly_2"),
    ):
        finger = base.find(f"body[@name='{side}_finger']")
        for suffix in ("", "_z"):
            anchor = finger.find(f"site[@name='closing_{side}_link_finger_pin_1{suffix}']").get(
                "pos"
            )
            ET.SubElement(
                equality,
                "connect",
                {
                    "body1": finger.get("name"),
                    "body2": rod_name,
                    "anchor": anchor,
                    "solref": "0.002 1",
                    "solimp": "0.99 0.999 0.0005 0.5 2",
                },
            )

    _write_xml(root, out_path)
    return {
        "base_kg": base_mass,
        "motor_kg": motor_mass,
        "left_rod_kg": rod_masses[0],
        "right_rod_kg": rod_masses[1],
        "left_finger_kg": finger_masses[0],
        "right_finger_kg": finger_masses[1],
        "chassis_total_kg": base_mass + motor_mass + sum(rod_masses) + sum(finger_masses),
    }


def _assemble_single_mjcf(
    chassis_path: Path,
    left_sensor_path: Path | None,
    right_sensor_path: Path | None,
    camera_path: Path | None,
    out_path: Path,
) -> None:
    """用 MjSpec 组装自包含的单文件 MJCF 并规范化输出路径与属性。"""
    spec = mujoco.MjSpec.from_file(str(chassis_path))

    if camera_path is not None:
        cam_spec = mujoco.MjSpec.from_file(str(camera_path))
        spec.attach(cam_spec, prefix="cam_", frame=spec.frame("camera_mount_frame"))

    for side, s_path in (("left", left_sensor_path), ("right", right_sensor_path)):
        if s_path is None:
            continue
        s_spec = mujoco.MjSpec.from_file(str(s_path))
        spec.attach(s_spec, prefix=f"{side}_", frame=spec.frame(f"{side}_sensor_mount"))

    _add_module_contact_exclusions(
        spec, left_sensor_path is not None, right_sensor_path is not None, camera_path is not None
    )
    if left_sensor_path is not None and right_sensor_path is not None:
        tip_protrusion = {
            "pts_2_0": TIP_PROTRUSION_M,
            "flat_pad": TIP_PROTRUSION_M,
            "paxini_m2824": PAXINI_M2824_TIP_PROTRUSION_M,
            "soft_dome": SOFT_DOME_TIP_PROTRUSION_M,
        }
        contact_slide = (
            FULL_OPEN_MOUNT_GAP_M
            - tip_protrusion[left_sensor_path.stem]
            - tip_protrusion[right_sensor_path.stem]
        ) / 2.0
        if contact_slide <= SINGLE_FINGER_STROKE_M:
            _add_opposing_tip_stop(spec, contact_slide)
    spec.compile()
    root = ET.fromstring(spec.to_xml())
    base = root.find(".//body[@name='base']")
    joint = base.find("joint[@name='base_freejoint'][@type='free']")
    if joint is not None:
        base.remove(joint)
        base.insert(0, ET.Element("freejoint", {"name": "base_freejoint"}))
    # MjSpec 序列化会舍入数值。恢复原始惯量和关节参数，保留构建精度。
    sources = [
        ("", chassis_path),
        ("left_", left_sensor_path),
        ("right_", right_sensor_path),
        ("cam_", camera_path),
    ]
    for prefix, path in sources:
        if path is None:
            continue
        for body in ET.parse(path).iter("body"):
            name = prefix + body.get("name")
            flat_body = root.find(f".//body[@name='{name}']")
            flat_inertial = flat_body.find("inertial")
            flat_inertial.attrib.clear()
            flat_inertial.attrib.update(body.find("inertial").attrib)
            for joint in body.findall("joint"):
                name = prefix + joint.get("name")
                flat_joint = flat_body.find(f"joint[@name='{name}']")
                flat_joint.attrib.update(joint.attrib)
                flat_joint.set("name", name)
    compiler = root.find("compiler")
    compiler.set("meshdir", "../meshes")
    compiler.set("autolimits", "true")
    for mesh_el in root.findall("./asset/mesh"):
        file_attr = mesh_el.get("file", "")
        if file_attr:
            mesh_el.set("file", Path(file_attr).name)

    _write_xml(root, out_path)


def _compute_composite_payload_in_base(mjcf_path: Path, q_drive: float) -> dict[str, object]:
    """在指定电机转角 q_drive 下计算整机相对于法兰原点 base 的合质量、质心与合惯量。"""
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    spec.delete(spec.joint("base_freejoint"))
    m = spec.compile()
    d = mujoco.MjData(m)
    target = get_joint_positions(q_drive)
    for name, position in target.items():
        d.joint(name).qpos[0] = position
    mujoco.mj_forward(m, d)

    actual = {name: float(d.joint(name).qpos[0]) for name in target}
    max_site_error = max(
        float(
            np.linalg.norm(
                d.site(f"closing_{side}_link_finger_pin_1{suffix}").xpos
                - d.site(f"closing_{side}_link_finger_pin_2{suffix}").xpos
            )
        )
        for side in ("left", "right")
        for suffix in ("", "_z")
    )
    if max_site_error > 1e-6 or d.ncon:
        raise ValueError(f"静态负载构型无效：闭环残差 {max_site_error:.3g} m，接触数 {d.ncon}")

    base_id = m.body("base").id
    R_base_w = np.asarray(d.xmat[base_id], dtype=float).reshape(3, 3)
    p_base_w = np.asarray(d.xpos[base_id], dtype=float)

    parts: list[InertialPart] = []
    for bid in range(1, m.nbody):
        mass = float(m.body_mass[bid])
        if mass <= 0.0:
            continue
        com_w = np.asarray(d.xipos[bid], dtype=float)
        R_i_w = np.asarray(d.ximat[bid], dtype=float).reshape(3, 3)
        I_body_diag = np.diag(np.asarray(m.body_inertia[bid], dtype=float))
        com_b = R_base_w.T @ (com_w - p_base_w)
        R_i_b = R_base_w.T @ R_i_w
        I_b = R_i_b @ I_body_diag @ R_i_b.T
        parts.append(InertialPart(m.body(bid).name, mass, com_b, I_b))

    total_mass, com_b, I_total = _combine_inertials(parts)
    return {
        "q_drive_rad": actual["gripper_drive"],
        "q_backlash_rad": actual["gripper_backlash"],
        "left_link_crank_pin_rad": actual["left_link_crank_pin"],
        "right_link_crank_pin_rad": actual["right_link_crank_pin"],
        "left_finger_slide_m": actual["left_finger_slide"],
        "right_finger_slide_m": actual["right_finger_slide"],
        "drive_error_rad": actual["gripper_drive"] - q_drive,
        "max_closure_error_m": max_site_error,
        "max_angular_speed_rad_s": float(
            max(abs(d.joint(name).qvel[0]) for name in target if not name.endswith("_slide"))
        ),
        "max_slide_speed_m_s": float(
            max(abs(d.joint(name).qvel[0]) for name in target if name.endswith("_slide"))
        ),
        "mass_kg": round(total_mass, 6),
        "F_x_Cee_m": [round(float(v), 6) for v in com_b],
        "F_I_ee_kg_m2": [[round(float(x), 8) for x in row] for row in I_total],
    }


def _build_urdf_xacros(
    chassis_xml: Path,
    pts_xml: Path,
    flat_xml: Path,
    paxini_xml: Path,
    soft_dome_xml: Path,
    camera_xml: Path,
    urdf_dir: Path,
    srdf_dir: Path,
) -> None:
    """以生成的 MJCF 模块为几何和惯量源，导出 URDF 与 SRDF。"""
    (urdf_dir / "sensors").mkdir(parents=True, exist_ok=True)
    (urdf_dir / "accessories").mkdir(parents=True, exist_ok=True)
    srdf_dir.mkdir(parents=True, exist_ok=True)

    def pose(elem: ET.Element) -> np.ndarray:
        return _pose_to_mat(
            _parse_vec(elem.get("pos"), [0, 0, 0]),
            _parse_vec(elem.get("quat"), [1, 0, 0, 0]),
        )

    def number(value: float) -> str:
        return f"{0.0 if abs(value) < 1e-14 else value:.12g}"

    def origin(link_pose: np.ndarray, element_pose: np.ndarray, *, pad_back: bool = False) -> str:
        relative = np.linalg.inv(link_pose) @ element_pose
        xyz = [number(value) for value in relative[:3, 3]]
        if pad_back:
            backwards = -0.5 * relative[:3, 2]
            xyz = [
                f"${{{coord} + ({number(delta)}) * safety_distance}}"
                if abs(delta) > 1e-14
                else coord
                for coord, delta in zip(xyz, backwards, strict=True)
            ]
        rpy = " ".join(map(number, _mat_to_rpy(relative[:3, :3])))
        return f'<origin xyz="{" ".join(xyz)}" rpy="{rpy}"/>'

    def inertial(body: ET.Element, body_pose: np.ndarray) -> str:
        element = body.find("inertial")
        if "fullinertia" in element.attrib:
            values = element.get("fullinertia").split()
        else:
            values = element.get("diaginertia").split() + ["0", "0", "0"]
        tensor = " ".join(
            f'{key}="{value}"'
            for key, value in zip(
                ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"),
                values,
                strict=True,
            )
        )
        return f'''      <inertial>
        {origin(body_pose, body_pose @ pose(element))}
        <mass value="{element.get("mass")}"/>
        <inertia {tensor}/>
      </inertial>
'''

    def link_geometry(
        root: ET.Element, body: ET.Element, body_pose: np.ndarray, prefix: str
    ) -> str:
        materials = {item.get("name"): item.get("rgba") for item in root.findall("asset/material")}
        meshes = {item.get("name"): item.get("file") for item in root.findall("asset/mesh")}
        xml = inertial(body, body_pose)
        for geom in body.findall("geom"):
            visual = geom.get("class") == "visual" or geom.get("contype") == "0"
            tag = "visual" if visual else "collision"
            name = geom.get("name")
            shape = geom.get("type")
            pad_back = name == "sensor_housing_collision" or (name or "").endswith(
                "_back_collision"
            )
            pad_all = not visual and not pad_back and geom.get("class") == "collision"
            geom_pose = origin(body_pose, body_pose @ pose(geom), pad_back=pad_back)
            dims = _parse_vec(geom.get("size"), [])
            if shape == "mesh":
                filename = meshes[geom.get("mesh")]
                geometry = f'<mesh filename="package://{PACKAGE_NAME}/meshes/{filename}"/>'
            elif shape == "box":
                size = [number(2 * value) for value in dims]
                if pad_all or pad_back:
                    size = [
                        f"${{{value} + {1 if pad_back and idx == 2 else 2} * safety_distance}}"
                        for idx, value in enumerate(size)
                    ]
                geometry = f'<box size="{" ".join(size)}"/>'
            elif shape == "cylinder":
                geometry = (
                    f'<cylinder radius="${{{number(dims[0])} + safety_distance}}" '
                    f'length="${{{number(2 * dims[1])} + {1 if pad_back else 2} * safety_distance}}"/>'
                )
            elif shape == "sphere":
                geometry = f'<sphere radius="{number(dims[0])}"/>'
            else:
                raise ValueError(f"不支持的 URDF 几何体：{shape}")
            name_attr = f' name="${{{prefix}}}_{name}"' if name else ""
            xml += f"      <{tag}{name_attr}>\n        {geom_pose}\n        <geometry>{geometry}</geometry>\n"
            if visual:
                material = geom.attrib["material"]
                rgba = geom.get("rgba", materials[material])
                if "rgba" in geom.attrib:
                    material += "__" + geom.get("mesh", name)
                xml += f'        <material name="${{{prefix}}}_{material}"><color rgba="{rgba}"/></material>\n'
            xml += f"      </{tag}>\n"
        return xml

    def joint(
        name: str,
        parent: str,
        child: str,
        parent_pose: np.ndarray,
        child_pose: np.ndarray,
        kind: str = "fixed",
        extra: str = "",
    ) -> str:
        return f'''    <joint name="{name}" type="{kind}">
      <parent link="{parent}"/>
      <child link="{child}"/>
      {origin(parent_pose, child_pose)}
{extra}    </joint>
'''

    def frame(
        name: str,
        parent: str,
        child: str,
        parent_pose: np.ndarray,
        child_pose: np.ndarray,
    ) -> str:
        return joint(name, parent, child, parent_pose, child_pose) + f'    <link name="{child}"/>\n'

    identity = np.eye(4)
    xml_start = """<?xml version="1.0" encoding="utf-8"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
"""
    xml_end = "  </xacro:macro>\n</robot>\n"

    for name, path in (
        ("pts_2_0", pts_xml),
        ("flat_pad", flat_xml),
        ("paxini_m2824", paxini_xml),
        ("soft_dome", soft_dome_xml),
    ):
        root = ET.parse(path).getroot()
        body = root.find("worldbody/body")
        body_pose = pose(body)
        sensor = "${prefix}_sensor_link"
        fingertip = "${prefix}_fingertip"
        xml = (
            xml_start
            + f'  <xacro:macro name="{name}_sensor" params="prefix parent safety_distance:=0.0">\n'
        )
        xml += joint("${prefix}_sensor_joint", "${parent}", sensor, identity, body_pose)
        xml += f'    <link name="{sensor}">\n{link_geometry(root, body, body_pose, "prefix")}    </link>\n'
        xml += frame(
            "${prefix}_fingertip_joint",
            sensor,
            fingertip,
            body_pose,
            body_pose @ pose(body.find("site[@name='fingertip']")),
        )
        (urdf_dir / "sensors" / f"{name}.xacro").write_text(xml + xml_end, encoding="utf-8")

    camera_root = ET.parse(camera_xml).getroot()
    camera = camera_root.find("worldbody/body")
    camera_pose = pose(camera)
    mount = "${ns}_camera_mount_link"
    xml = (
        xml_start
        + '  <xacro:macro name="realsense_d435i_mount" params="ns parent safety_distance:=0.0 use_nominal_extrinsics:=true">\n'
    )
    xml += joint("${ns}_camera_mount_joint", "${parent}", mount, identity, camera_pose)
    xml += f'    <link name="{mount}">\n{link_geometry(camera_root, camera, camera_pose, "ns")}    </link>\n'
    camera_sites = {site.get("name"): camera_pose @ pose(site) for site in camera.findall("site")}
    screw = "d435i_bottom_screw_frame"
    camera_link = "d435i_link"
    xml += frame(
        "${ns}_d435i_bottom_screw_joint",
        mount,
        f"${{ns}}_{screw}",
        camera_pose,
        camera_sites[screw],
    )
    xml += frame(
        "${ns}_d435i_link_joint",
        f"${{ns}}_{screw}",
        f"${{ns}}_{camera_link}",
        camera_sites[screw],
        camera_sites[camera_link],
    )
    xml += '    <xacro:if value="${use_nominal_extrinsics}">\n'
    nominal_camera_links = []
    for name, child_pose in camera_sites.items():
        if name in (screw, camera_link):
            continue
        parent = (
            name.replace("_optical_frame", "_frame")
            if name.endswith("_optical_frame")
            else camera_link
        )
        name_joint = name.removesuffix("_frame") + "_joint"
        subtree = frame(
            f"${{ns}}_{name_joint}",
            f"${{ns}}_{parent}",
            f"${{ns}}_{name}",
            camera_sites[parent],
            child_pose,
        )
        xml += "".join("  " + line for line in subtree.splitlines(keepends=True))
        nominal_camera_links.append(name)
    xml += "    </xacro:if>\n"
    (urdf_dir / "accessories" / "realsense_d435i.xacro").write_text(xml + xml_end, encoding="utf-8")

    chassis = ET.parse(chassis_xml).getroot()
    base = chassis.find("worldbody/body")
    base_pose = pose(base)
    xml = (
        xml_start
        + """  <!-- URDF 使用开链描述。连杆位姿由闭环关节状态同步。 -->
  <xacro:macro name="adaptive_tactile_gripper_chassis"
               params="ns:='panda_gripper' connected_to:='' xyz:='0 0 0' rpy:='0 0 0'
                       safety_distance:=0.0">
    <xacro:unless value="${connected_to == ''}">
      <joint name="${ns}_mount_joint" type="fixed">
        <parent link="${connected_to}"/>
        <child link="${ns}_base"/>
        <origin xyz="${xyz}" rpy="${rpy}"/>
      </joint>
    </xacro:unless>
"""
    )
    chassis_links = ["base"]
    chassis_joints = []

    def export_body(body: ET.Element, parent: str, parent_pose: np.ndarray) -> str:
        name = body.get("name")
        if name.startswith("rod_bearing_shim_subassembly"):
            name = "right_rod" if name.endswith("_2") else "left_rod"
        body_pose = parent_pose @ pose(body)
        link = f"${{ns}}_{name}"
        result = ""
        element = body.find("joint")
        if element is not None:
            chassis_links.append(name)
            jname = element.get("name")
            chassis_joints.append(jname)
            kind = {"slide": "prismatic", "hinge": "revolute"}[element.get("type")]
            axis = element.get("axis")
            if kind == "prismatic":
                lower, upper = element.get("range").split()
                effort, velocity = "100", "0.5"
            else:
                lower, upper = "0", number(POSITION_MAX_RAD)
                effort, velocity = "4", "40"
            extra = (
                f'      <axis xyz="{axis}"/>\n'
                f'      <limit lower="{lower}" upper="{upper}" effort="{effort}" velocity="{velocity}"/>\n'
            )
            if name == "right_finger":
                extra += (
                    '      <mimic joint="${ns}_left_finger_slide" multiplier="1" offset="0"/>\n'
                )
            result += joint(f"${{ns}}_{jname}", parent, link, parent_pose, body_pose, kind, extra)
        result += f'    <link name="{link}">\n{link_geometry(chassis, body, body_pose, "ns")}    </link>\n'
        for child in body.findall("body"):
            result += export_body(child, link, body_pose)
        for element in body.findall("frame"):
            frame_name = element.get("name")
            joint_name = (
                "camera_frame_joint"
                if frame_name == "camera_mount_frame"
                else f"{frame_name}_joint"
            )
            chassis_links.append(frame_name)
            result += frame(
                f"${{ns}}_{joint_name}",
                link,
                f"${{ns}}_{frame_name}",
                body_pose,
                body_pose @ pose(element),
            )
        return result

    xml += export_body(base, "", identity)
    (urdf_dir / "gripper_chassis.xacro").write_text(xml + xml_end, encoding="utf-8")
    tcp_frame = frame(
        "${ns}_tcp_joint",
        "${ns}_base",
        "${ns}_tcp",
        base_pose,
        base_pose @ pose(base.find("site[@name='tcp']")),
    )

    # 5. gripper.xacro
    top_xacro = f"""<?xml version="1.0" encoding="utf-8"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="$(find {PACKAGE_NAME})/urdf/gripper_chassis.xacro"/>
  <xacro:include filename="$(find {PACKAGE_NAME})/urdf/sensors/pts_2_0.xacro"/>
  <xacro:include filename="$(find {PACKAGE_NAME})/urdf/sensors/flat_pad.xacro"/>
  <xacro:include filename="$(find {PACKAGE_NAME})/urdf/sensors/paxini_m2824.xacro"/>
  <xacro:include filename="$(find {PACKAGE_NAME})/urdf/sensors/soft_dome.xacro"/>
  <xacro:include filename="$(find {PACKAGE_NAME})/urdf/accessories/realsense_d435i.xacro"/>

  <!-- 顶层可配置夹爪宏：支持左右传感器独立替换与相机可选挂载 -->
  <xacro:macro name="adaptive_tactile_gripper"
               params="ns:='panda_gripper' connected_to:='' xyz:='0 0 0' rpy:='0 0 0'
                       left_sensor:='pts_2_0' right_sensor:='pts_2_0'
                       with_camera:='true' safety_distance:=0.0 use_nominal_extrinsics:=true">
    <xacro:adaptive_tactile_gripper_chassis
        ns="${{ns}}" connected_to="${{connected_to}}"
        xyz="${{xyz}}" rpy="${{rpy}}" safety_distance="${{safety_distance}}"/>

    <xacro:if value="${{left_sensor == 'pts_2_0'}}">
      <xacro:pts_2_0_sensor prefix="${{ns}}_left" parent="${{ns}}_left_sensor_mount" safety_distance="${{safety_distance}}"/>
    </xacro:if>
    <xacro:if value="${{left_sensor == 'flat_pad'}}">
      <xacro:flat_pad_sensor prefix="${{ns}}_left" parent="${{ns}}_left_sensor_mount" safety_distance="${{safety_distance}}"/>
    </xacro:if>
    <xacro:if value="${{left_sensor == 'paxini_m2824'}}">
      <xacro:paxini_m2824_sensor prefix="${{ns}}_left" parent="${{ns}}_left_sensor_mount" safety_distance="${{safety_distance}}"/>
    </xacro:if>
    <xacro:if value="${{left_sensor == 'soft_dome'}}">
      <xacro:soft_dome_sensor prefix="${{ns}}_left" parent="${{ns}}_left_sensor_mount" safety_distance="${{safety_distance}}"/>
    </xacro:if>

    <xacro:if value="${{right_sensor == 'pts_2_0'}}">
      <xacro:pts_2_0_sensor prefix="${{ns}}_right" parent="${{ns}}_right_sensor_mount" safety_distance="${{safety_distance}}"/>
    </xacro:if>
    <xacro:if value="${{right_sensor == 'flat_pad'}}">
      <xacro:flat_pad_sensor prefix="${{ns}}_right" parent="${{ns}}_right_sensor_mount" safety_distance="${{safety_distance}}"/>
    </xacro:if>
    <xacro:if value="${{right_sensor == 'paxini_m2824'}}">
      <xacro:paxini_m2824_sensor prefix="${{ns}}_right" parent="${{ns}}_right_sensor_mount" safety_distance="${{safety_distance}}"/>
    </xacro:if>
    <xacro:if value="${{right_sensor == 'soft_dome'}}">
      <xacro:soft_dome_sensor prefix="${{ns}}_right" parent="${{ns}}_right_sensor_mount" safety_distance="${{safety_distance}}"/>
    </xacro:if>

    <xacro:if value="${{with_camera}}">
      <xacro:realsense_d435i_mount ns="${{ns}}" parent="${{ns}}_camera_mount_frame"
          safety_distance="${{safety_distance}}" use_nominal_extrinsics="${{use_nominal_extrinsics}}"/>
    </xacro:if>

{tcp_frame.rstrip()}
  </xacro:macro>
</robot>
"""
    (urdf_dir / "gripper.xacro").write_text(top_xacro, encoding="utf-8")

    # 6. gripper_standalone.urdf.xacro
    standalone_xacro = f"""<?xml version="1.0" encoding="utf-8"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="adaptive_tactile_gripper">
  <xacro:arg name="ns" default="panda_gripper"/>
  <xacro:arg name="left_sensor" default="pts_2_0"/>
  <xacro:arg name="right_sensor" default="pts_2_0"/>
  <xacro:arg name="with_camera" default="true"/>
  <xacro:arg name="safety_distance" default="0.0"/>
  <xacro:arg name="use_nominal_extrinsics" default="true"/>

  <xacro:include filename="$(find {PACKAGE_NAME})/urdf/gripper.xacro"/>
  <xacro:adaptive_tactile_gripper
      ns="$(arg ns)"
      left_sensor="$(arg left_sensor)"
      right_sensor="$(arg right_sensor)"
      with_camera="$(arg with_camera)"
      safety_distance="$(arg safety_distance)"
      use_nominal_extrinsics="$(arg use_nominal_extrinsics)"/>
</robot>
"""
    (urdf_dir / "gripper_standalone.urdf.xacro").write_text(standalone_xacro, encoding="utf-8")

    # SRDF 只引用当前 URDF 配置实际存在的夹爪成员。
    def group_members(names: list[str], tag: str, indent: int = 6) -> str:
        return "".join(f'{" " * indent}<{tag} name="${{ns}}_{name}"/>\n' for name in names)

    chassis_links.append("tcp")
    fixed_joints = [
        f"{name}_joint" for name in chassis_links if name.endswith("_mount") or name == "tcp"
    ]
    fixed_joints.append("camera_frame_joint")
    nominal_camera_joints = [
        name.removesuffix("_frame") + "_joint" for name in nominal_camera_links
    ]
    srdf_xacro = """<?xml version="1.0" encoding="utf-8"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="adaptive_tactile_gripper_srdf">
  <xacro:macro name="adaptive_tactile_gripper_srdf"
               params="arm_id:='panda' ns:='panda_gripper' with_camera:='true'
                       left_sensor:='pts_2_0' right_sensor:='pts_2_0' use_nominal_extrinsics:=true">
    <group name="${ns}">
"""
    srdf_xacro += group_members(chassis_joints + fixed_joints, "joint") + group_members(
        chassis_links, "link"
    )
    for side in ("left", "right"):
        srdf_xacro += f"""      <xacro:if value="${{{side}_sensor in ('pts_2_0', 'flat_pad', 'paxini_m2824', 'soft_dome')}}">
"""
        srdf_xacro += group_members([f"{side}_sensor_joint", f"{side}_fingertip_joint"], "joint", 8)
        srdf_xacro += group_members([f"{side}_sensor_link", f"{side}_fingertip"], "link", 8)
        srdf_xacro += "      </xacro:if>\n"
    srdf_xacro += '      <xacro:if value="${with_camera}">\n'
    srdf_xacro += group_members(
        ["camera_mount_joint", "d435i_bottom_screw_joint", "d435i_link_joint"], "joint", 8
    )
    srdf_xacro += group_members(["camera_mount_link", screw, camera_link], "link", 8)
    srdf_xacro += '        <xacro:if value="${use_nominal_extrinsics}">\n'
    srdf_xacro += group_members(nominal_camera_joints, "joint", 10)
    srdf_xacro += group_members(nominal_camera_links, "link", 10)
    srdf_xacro += """        </xacro:if>
      </xacro:if>
    </group>
    <!-- 规划组的唯一独立变量为左滑块。完整末端组用于同步后的状态与接触链。 -->
    <group name="${ns}_motion">
      <joint name="${ns}_left_finger_slide"/>
    </group>
"""
    for name in (
        "right_finger_slide",
        "gripper_drive",
        "left_link_crank_pin",
        "right_link_crank_pin",
    ):
        srdf_xacro += f'    <passive_joint name="${{ns}}_{name}"/>\n'

    for state, closure in (("open", 0.0), ("close", SRDF_CLOSE_M)):
        positions = get_joint_positions(drive_angle_for_closure(closure))
        srdf_xacro += f'\n    <group_state name="{state}" group="${{ns}}">\n'
        for name in (
            "left_finger_slide",
            "gripper_drive",
            "left_link_crank_pin",
            "right_link_crank_pin",
        ):
            srdf_xacro += (
                f'      <joint name="${{ns}}_{name}" value="{number(positions[name])}"/>\n'
            )
        srdf_xacro += "    </group_state>\n"
        srdf_xacro += f'''    <group_state name="{state}" group="${{ns}}_motion">
      <joint name="${{ns}}_left_finger_slide" value="{number(closure)}"/>
    </group_state>
'''

    srdf_xacro += """
    <end_effector name="${ns}_eef" parent_link="${arm_id}_link8" group="${ns}" parent_group="${arm_id}_arm"/>

    <disable_collisions link1="${arm_id}_link7" link2="${ns}_base" reason="Adjacent"/>
    <disable_collisions link1="${arm_id}_link8" link2="${ns}_base" reason="Adjacent"/>
    <disable_collisions link1="${ns}_base" link2="${ns}_left_finger" reason="Adjacent"/>
    <disable_collisions link1="${ns}_base" link2="${ns}_right_finger" reason="Adjacent"/>

    <xacro:if value="${with_camera}">
      <disable_collisions link1="${arm_id}_link7" link2="${ns}_camera_mount_link" reason="Adjacent"/>
      <disable_collisions link1="${arm_id}_link8" link2="${ns}_camera_mount_link" reason="Adjacent"/>
      <disable_collisions link1="${ns}_base" link2="${ns}_camera_mount_link" reason="Adjacent"/>
      <disable_collisions link1="${ns}_camera_mount_link" link2="${ns}_left_finger" reason="Never"/>
      <disable_collisions link1="${ns}_camera_mount_link" link2="${ns}_right_finger" reason="Never"/>
    </xacro:if>

    <xacro:if value="${left_sensor in ('pts_2_0', 'flat_pad', 'paxini_m2824', 'soft_dome')}">
      <disable_collisions link1="${ns}_left_finger" link2="${ns}_left_sensor_link" reason="Adjacent"/>
      <disable_collisions link1="${ns}_base" link2="${ns}_left_sensor_link" reason="Adjacent"/>
    </xacro:if>
    <xacro:if value="${right_sensor in ('pts_2_0', 'flat_pad', 'paxini_m2824', 'soft_dome')}">
      <disable_collisions link1="${ns}_right_finger" link2="${ns}_right_sensor_link" reason="Adjacent"/>
      <disable_collisions link1="${ns}_base" link2="${ns}_right_sensor_link" reason="Adjacent"/>
    </xacro:if>
  </xacro:macro>
</robot>
"""
    (srdf_dir / "gripper.srdf.xacro").write_text(srdf_xacro, encoding="utf-8")


def _read_export_microversion(export_dir: Path) -> str:
    """从导出目录的 .part 元数据中提取主文档 ee6a871f9e00fc227503180b 的 Onshape 微版本。"""
    assets_dir = export_dir / "assets"
    versions: set[str] = set()
    for part_file in sorted(assets_dir.glob("*.part")):
        try:
            payload = json.loads(part_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("documentId") != "ee6a871f9e00fc227503180b":
            continue
        mv = payload.get("documentMicroversion")
        if isinstance(mv, str) and mv.strip():
            versions.add(mv.strip())
    if len(versions) == 1:
        return f'"{next(iter(versions))}"'
    return "null"


def _write_specs(
    taxels_spec: list[dict[str, object]],
    paxini_mass_kg: float,
    paxini_taxels: list[dict[str, object]],
    soft_dome_mass_kg: float,
    phi_print: float,
    rho_pla: float,
    rho_placf: float,
    chassis_masses: dict[str, float],
    cam_info: dict[str, object],
    payload_no_cam_open: dict[str, object],
    payload_no_cam_mid: dict[str, object],
    payload_with_cam_open: dict[str, object],
    payload_with_cam_mid: dict[str, object],
    config_dir: Path,
    meshes_dir: Path,
    raw_mjcf_hash: str,
    cad_microversion: str,
) -> None:
    """写入单一真值 config/specs.yaml（含完整质量账本、DM4310/D435i 参数与 Franka Desk 负载）及传感器规格。"""
    (config_dir / "sensors").mkdir(parents=True, exist_ok=True)
    total_no_cam = chassis_masses["chassis_total_kg"] + 2.0 * PTS_2_0_MASS_KG
    total_with_cam = total_no_cam + float(cam_info["mass_kg"])

    mesh_hashes = "\n".join(
        f'    {path.name}: "{hashlib.sha256(path.read_bytes()).hexdigest()}"'
        for path in sorted(meshes_dir.glob("*.stl"))
    )
    first_contact_slide = FULL_OPEN_MOUNT_GAP_M / 2 - TIP_PROTRUSION_M
    first_contact_q = drive_angle_for_closure(first_contact_slide)

    def flow(value: object) -> str:
        return yaml.safe_dump(value, default_flow_style=True, sort_keys=False, width=1000).strip()

    specs_yaml = f"""# 自适应触觉夹爪（Onshape B1）统一规格
source:
  platform: onshape
  document_id: ee6a871f9e00fc227503180b
  workspace_id: c90cab421a5e9546e31f5c11
  element_id: 16b75a7cc71174b7cd5960db
  branch: B1
  microversion_id: {cad_microversion}
  onshape_to_robot_version: "1.8.4"
  raw_mjcf_sha256: "{raw_mjcf_hash}"
  generator_mujoco_version: "{mujoco.__version__}"
  mesh_sha256:
{mesh_hashes}

kinematics:
  type: crank_slider
  theta0_rad: {THETA0_RAD:.16f}
  crank_radius_m: {CRANK_RADIUS_M}
  link_length_m: {LINK_LENGTH_M:.16f}
  offset_m: {OFFSET_M:.16f}
  # 协议范围表示控制接口可编码的角度，机构行程由实际曲柄角限定。
  protocol_range_rad: {list(PROTOCOL_RANGE_RAD)}
  mechanism_range_rad: [{POSITION_MIN_RAD}, {POSITION_MAX_RAD:.16f}]
  position_min_rad: {POSITION_MIN_RAD}
  position_max_rad: {POSITION_MAX_RAD:.16f}
  single_finger_stroke_m: {SINGLE_FINGER_STROKE_M:.16f}
  total_stroke_m: {TOTAL_STROKE_M:.16f}

  module_contact:
    # 两侧各装一个 PTS 2.0 或平垫时，名义首次接触由 6.6 mm 单侧突出量决定；其余组合由 get_stroke_limits 计算。
    applicable_modules: [pts_2_0, flat_pad]
    first_contact_q_rad: {first_contact_q:.12f}
    no_contact_range_rad: [0.0, {first_contact_q:.12f}]
    first_contact_single_finger_closure_m: {first_contact_slide:.8g}

geometry:
  # 从 franka_adapter 法兰底面到左右传感器中心抓取点的偏置
  tool_center_in_flange_m: {TCP_IN_FLANGE_M}
  left_sensor_mount_in_finger_pos_m: {LEFT_SENSOR_MOUNT_IN_FINGER_POS}
  left_sensor_mount_in_finger_quat_wxyz: {LEFT_SENSOR_MOUNT_IN_FINGER_QUAT}
  right_sensor_mount_in_finger_pos_m: {RIGHT_SENSOR_MOUNT_IN_FINGER_POS}
  right_sensor_mount_in_finger_quat_wxyz: {RIGHT_SENSOR_MOUNT_IN_FINGER_QUAT}
  nominal_full_open_sensor_mount_gap_m: {FULL_OPEN_MOUNT_GAP_M}
  nominal_full_open_housing_gap_m: 0.0739
  nominal_full_open_center_pillar_gap_m: 0.0663

actuator:
  model: DM-J4310P-2EC
  manufacturer: DAMIAO
  gear_ratio: 10.0
  total_mass_kg: {DM4310P_STATOR_MASS_KG + DM4310P_ROTOR_OUTPUT_MASS_KG:.4f}
  stator_mass_kg: {DM4310P_STATOR_MASS_KG:.4f}
  rotor_output_mass_kg: {DM4310P_ROTOR_OUTPUT_MASS_KG:.4f}
  rated_torque_nm: 3.5
  peak_torque_nm: 12.5
  # 定转子拆分为估计。反射换算沿用转子侧参数假设。
  armature_kg_m2: 0.002074755
  damping_nms_rad: 0.02532329
  frictionloss_nm: 0.04
  backlash_rad: 0.005

camera:
  model: Intel RealSense D435i
  bare_camera_mass_kg: {D435I_BARE_MASS_KG:.4f}
  mount_and_fasteners_mass_kg: {float(cam_info["mass_kg"]) - D435I_BARE_MASS_KG:.6f}
  total_module_mass_kg: {float(cam_info["mass_kg"]):.6f}
  inertia_source: https://github.com/google-deepmind/mujoco_menagerie/blob/eea4c7431b63da8278b0ff31638e0b0e941cec0e/realsense_d435i/d435i.xml
  # realsense2_description 来自 realsense-ros v4.58.4。
  extrinsics_source: https://github.com/realsenseai/realsense-ros/blob/c8fac23cec47bb18544810f96bc1668edaec0d08/realsense2_description/urdf/_d435i.urdf.xacro
  imu_extrinsics_source: https://github.com/realsenseai/realsense-ros/blob/c8fac23cec47bb18544810f96bc1668edaec0d08/realsense2_description/urdf/_d435i_imu_modules.urdf.xacro
  bare_com_in_stl_m: {D435I_COM_IN_STL_M.tolist()}
  bare_principal_inertia_kg_m2: {flow(D435I_DIAGINERTIA_LOCAL.tolist())}
  bare_principal_axes_in_stl_wxyz: {D435I_IQUAT_IN_STL_WXYZ}
  color_fovy_deg: 42.5
  depth_fovy_deg: 58.0
  depth_to_color_offset_y_m: 0.015
  depth_to_infra2_offset_y_m: -0.050

materials_and_mass_budget:
  calibration_note: >
    电子秤实测 0.686000 kg 总质量用于约束无相机、双侧 PTS 2.0 装配。
    在打印件均匀填充假设下估计各刚体的质量分布和惯量，统一有效打印实心率为 {phi_print:.5f}。
    手指主体为 PLA，其余打印结构件为 PLA-CF，标准件使用标称规格或质量分布估计。
  effective_print_fill_ratio: {round(phi_print, 6)}
  effective_pla_density_kg_m3: {round(rho_pla, 2)}
  effective_placf_density_kg_m3: {round(rho_placf, 2)}
  breakdown_kg:
    base_body: {round(chassis_masses["base_kg"], 6)}
    motor_and_crank_body: {round(chassis_masses["motor_kg"], 6)}
    left_rod_assembly: {round(chassis_masses["left_rod_kg"], 6)}
    right_rod_assembly: {round(chassis_masses["right_rod_kg"], 6)}
    left_finger_chassis: {round(chassis_masses["left_finger_kg"], 6)}
    right_finger_chassis: {round(chassis_masses["right_finger_kg"], 6)}
    left_pts_2_0: {PTS_2_0_MASS_KG:.6f}
    right_pts_2_0: {PTS_2_0_MASS_KG:.6f}
    total_without_camera: {round(total_no_cam, 6)}
    realsense_d435i_module: {round(float(cam_info["mass_kg"]), 6)}
    total_with_camera: {round(total_with_cam, 6)}
    optional_flange_m6x12_bolts_x4: 0.02049

franka_desk_payload:
  # 固定基座、解析闭环构型（回差角为零）下的质量属性估计。
  # 惯量绕合质心，坐标轴平行于法兰参考系。四颗法兰 M6x12 螺钉需另计。
  method: analytic_closed_loop_mj_forward
  without_camera_open: {flow(payload_no_cam_open)}
  without_camera_nominal_grasp: {flow(payload_no_cam_mid)}
  with_camera_open: {flow(payload_with_cam_open)}
  with_camera_nominal_grasp: {flow(payload_with_cam_mid)}

default_configuration:
  left_sensor: pts_2_0
  right_sensor: pts_2_0
  with_camera: true
"""
    (config_dir / "specs.yaml").write_text(specs_yaml, encoding="utf-8")

    pts_lines = [
        "# Contactile PapillArray PTS 2.0 3x3 传感器模组规格（碰撞优先使用 9 颗 r=2.8mm 球体）",
        "name: pts_2_0",
        "kind: tactile_3axis_array",
        f"mass_kg: {PTS_2_0_MASS_KG}",
        "# 中心柱顶沿安装系 +Z。双侧装配的名义接触界限见 specs.yaml。",
        f"thickness_to_tip_m: {TIP_PROTRUSION_M}",
        "tactile:",
        "  mode: contact_geom",
        "  rows: 3",
        "  cols: 3",
        "  left_prefix: left_taxel_geom_",
        "  right_prefix: right_taxel_geom_",
        "  left_site_prefix: left_taxel_",
        "  right_site_prefix: right_taxel_",
        "  site_quat_in_sensor_mount_wxyz: [0, 1, 0, 0]",
        "contact:",
        "  # 突出量从安装原点沿 +Z 测到最前接触面。",
        "  friction_coefficient: 0.8",
        "  solref: [-1200.0, -10.0]",
        "  solimp: [0.75, 0.95, 0.0025, 0.5, 2.0]",
        "taxels_in_sensor_mount:",
    ]
    for item in taxels_spec:
        pts_lines.append(
            f"  - {flow({'row': item['row'], 'col': item['col'], 'pos_m': item['pos_m'], 'radius_m': item['radius_m']})}"
        )
    (config_dir / "sensors" / "pts_2_0.yaml").write_text(
        "\n".join(pts_lines) + "\n", encoding="utf-8"
    )

    flat_yaml = f"""# 无触觉平垫占位模组规格（用于消融对比或纯视觉抓取）
name: flat_pad
kind: rigid_pad
mass_kg: {FLAT_PAD_MASS_KG}
# 前表面沿安装系 +Z；双侧装配的接触界限由 specs.yaml 定义。
thickness_to_tip_m: {TIP_PROTRUSION_M}
tactile: null
contact:
  friction_coefficient: 0.8
  solref: [-1200.0, -10.0]
  solimp: [0.75, 0.95, 0.0025, 0.5, 2.0]
"""
    (config_dir / "sensors" / "flat_pad.yaml").write_text(flat_yaml, encoding="utf-8")

    paxini_mesh_hashes = "\n".join(
        f'  {path.name}: "{hashlib.sha256(path.read_bytes()).hexdigest()}"'
        for path in sorted((meshes_dir / "paxini_m2824").glob("*.stl"))
    )
    paxini_lines = [
        "# PaXini PX6AX-GEN4-GR-M2824-Gripper 135 测点夹爪触觉模组规格（含 3.5 mm 转接垫块，盒式平面碰撞）",
        "name: paxini_m2824",
        "model: PX6AX-GEN4-GR-M2824-Gripper",
        "kind: tactile_3axis_pad",
        f"mass_kg: {round(paxini_mass_kg, 6)}",
        "# 前接触面沿安装系 +Z（含 3.5 mm 转接垫块对齐至 3.7 mm）；双侧首次接触角由 get_stroke_limits 计算。",
        f"thickness_to_tip_m: {PAXINI_M2824_TIP_PROTRUSION_M}",
        "mesh_sha256:",
        paxini_mesh_hashes,
        "tactile:",
        "  mode: pad_sites",
        f"  count: {len(paxini_taxels)}",
        "  left_pad_geom: left_pad_collision",
        "  right_pad_geom: right_pad_collision",
        "  left_site_prefix: left_taxel_",
        "  right_site_prefix: right_taxel_",
        "  site_quat_in_sensor_mount_wxyz: [0, 1, 0, 0]",
        "contact:",
        "  friction_coefficient: 0.8",
        "  solref: [-1200.0, -10.0]",
        "  solimp: [0.75, 0.95, 0.0025, 0.5, 2.0]",
        "taxels_in_sensor_mount:",
    ]
    for item in paxini_taxels:
        paxini_lines.append(f"  - {flow({'id': item['id'], 'pos_m': item['pos_m']})}")
    (config_dir / "sensors" / "paxini_m2824.yaml").write_text(
        "\n".join(paxini_lines) + "\n", encoding="utf-8"
    )

    soft_dome_mesh_hashes = "\n".join(
        f'  {path.name}: "{hashlib.sha256(path.read_bytes()).hexdigest()}"'
        for path in sorted((meshes_dir / "soft_dome").glob("*.stl"))
    )
    cap_height_m = round(SOFT_DOME_TIP_PROTRUSION_M - SOFT_DOME_BASE_FRONT_Z_M, 6)
    cap_base_radius_m = round(
        math.sqrt(
            SOFT_DOME_RADIUS_M**2 - (SOFT_DOME_BASE_FRONT_Z_M - SOFT_DOME_SPHERE_CENTER_Z_M) ** 2
        ),
        9,
    )
    base_thickness_m = round(SOFT_DOME_BASE_FRONT_Z_M - SOFT_DOME_BACK_Z_M, 6)
    soft_dome_yaml = f"""# 硅胶扁球冠六轴力/力矩指尖模组规格（2.0 mm 薄圆柱底座 + 7.5 mm 球冠，曲率半径 R=12 mm）
name: soft_dome
kind: tactile_6axis_dome
mass_kg: {round(soft_dome_mass_kg, 6)}
# 球冠顶点沿安装系 +Z 位于 3.7 mm（距指腹安装面总高 9.5 mm）；半球对应冠高 h=R 特例。
thickness_to_tip_m: {SOFT_DOME_TIP_PROTRUSION_M}
opposing_tip_stop: true
dome_geometry:
  curvature_radius_m: {SOFT_DOME_RADIUS_M}
  cap_height_m: {cap_height_m}
  cap_base_radius_m: {cap_base_radius_m}
  base_radius_m: {SOFT_DOME_BASE_RADIUS_M}
  base_thickness_m: {base_thickness_m}
  base_front_z_m: {SOFT_DOME_BASE_FRONT_Z_M}
  sphere_center_in_sensor_mount_m: [0.0, 0.0, {SOFT_DOME_SPHERE_CENTER_Z_M}]
mesh_sha256:
{soft_dome_mesh_hashes}
tactile:
  mode: pad_wrench
  left_pad_geom: left_pad_collision
  right_pad_geom: right_pad_collision
  left_site: left_pad_center
  right_site: right_pad_center
  left_base_site: left_ft_sensor
  right_base_site: right_ft_sensor
  site_quat_in_sensor_mount_wxyz: [0, 1, 0, 0]
contact:
  condim: 4
  friction_coefficient: 0.8
  torsional_friction: 0.0064
  rolling_friction: 0.0001
  solref: [-4800.0, -24.0]
  solimp: [0.75, 0.97, 0.002, 0.5, 2.0]
"""
    (config_dir / "sensors" / "soft_dome.yaml").write_text(soft_dome_yaml, encoding="utf-8")


def build_all(
    export_dir: Path,
    out_root: Path,
    appearance: str = "realistic",
    paxini_export_dir: Path = DEFAULT_PAXINI_EXPORT_DIR,
) -> None:
    """从 Onshape 导出结果构建完整模块化夹爪描述资产。"""
    raw_xml = export_dir / "raw_gripper.xml"
    raw_assets = export_dir / "assets"
    if not raw_xml.is_file():
        raise FileNotFoundError(f"未找到导出文件: {raw_xml}")

    paxini_taxels = _load_paxini_taxels(paxini_export_dir / "taxels.json")
    _validate_paxini_meshes(paxini_export_dir)
    raw_data = raw_xml.read_bytes()
    raw_root = ET.fromstring(raw_data)
    meshes_dir = out_root / "meshes"
    palette = Appearance.from_library(_build_appearance_library(raw_root, out_root), appearance)
    _copy_and_rename_meshes(raw_root, raw_assets, meshes_dir)
    _copy_paxini_meshes(paxini_export_dir, meshes_dir)
    _generate_soft_dome_meshes(meshes_dir)

    phi_print, rho_pla, rho_placf = _solve_print_fill_ratio(raw_root, meshes_dir)

    mjcf_dir = out_root / "mjcf"
    chassis_path = mjcf_dir / "gripper_chassis.xml"
    pts_path = mjcf_dir / "sensors" / "pts_2_0.xml"
    flat_path = mjcf_dir / "sensors" / "flat_pad.xml"
    paxini_path = mjcf_dir / "sensors" / "paxini_m2824.xml"
    soft_dome_path = mjcf_dir / "sensors" / "soft_dome.xml"
    cam_path = mjcf_dir / "accessories" / "realsense_d435i.xml"

    chassis_masses = _build_chassis_mjcf(
        raw_root, palette, meshes_dir, rho_pla, rho_placf, chassis_path
    )
    taxels_spec = _build_pts_2_0_mjcf(raw_root, palette, meshes_dir, pts_path)
    _build_flat_pad_mjcf(palette, flat_path)
    paxini_mass_kg = _build_paxini_m2824_mjcf(
        palette, meshes_dir, rho_placf, paxini_taxels, paxini_path
    )
    soft_dome_mass_kg = _build_soft_dome_mjcf(palette, meshes_dir, rho_placf, soft_dome_path)
    cam_info = _build_realsense_d435i_mjcf(raw_root, palette, meshes_dir, rho_placf, cam_path)

    full_xml = mjcf_dir / "gripper.xml"
    nocam_xml = mjcf_dir / "gripper_no_camera.xml"
    _assemble_single_mjcf(chassis_path, pts_path, pts_path, cam_path, full_xml)
    _assemble_single_mjcf(chassis_path, pts_path, pts_path, None, nocam_xml)

    payload_no_cam_open = _compute_composite_payload_in_base(nocam_xml, 0.0)
    payload_no_cam_mid = _compute_composite_payload_in_base(nocam_xml, 0.75)
    payload_with_cam_open = _compute_composite_payload_in_base(full_xml, 0.0)
    payload_with_cam_mid = _compute_composite_payload_in_base(full_xml, 0.75)

    _build_urdf_xacros(
        chassis_path,
        pts_path,
        flat_path,
        paxini_path,
        soft_dome_path,
        cam_path,
        out_root / "urdf",
        out_root / "srdf",
    )
    _write_specs(
        taxels_spec,
        paxini_mass_kg,
        paxini_taxels,
        soft_dome_mass_kg,
        phi_print,
        rho_pla,
        rho_placf,
        chassis_masses,
        cam_info,
        payload_no_cam_open,
        payload_no_cam_mid,
        payload_with_cam_open,
        payload_with_cam_mid,
        out_root / "config",
        meshes_dir,
        hashlib.sha256(raw_data).hexdigest(),
        _read_export_microversion(export_dir),
    )

    _validate_module_appearances(mjcf_dir, palette)


def _validate_module_appearances(mjcf_dir: Path, palette: Appearance) -> None:
    """只读核对全部可发现模组的视觉网格与原语。"""
    # MjSpec 会解析默认类与 childclass 继承，校验过程仅访问内存中的几何。
    for directory in (mjcf_dir / "sensors", mjcf_dir / "accessories"):
        for path in sorted(directory.glob("*.xml")):
            module = mujoco.MjSpec.from_file(str(path))
            for geom in module.geoms:
                if is_visual(geom):
                    key = geom.meshname or geom.name
                    if palette.mesh_classes.get(key) not in palette.materials:
                        raise ValueError(f"{path}：视觉件 {key!r} 缺少有效的材质类别映射")


def main() -> int:
    parser = argparse.ArgumentParser(description="构建模块化夹爪描述仓库资产。")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--paxini-export-dir", type=Path, default=DEFAULT_PAXINI_EXPORT_DIR)
    parser.add_argument("--out-root", type=Path, default=ROOT)
    parser.add_argument(
        "--appearance", choices=("realistic", "cad"), default="realistic", help="生成资产的外观预设"
    )
    args = parser.parse_args()
    build_all(args.export_dir, args.out_root, args.appearance, args.paxini_export_dir)
    print(f"已完成模块化资产构建：{args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
