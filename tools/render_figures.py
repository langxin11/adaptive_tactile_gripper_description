#!/usr/bin/env python3
"""从随包 MJCF、运动学和质量账本生成三张中文出版图。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

# 源码脚本也可由下游 Python 环境直接调用。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray

from adaptive_tactile_gripper_description import (
    component_mass,
    get_closure_jacobian,
    get_joint_positions,
    get_mjcf_path,
    get_specs,
)

if TYPE_CHECKING or __package__:
    from ._style import COLORS, WIDTH_IN, XYZ_COLORS, label_panels, save_figure, use_style
else:
    from _style import COLORS, WIDTH_IN, XYZ_COLORS, label_panels, save_figure, use_style

_mj: Any = mujoco
OUT_DIR = Path(__file__).resolve().parents[1] / "docs/figures"
BLUE, ORANGE = COLORS["pla_cf_structure"], COLORS["finger_brace"]


def frame_data(aspect=4 / 3):
    """将编译后的网格和各坐标系表达在 base 系，长度统一换算为毫米。"""
    model = _mj.MjModel.from_xml_path(str(get_mjcf_path()))
    data = _mj.MjData(model)
    _mj.mj_forward(model, data)
    base = data.body("base")
    origin, basis = base.xpos.copy(), base.xmat.reshape(3, 3)
    frames = {"base": ((base.xpos - origin) @ basis * 1000, basis.T @ basis)}
    for key, name in (
        ("tcp", "tcp"),
        ("color", "cam_d435i_color_optical_frame"),
        ("depth", "cam_d435i_depth_optical_frame"),
    ):
        site = data.site(name)
        frames[key] = ((site.xpos - origin) @ basis * 1000, basis.T @ site.xmat.reshape(3, 3))

    meshes = []
    for gid in np.flatnonzero(model.geom_group == 2):
        mid = model.geom_dataid[gid]
        if model.geom_type[gid] != _mj.mjtGeom.mjGEOM_MESH:
            continue
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vertices = model.mesh_vert[start : start + count] @ data.geom_xmat[gid].reshape(3, 3).T
        meshes.append((vertices + data.geom_xpos[gid] - origin) @ basis * 1000)

    optics = {}
    for key in ("color", "depth"):
        center, rotation = frames[key]
        target = rotation.T @ (frames["tcp"][0] - center)
        vertical = np.deg2rad(model.cam("cam_d435i_" + key).fovy.item()) / 2
        # 模型仅给出垂直视场，水平视场采用与预览相同的 4:3 画幅。
        tangents = np.tan(vertical) * np.array([aspect, 1.0])
        extent = target[2]
        axis_end = center + extent * rotation[:, 2]
        sections = [
            (axis_end - extent * t * rotation[:, i], axis_end + extent * t * rotation[:, i])
            for i, t in enumerate(tangents)
        ]
        optics[key] = dict(
            sections=sections,
            end=axis_end,
            fov=np.rad2deg(2 * np.arctan(tangents)),
            offsets=np.rad2deg(np.arctan2(target[:2], extent)),
            angle=np.rad2deg(np.arctan2(np.linalg.norm(target[:2]), extent)),
        )
    return frames, meshes, optics


def convex_outline(points):
    """求投影点集的二维凸包，用淡色分件轮廓定位各坐标系。"""
    ordered = sorted(set(map(tuple, points)))

    def half(sequence):
        hull = []
        for p in sequence:
            while len(hull) >= 2:
                a, b = hull[-2:]
                if (b[0] - a[0]) * (p[1] - a[1]) > (b[1] - a[1]) * (p[0] - a[0]):
                    break
                hull.pop()
            hull.append(p)
        return hull[:-1]

    return np.asarray(half(ordered) + half(reversed(ordered)))


def frames_figure():
    frames, meshes, optics = frame_data()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN["double"], 3.65))
    fig.subplots_adjust(left=0.073, right=0.982, bottom=0.31, top=0.83, wspace=0.35)
    # 斜视采用正交投影，使三根局部轴沿各自真实方向投影。
    right = np.array([0.84, 0.54, 0])
    right /= np.linalg.norm(right)
    up = np.array([-right[1] * 0.34, right[0] * 0.34, np.sqrt(1 - 0.34**2)])
    projections = (np.array([right, up]), np.eye(3)[[0, 2]], np.eye(3)[[1, 2]])
    markers = {"base": "s", "tcp": "D", "color": "o", "depth": "^"}
    for i, (ax, projection) in enumerate(zip(axes, projections, strict=True)):
        for mesh in meshes:
            hull = convex_outline(mesh @ projection.T)
            ax.fill(*hull.T, fc="#f3f4f5", ec="#c9ced4", lw=0.33)
        for key in ("depth", "color"):
            center = frames[key][0]
            optics_color = BLUE if key == "depth" else ORANGE
            if i:
                a, b = optics[key]["sections"][i - 1]
                poly = np.array([a, center, b]) @ projection.T
                ax.plot(*poly.T, color=optics_color, lw=0.95, ls="--" if key == "depth" else "-")
                if key == "color":
                    ax.fill(*poly.T, color=optics_color, alpha=0.09)
            if key == "color":
                for point, ls in ((optics[key]["end"], ":"), (frames["tcp"][0], "--")):
                    ax.plot(
                        *(np.array([center, point]) @ projection.T).T,
                        color="#222222",
                        lw=1,
                        ls=ls,
                        zorder=4,
                    )
        for key, (origin, rotation) in frames.items():
            point = projection @ origin
            if i == 0 and key != "depth":
                for vector, color in zip(rotation.T, XYZ_COLORS, strict=True):
                    ax.annotate(
                        "",
                        point + 22 * projection @ vector,
                        point,
                        arrowprops=dict(
                            arrowstyle="-|>",
                            lw=1.15,
                            color=color,
                            shrinkA=0,
                            shrinkB=0,
                            mutation_scale=6,
                        ),
                    )
            # 侧视下两个光心重合，用空心圆套住三角形。
            ax.plot(
                *point,
                marker=markers[key],
                color={"color": ORANGE, "depth": BLUE}.get(key, "#252b30"),
                mfc="none" if key == "color" else None,
                ms=6 if key == "color" else 3,
                mew=1,
                zorder=6,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0.055)
        ax.set_xlabel(["投影横坐标 $u$ (mm)", "$X$ (mm)", "$Y$ (mm)"][i])
        ax.set_title(["坐标系斜视", "正视 · 水平截面", "侧视 · 垂直截面"][i], pad=7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(length=2)
    axes[0].set_ylabel("投影纵坐标 $v$ (mm)")
    axes[1].set_ylabel("$Z$ (mm)", labelpad=0)
    axes[2].set_ylabel("$Z$ (mm)", labelpad=0)
    # 三个面板共用相同的毫米比例，并保留所有视场边界。
    span = max(np.ptp(ax.get_xlim()) for ax in axes)
    bottom = min(ax.get_ylim()[0] for ax in axes)
    top = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        mid = np.mean(ax.get_xlim())
        ax.set_xlim(mid - span / 2, mid + span / 2)
        ax.set_ylim(bottom, top)
    label_panels(axes)
    handles = [
        Line2D(
            [], [], color=c, marker=m, ls="none", mfc="none" if k == "color" else c, ms=4, label=t
        )
        for k, m, c, t in (
            ("base", "s", "#252b30", "基座"),
            ("tcp", "D", "#252b30", "工具中心"),
            ("color", "o", ORANGE, "彩色光心"),
            ("depth", "^", BLUE, "深度光心"),
        )
    ]
    handles += [
        Line2D([], [], color=c, label=t)
        for t, c in zip(("$+X$", "$+Y$", "$+Z$"), XYZ_COLORS, strict=True)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=7,
        columnspacing=1,
        handlelength=1,
        handletextpad=0.4,
    )
    rgb, depth = optics["color"], optics["depth"]
    fig.text(
        0.5,
        0.157,
        f"两光学系同向 · 4∶3 画幅   彩色视场 {rgb['fov'][0]:.1f}° × {rgb['fov'][1]:.1f}°"
        f"     深度视场 {depth['fov'][0]:.1f}° × {depth['fov'][1]:.1f}°",
        ha="center",
    )
    fig.text(
        0.5,
        0.09,
        f"彩色光轴（点线）至工具中心视线（虚线）的空间夹角 {rgb['angle']:.1f}°"
        f"    水平 {rgb['offsets'][0]:+.1f}°，垂直 {rgb['offsets'][1]:+.1f}°",
        ha="center",
    )
    fig.text(
        0.5,
        0.036,
        f"基座系下光心横坐标   彩色 {frames['color'][0][0]:.1f} mm"
        f"    深度 {frames['depth'][0][0]:.1f} mm"
        f"    工具中心高度 {frames['tcp'][0][2]:.2f} mm",
        ha="center",
    )
    return fig


class KinematicsData(TypedDict):
    q: NDArray[np.float64]
    closure: NDArray[np.float64]
    gap: NDArray[np.float64]
    jac: NDArray[np.float64]
    forces: list[NDArray[np.float64]]
    torques: list[float]
    contact: float
    protocol: float


def kinematics_data() -> KinematicsData:
    """两指等载时，输出力矩除以总闭合雅可比即为单指力。"""
    specs = get_specs()
    kin = specs["kinematics"]
    contact = kin["module_contact"]["first_contact_q_rad"]
    lower, upper = kin["mechanism_range_rad"]
    q = np.unique(np.r_[np.linspace(lower, upper, 301), contact])
    closure = np.array([get_joint_positions(v)["left_finger_slide"] for v in q])
    jac = np.array([get_closure_jacobian(v) for v in q])
    torques = [specs["actuator"][k] for k in ("rated_torque_nm", "peak_torque_nm")]
    gap = specs["geometry"]["nominal_full_open_center_pillar_gap_m"] - 2 * closure
    return KinematicsData(
        q=q,
        closure=closure * 1000,
        gap=gap * 1000,
        jac=jac * 1000,
        forces=[t / jac for t in torques],
        torques=torques,
        contact=contact,
        protocol=kin["protocol_range_rad"][1],
    )


def kinematics_figure():
    d = kinematics_data()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN["double"], 3.65))
    fig.subplots_adjust(left=0.075, right=0.982, bottom=0.40, top=0.77, wspace=0.48)
    q = d["q"]
    before = q <= d["contact"]
    axes[0].plot(q, d["closure"], color=BLUE, label="单指闭合量")
    axes[0].plot(q[before], d["gap"][before], color=ORANGE, ls="-.", label="触柱间净开口")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1)
    axes[1].plot(q, d["jac"], color=BLUE)
    for force, torque, name, color, ls in zip(
        d["forces"], d["torques"], ("额定", "峰值"), (BLUE, ORANGE), ("-", "-."), strict=True
    ):
        axes[2].plot(q, force, color=color, ls=ls, label=f"{name} {torque:g} N·m")
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.30))
    for ax, title, unit in zip(
        axes,
        ("闭合与开口", "闭合雅可比", "理论单指力"),
        ("位移 (mm)", r"$\mathrm{d}s/\mathrm{d}q$ (mm/rad)", "$F$ (N)"),
        strict=True,
    ):
        ax.set_title(title, pad=7)
        ax.set_ylabel(unit, labelpad=2)
        ax.set_xlabel("曲柄角 $q$ (rad)", labelpad=1)
        ax.axvspan(q[-1], d["protocol"], color="#eceff2", zorder=0)
        ax.axvline(d["contact"], color="#bf3030", ls="--", lw=0.9)
        ax.axvline(d["protocol"], color="#303238", ls=":", lw=1)
        ax.set_xlim(-0.03, d["protocol"] + 0.035)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="y", color="#e0e3e6", lw=0.45, ls="-")
    label_panels(axes)
    handles = [
        Line2D([], [], ls="--", color="#bf3030", label=f"PTS 首次接触 {d['contact']:.4f} rad"),
        Patch(color="#eceff2", label=r"超出机构行程 $q>\pi/2$"),
        Line2D([], [], ls=":", color="#303238", label=f"协议上限 {d['protocol']:.2f} rad"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.994),
        ncol=3,
        handlelength=1.4,
        columnspacing=1,
    )
    fig.text(
        0.5,
        0.075,
        r"$s$ 为双指总闭合量。对称准静态夹持，忽略摩擦与传动损耗，"
        r"每指 $F=\tau/(\mathrm{d}s/\mathrm{d}q)$。",
        ha="center",
    )
    fig.text(
        0.5,
        0.025,
        "3.5 / 12.5 N·m 为电机规格；默认 MJCF 驱动限幅 ±4 N·m；首次接触至机构上限为理想机构外推。",
        ha="center",
    )
    return fig


def mass_data():
    """按构建时的硬件口径拆账，打印件承接秤测总质量扣除标准件后的余额。"""
    specs = get_specs()
    ledger = specs["materials_and_mass_budget"]["breakdown_kg"]
    motor = specs["actuator"]["total_mass_kg"]
    guide = component_mass.MGN9_RAIL_150MM_MASS_KG + 2 * component_mass.MGN9C_BLOCK_MASS_KG
    fasteners = (
        component_mass.BASE_FASTENERS_MASS_KG
        + component_mass.MOTOR_FASTENERS_MASS_KG
        + 2
        * (
            component_mass.FINGER_FASTENERS_MASS_KG
            + 4 * component_mass.MF84_BEARING_MASS_KG
            + component_mass.ROD_SPACER_HARDWARE_MASS_KG
        )
    )
    sensors = ledger["left_pts_2_0"] + ledger["right_pts_2_0"]
    printed = ledger["total_without_camera"] - motor - guide - fasteners - sensors
    return (
        1000 * np.array([motor, printed, guide, fasteners, sensors]),
        1000
        * np.array(
            [specs["camera"]["bare_camera_mass_kg"], specs["camera"]["mount_and_fasteners_mass_kg"]]
        ),
    )


def mass_figure():
    main, camera = mass_data()
    fig, ax = plt.subplots(figsize=(WIDTH_IN["double"], 2.53))
    fig.subplots_adjust(left=0.125, right=0.96, bottom=0.39, top=0.83)
    names = (
        "电机",
        "打印件",
        "导轨与滑块",
        "轴承等连接件",
        "双侧 PTS",
        "相机本体",
        "相机支架与紧固件",
    )
    keys = (
        "anodized_black",
        "pla_cf_structure",
        "steel",
        "finger_brace",
        "silicone",
        "aluminum",
        "pla_finger",
    )
    handles = []
    for y, values, offset in ((1, main, 0), (0, camera, len(main))):
        start = 0
        for j, value in enumerate(values, offset):
            color = COLORS[keys[j]]
            hatch = ("", "", "/", "xx", "..", "///", "++")[j]
            ax.barh(
                y,
                value,
                left=start,
                color=color,
                edgecolor="#424242",
                hatch=hatch,
                lw=0.5,
                height=0.49,
            )
            handles.append(
                Patch(
                    facecolor=color,
                    hatch=hatch,
                    edgecolor="#424242",
                    label=f"{names[j]}  {value:.3f}".rstrip("0").rstrip(".") + " g",
                )
            )
            start += value
        ax.annotate(
            f"{start:.3f}".rstrip("0").rstrip(".") + " g",
            xy=(start, y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, max(main.sum(), camera.sum()) * 1.11)
    ax.set_yticks([0, 1], ["相机模组", "无相机整机"])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("质量 $m$ (g)", labelpad=1)
    ax.set_title("双侧 PTS 夹爪质量分布", pad=9)
    for row, y in ((handles[: len(main)], 0.16), (handles[len(main) :], 0.078)):
        fig.legend(
            handles=row,
            loc="lower center",
            bbox_to_anchor=(0.5, y),
            ncol=len(row),
            handlelength=0.85,
            columnspacing=0.8,
            handletextpad=0.3,
        )
    fig.text(
        0.5,
        0.036,
        "无相机总质量由电子秤测得，打印件按质量余额估计。四颗法兰安装螺钉另计。",
        ha="center",
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="图表输出目录")
    args = parser.parse_args()
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")
    use_style()
    for name, draw in (
        ("frames", frames_figure),
        ("kinematics", kinematics_figure),
        ("mass_budget", mass_figure),
    ):
        fig = draw()
        save_figure(fig, args.out_dir / name)
        plt.close(fig)
    print(f"已生成三张 PNG 图表到 {args.out_dir}")


if __name__ == "__main__":
    main()
