#!/usr/bin/env python3
"""离屏渲染夹爪模块化资产高清预览图集（保存至 docs/images/）。"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams["figure.constrained_layout.use"] = False
plt.rcParams["figure.autolayout"] = False

from adaptive_tactile_gripper_description import build_gripper_spec, get_joint_positions

OUT_DIR = ROOT / "docs" / "images"
FONT_FAMILIES = ("Noto Sans CJK SC", "Microsoft YaHei", "Noto Sans CJK JP", "WenQuanYi Zen Hei")
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [name for name in FONT_FAMILIES if name in available_fonts] + ["DejaVu Sans"]


def _prepare_spec(
    *,
    left_sensor: str | None = "pts_2_0",
    right_sensor: str | None = "pts_2_0",
    with_camera: bool = True,
    with_ref_box: bool = False,
) -> mujoco.MjSpec:
    """构建带照明、参考坐标与高清离屏缓冲的渲染用 MjSpec。"""
    spec = build_gripper_spec(
        left_sensor=left_sensor,
        right_sensor=right_sensor,
        with_camera=with_camera,
        fixed_base=True,
        backlash_rad=0.0,
        slide_frictionloss_n=0.0,
        drive_frictionloss_nm=0.0,
    )
    # 配置干净的背景色与充足的三点照明
    spec.visual.headlight.ambient = [0.35, 0.35, 0.38]
    spec.visual.headlight.diffuse = [0.65, 0.65, 0.65]
    spec.visual.headlight.specular = [0.25, 0.25, 0.25]
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1440

    spec.worldbody.add_light(
        name="key_light", pos=[0.4, -0.5, 0.8], dir=[-0.4, 0.5, -0.7], diffuse=[0.85, 0.85, 0.85]
    )
    spec.worldbody.add_light(
        name="fill_light", pos=[-0.5, 0.4, 0.4], dir=[0.5, -0.4, -0.4], diffuse=[0.45, 0.48, 0.52]
    )

    if with_ref_box:
        box = spec.worldbody.add_body(name="ref_cube", pos=[0.0, 0.0, 0.14555])
        box.add_geom(
            name="ref_cube_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.015, 0.015, 0.015],
            rgba=[0.18, 0.68, 0.48, 0.82],
            contype=0,
            conaffinity=0,
            group=2,
        )
    return spec


def _render_view(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    width: int = 960,
    height: int = 720,
    lookat: list[float] | None = None,
    distance: float = 0.36,
    azimuth: float = 132.0,
    elevation: float = -22.0,
    camera_name: str | None = None,
    show_taxels: bool = False,
    show_sites: bool = False,
) -> np.ndarray:
    """用给定视角和可视化分组渲染单帧 RGB 图像。"""
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    if show_sites:
        model.vis.scale.framelength = 0.10
        model.vis.scale.framewidth = 0.015
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
        opt.geomgroup[2] = 1
        opt.geomgroup[3] = 0
        opt.geomgroup[5] = show_taxels
        opt.sitegroup[3] = 1 if show_sites else 0
        opt.sitegroup[4] = 0
        if show_sites:
            opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        if camera_name is not None:
            renderer.update_scene(data, camera=camera_name, scene_option=opt)
        else:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = lookat if lookat is not None else [0.0, -0.015, 0.085]
            cam.distance = distance
            cam.azimuth = azimuth
            cam.elevation = elevation
            renderer.update_scene(data, camera=cam, scene_option=opt)
        return renderer.render().copy()
    finally:
        renderer.close()


def _save_panel(
    images: list[np.ndarray],
    titles: list[str],
    out_path: Path,
    *,
    suptitle: str,
) -> None:
    """将多张渲染帧合成为带中文标题的对比图。"""
    n = len(images)
    fig, axes = plt.subplots(
        1, n, figsize=(5.8 * n, 5.4), dpi=180, facecolor="#11161d"
    )
    fig.set_layout_engine(None)
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles, strict=True):
        ax.imshow(img)
        ax.set_title(title, color="#e6edf3", fontsize=11, pad=10, fontweight="normal")
        ax.axis("off")
    fig.suptitle(suptitle, color="#f0f6fc", fontsize=14, fontweight="normal", y=0.96)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.80, bottom=0.02, wspace=0.03)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_overview_default() -> None:
    """渲染默认完整版 3/4 视图与机载 D435i 第一视角。"""
    spec = _prepare_spec(left_sensor="pts_2_0", right_sensor="pts_2_0", with_camera=True, with_ref_box=True)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    img_iso = _render_view(
        m, d, lookat=[0.0, -0.015, 0.082], distance=0.35, azimuth=135.0, elevation=-22.0
    )
    img_cam = _render_view(m, d, camera_name="cam_d435i_color")
    _save_panel(
        [img_iso, img_cam],
        ["夹爪斜视图（含参考方块）", "D435i 彩色相机视角"],
        OUT_DIR / "overview_default.png",
        suptitle="默认配置 · 双 PTS 2.0 与 RealSense D435i",
    )


def render_modular_variants() -> None:
    """渲染 3 种模块化配置对比图。"""
    configs = [
        ("pts_2_0", "pts_2_0", True, "双 PTS 2.0 与 D435i"),
        ("pts_2_0", "flat_pad", False, "左 PTS 2.0、右平垫、无相机"),
        (None, None, False, "底盘（保留模组安装面）"),
    ]
    imgs, titles = [], []
    for left, right, cam, title in configs:
        spec = _prepare_spec(left_sensor=left, right_sensor=right, with_camera=cam)
        m = spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        imgs.append(
            _render_view(
                m, d, lookat=[0.0, -0.01, 0.085], distance=0.34, azimuth=132.0, elevation=-20.0
            )
        )
        titles.append(title)
    _save_panel(
        imgs,
        titles,
        OUT_DIR / "modular_variants.png",
        suptitle="指尖与相机模组配置",
    )


def render_tactile_collision_and_sites() -> None:
    """渲染指尖视觉特写与触觉碰撞球、局部坐标轴的对照图。"""
    spec = _prepare_spec(left_sensor="pts_2_0", right_sensor="pts_2_0", with_camera=False)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    img_vis = _render_view(
        m,
        d,
        lookat=[0.0, 0.0, 0.145],
        distance=0.14,
        azimuth=128.0,
        elevation=-16.0,
    )
    # 保留视觉指体。隐藏传感器外壳与触柱网格，以显示朝壳内的局部 Z 轴。
    for gid in range(m.ngeom):
        if "taxel_geom_" in m.geom(gid).name:
            m.geom_group[gid] = 5
            m.geom_rgba[gid] = [0.05, 0.88, 1.0, 0.52]
        elif m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_name = m.mesh(m.geom_dataid[gid]).name
            if "pillars_" in mesh_name or mesh_name.endswith("_housing"):
                m.geom_group[gid] = 3
    for sid in range(m.nsite):
        if "_taxel_" in m.site(sid).name:
            m.site_size[sid] = [0.0005, 0.0005, 0.0005]
        else:
            m.site_group[sid] = 4
    closeups = []
    for side, azimuth in (("left", 148.0), ("right", 32.0)):
        center = d.site(f"{side}_taxel_11").xpos.copy()
        center[2] -= 0.006
        closeups.append(_render_view(
            m, d, width=480, lookat=center.tolist(), distance=0.070,
            azimuth=azimuth, elevation=-32.0, show_taxels=True, show_sites=True,
        ))
    img_col = np.hstack(closeups)
    _save_panel(
        [img_vis, img_col],
        ["PTS 2.0 外壳与 3×3 触柱网格", "左右触觉阵列与局部坐标轴（碰撞球半径 2.8 mm）"],
        OUT_DIR / "tactile_collision_and_sites.png",
        suptitle="指尖触觉几何与受力站点",
    )


def render_closure_sequence() -> None:
    """渲染闭环曲柄滑块在 0.00 / 0.75 / 1.45 rad 下的对称闭合序列。"""
    spec = _prepare_spec(left_sensor="pts_2_0", right_sensor="pts_2_0", with_camera=False)
    m = spec.compile()
    d = mujoco.MjData(m)
    imgs, titles = [], []
    for q in (0.0, 0.75, 1.45):
        positions = get_joint_positions(q)
        for name, value in positions.items():
            if name != "gripper_backlash":
                d.joint(name).qpos[0] = value
        mujoco.mj_forward(m, d)
        ql = positions["left_finger_slide"] * 1000
        qr = positions["right_finger_slide"] * 1000
        imgs.append(
            _render_view(
                m, d, lookat=[0.0, 0.0, 0.095], distance=0.28, azimuth=90.0, elevation=-18.0
            )
        )
        titles.append(f"q = {q:.2f} rad · 左 {ql:.1f} mm · 右 {qr:.1f} mm")

    _save_panel(
        imgs,
        titles,
        OUT_DIR / "closure_sequence.png",
        suptitle="闭环曲柄滑块的对称闭合",
    )


def main() -> int:
    render_overview_default()
    render_modular_variants()
    render_tactile_collision_and_sites()
    render_closure_sequence()
    print(f"已生成 4 张预览图到 {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
