#!/usr/bin/env python3
"""离屏渲染夹爪模块化资产高清预览图集（保存至 docs/images/）。"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np
from _style import panel_font
from PIL import Image, ImageDraw

from adaptive_tactile_gripper_description import (
    build_gripper_spec,
    get_joint_positions,
    get_stroke_limits,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "images"


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
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 8
    spec.visual.headlight.ambient = [0.26, 0.26, 0.28]
    spec.visual.headlight.diffuse = [0.14, 0.14, 0.14]
    spec.visual.headlight.specular = [0, 0, 0]
    spec.visual.map.fogstart = 1.4
    spec.visual.map.fogend = 3.1
    spec.visual.rgba.fog = [0.085, 0.109, 0.14, 1]
    spec.visual.rgba.haze = [0.085, 0.109, 0.14, 1]
    spec.add_texture(
        name="studio_sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        width=256,
        height=1536,
        rgb1=[0.075, 0.105, 0.15],
        rgb2=[0.13, 0.16, 0.20],
    )
    spec.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        width=128,
        height=128,
        rgb1=[0.111, 0.133, 0.16],
        markrgb=[0.128, 0.153, 0.18],
    )
    ground = spec.add_material(
        name="studio_floor",
        texrepeat=[32, 32],
        texuniform=True,
        specular=0.03,
        shininess=0.1,
        reflectance=0,
    )
    ground.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        pos=[0, 0, -0.0051 if with_camera else -0.0001],
        material=ground.name,
        contype=0,
        conaffinity=0,
    )
    # 暖主光形成落影，冷补光保留暗部，后侧轮廓光勾出黑色电机与支架。
    for name, position, diffuse, specular in (
        ("key", [0.18, -0.27, 0.46], [0.72, 0.68, 0.62], [0.34, 0.31, 0.28]),
        ("fill", [-0.30, -0.12, 0.24], [0.24, 0.28, 0.34], [0.10, 0.11, 0.14]),
        ("rim", [-0.09, 0.24, 0.33], [0.38, 0.43, 0.48], [0.25, 0.28, 0.31]),
    ):
        spec.worldbody.add_light(
            name=name,
            pos=position,
            dir=np.array([0, 0, 0.07]) - position,
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=diffuse,
            specular=specular,
            castshadow=name == "key",
        )

    if with_ref_box:
        box = spec.worldbody.add_body(name="ref_cube", pos=spec.site("tcp").pos)
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
    show_site_frames: bool = True,
    light_samples: int = 16,
) -> np.ndarray:
    """用给定视角和可视化分组渲染单帧 RGB 图像。"""
    model.vis.global_.offwidth = 2 * width
    model.vis.global_.offheight = 2 * height
    if show_sites and show_site_frames:
        model.vis.scale.framelength = 0.10
        model.vis.scale.framewidth = 0.011
    renderer = mujoco.Renderer(model, height=2 * height, width=2 * width)
    try:
        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
        opt.geomgroup[2] = 1
        opt.geomgroup[3] = 0
        opt.geomgroup[5] = show_taxels
        opt.sitegroup[3] = 1 if show_sites else 0
        opt.sitegroup[4] = 0
        if show_sites and show_site_frames:
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
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = True
        # 在主光方向周围的圆盘内确定性采样，平均各帧以得到随距离变宽的半影。
        key = next(
            light for light in renderer.scene.lights[: renderer.scene.nlight] if light.castshadow
        )
        direction = key.dir.copy()
        tangent = np.cross(direction, [0, 0, 1])
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(direction, tangent)
        total = np.zeros((height * 2, width * 2, 3), dtype=np.float32)
        for i in range(light_samples):
            angle = i * np.pi * (3 - np.sqrt(5))
            radius = 0.28 * np.sqrt((i + 0.5) / light_samples)
            sample = direction + radius * (np.cos(angle) * tangent + np.sin(angle) * bitangent)
            key.dir[:] = sample / np.linalg.norm(sample)
            total += renderer.render()
        frame = Image.fromarray(np.rint(total / light_samples).astype(np.uint8))
        return np.asarray(frame.resize((width, height), Image.Resampling.LANCZOS))
    finally:
        renderer.close()


def _save_panel(
    images: list[np.ndarray],
    titles: list[str],
    out_path: Path,
    *,
    suptitle: str,
    label_size: int = 37,
) -> None:
    """将多张渲染帧合成为带中文标题的对比图。"""
    margin, gutter = 30, 18
    width, height = images[0].shape[1], images[0].shape[0]
    panel = Image.new(
        "RGB",
        (2 * margin + len(images) * width + (len(images) - 1) * gutter, height + 174),
        "#121922",
    )
    draw = ImageDraw.Draw(panel)
    title_font = panel_font(46)
    label_font = panel_font(label_size)
    draw.text((margin, 22), suptitle, font=title_font, fill="#f2f5f7", anchor="lt")
    draw.line((margin, 85, panel.width - margin, 85), fill="#3d4b5a", width=2)
    for index, (array, title) in enumerate(zip(images, titles, strict=True)):
        x = margin + index * (width + gutter)
        draw.text((x, 102), title, font=label_font, fill="#ced9e2", anchor="lt")
        panel.paste(Image.fromarray(array), (x, 152))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def render_hero() -> None:
    """单独展示双侧 PTS 与侧挂相机的三分之四视图。"""
    model = _prepare_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frame = _render_view(
        model,
        data,
        width=1440,
        height=900,
        lookat=[0, -0.015, 0.076],
        distance=0.31,
        azimuth=52,
        elevation=-18,
        light_samples=32,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(OUT_DIR / "hero.png")


def render_overview_default() -> None:
    """渲染默认完整版 3/4 视图与机载 D435i 第一视角。"""
    spec = _prepare_spec(
        left_sensor="pts_2_0", right_sensor="pts_2_0", with_camera=True, with_ref_box=True
    )
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    img_iso = _render_view(
        m, d, lookat=[0.0, -0.015, 0.082], distance=0.35, azimuth=135.0, elevation=-22.0
    )
    img_cam = _render_view(m, d, camera_name="cam_d435i_color")
    _save_panel(
        [img_iso, img_cam],
        ["夹爪斜视图（含参考方块）", "D435i 彩色相机（完整视场）"],
        OUT_DIR / "overview_default.png",
        suptitle="默认配置 · 双 PTS 2.0 与 RealSense D435i",
    )


def render_modular_variants() -> None:
    """渲染 4 种模块化配置对比图（含 PaXini M2824）。"""
    configs = [
        ("pts_2_0", "pts_2_0", True, "双 PTS 2.0 与 D435i"),
        ("paxini_m2824", "paxini_m2824", True, "双 PaXini M2824 与 D435i"),
        ("pts_2_0", "flat_pad", False, "左 PTS 2.0、右平垫、无相机"),
        (None, None, False, "底盘（保留模组安装面）"),
    ]
    imgs, titles = [], []
    for left, right, cam, title in configs:
        spec = _prepare_spec(left_sensor=left, right_sensor=right, with_camera=cam)
        spec.light("fill").diffuse = [0.34, 0.38, 0.44]
        m = spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        imgs.append(
            _render_view(
                m,
                d,
                width=760,
                height=640,
                lookat=[0.0, -0.01, 0.085],
                distance=0.35,
                azimuth=132.0,
                elevation=-20.0,
            )
        )
        titles.append(title)
    _save_panel(
        imgs,
        titles,
        OUT_DIR / "modular_variants.png",
        suptitle="指尖与相机模组配置",
        label_size=33,
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
        closeups.append(
            _render_view(
                m,
                d,
                width=480,
                lookat=center.tolist(),
                distance=0.070,
                azimuth=azimuth,
                elevation=-32.0,
                show_taxels=True,
                show_sites=True,
            )
        )
    img_col = Image.fromarray(np.hstack(closeups))
    draw = ImageDraw.Draw(img_col)
    for x, side in ((20, "左指"), (500, "右指")):
        draw.text(
            (x, 16), side, font=panel_font(34), fill="white", stroke_width=2, stroke_fill="#17202a"
        )
    # 对应 MuJoCo 站点局部轴的原生红、绿、蓝配色。
    for i, (axis, color) in enumerate(zip("XYZ", ("#ff5555", "#69ec6a", "#6590ff"), strict=False)):
        draw.text(
            (773 + i * 54, 16),
            axis,
            font=panel_font(34),
            fill=color,
            stroke_width=2,
            stroke_fill="#17202a",
        )
    _save_panel(
        [img_vis, np.asarray(img_col)],
        ["PTS 2.0 外壳与 3×3 触柱网格", "局部坐标轴与碰撞球（半径 2.8 mm）"],
        OUT_DIR / "tactile_collision_and_sites.png",
        suptitle="指尖触觉几何与受力站点",
    )


def render_paxini_m2824_mounting() -> None:
    """渲染 PaXini M2824 安装特写、双侧零间隙闭合与 135 测点盒式碰撞示意。"""
    spec = _prepare_spec(left_sensor="paxini_m2824", right_sensor="paxini_m2824", with_camera=False)
    spec.light("fill").diffuse = [0.34, 0.38, 0.44]
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    # 1. 全开状态指尖特写（展示 3.5 mm PLA-CF 转接垫块与斜倒角传感头）
    img_mount = _render_view(
        m,
        d,
        lookat=[0.0, 0.0, 0.145],
        distance=0.145,
        azimuth=128.0,
        elevation=-16.0,
    )

    # 2. 双侧零间隙首次接触闭合位置
    contact_q = float(get_stroke_limits("paxini_m2824", "paxini_m2824")["first_contact_q_rad"])
    positions = get_joint_positions(contact_q)
    for name, value in positions.items():
        if name != "gripper_backlash":
            d.joint(name).qpos[0] = value
    mujoco.mj_forward(m, d)
    img_closed = _render_view(
        m,
        d,
        lookat=[0.0, 0.0, 0.144],
        distance=0.115,
        azimuth=90.0,
        elevation=-10.0,
    )

    # 3. 左指盒式碰撞包络（隐藏传感器视觉网格避免共面 Z-fighting）与 135 测点分布
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    for gid in range(m.ngeom):
        gname = m.geom(gid).name
        if gname == "left_pad_collision":
            m.geom_group[gid] = 5
            m.geom_rgba[gid] = [0.05, 0.85, 0.98, 0.45]
        elif gname == "left_sensor_housing_collision":
            m.geom_group[gid] = 5
            m.geom_rgba[gid] = [0.95, 0.65, 0.18, 0.48]
        elif m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_name = m.mesh(m.geom_dataid[gid]).name
            if m.body(m.geom_bodyid[gid]).name == "left_sensor_body" and mesh_name.startswith(
                "left_paxini_m2824"
            ):
                m.geom_group[gid] = 3
    for sid in range(m.nsite):
        sname = m.site(sid).name
        if sname.startswith("left_taxel_"):
            m.site_group[sid] = 3
            m.site_size[sid] = [0.00058, 0.00058, 0.00058]
            m.site_rgba[sid] = [0.22, 0.96, 0.48, 1.0]
        elif sname == "left_pad_center":
            m.site_group[sid] = 3
            m.site_size[sid] = [0.0011, 0.0011, 0.0011]
            m.site_rgba[sid] = [1.0, 0.28, 0.28, 1.0]
        else:
            m.site_group[sid] = 4

    center = d.site("left_pad_center").xpos.copy()
    center[2] -= 0.003
    img_sites = Image.fromarray(
        _render_view(
            m,
            d,
            lookat=center.tolist(),
            distance=0.068,
            azimuth=144.0,
            elevation=-18.0,
            show_taxels=True,
            show_sites=True,
            show_site_frames=False,
        )
    )
    draw = ImageDraw.Draw(img_sites)
    legend_items = (
        ("#22e2ff", "pad_collision"),
        ("#f5a933", "housing_collision"),
        ("#38f57a", "135 测点"),
        ("#ff4c4c", "pad_center"),
    )
    lx = 24
    legend_font = panel_font(26)
    for color, text in legend_items:
        draw.ellipse((lx, 26, lx + 16, 42), fill=color, outline="#17202a")
        draw.text(
            (lx + 24, 18),
            text,
            font=legend_font,
            fill="#f2f5f7",
            stroke_width=2,
            stroke_fill="#17202a",
        )
        lx += int(draw.textlength(text, font=legend_font)) + 44

    slide_mm = positions["left_finger_slide"] * 1000.0
    _save_panel(
        [img_mount, img_closed, np.asarray(img_sites)],
        [
            "指尖装配特写（3.5 mm 转接垫块）",
            f"零间隙闭合（q = {contact_q:.2f} rad，单指 {slide_mm:.2f} mm）",
            "盒式碰撞包络与 135 测点分布",
        ],
        OUT_DIR / "paxini_m2824_mounting.png",
        suptitle="PaXini M2824 触觉模组安装与碰撞示意",
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
                m, d, lookat=[0.0, 0.0, 0.095], distance=0.30, azimuth=90.0, elevation=-18.0
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
    render_hero()
    render_overview_default()
    render_modular_variants()
    render_tactile_collision_and_sites()
    render_paxini_m2824_mounting()
    render_closure_sequence()
    print(f"已生成 6 张预览图到 {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
