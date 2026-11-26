"""验证模块化夹爪描述仓库的 MuJoCo 编译、质量标定、闭环运动学、触觉符号与相机模型。"""

from __future__ import annotations

import math
from typing import get_type_hints

import mujoco
import numpy as np
import pytest

from adaptive_tactile_gripper_description import (
    build_gripper_spec,
    get_data_root,
    get_mjcf_path,
    get_sensor_spec,
    get_specs,
    list_accessories,
    list_sensors,
)


def _crank_slider_aperture(q: float, specs: dict) -> float:
    k = specs["kinematics"]
    r = k["crank_radius_m"]
    link_length = k["link_length_m"]
    off = k["offset_m"]
    alpha = q + k["theta0_rad"]
    radicand = link_length**2 - (r * math.sin(alpha) - off) ** 2
    return 2.0 * (r * math.cos(alpha) + math.sqrt(radicand))


def test_build_gripper_spec_return_annotation():
    assert get_type_hints(build_gripper_spec)["return"] is mujoco.MjSpec


def test_package_discovery_and_specs():
    """验证资产列表、单一真值规格、电机/相机参数和传感器规格均可正常读取。"""
    assert "pts_2_0" in list_sensors()
    assert "flat_pad" in list_sensors()
    assert "paxini_m2824" in list_sensors()
    assert "realsense_d435i" in list_accessories()

    specs = get_specs()
    assert math.isclose(specs["kinematics"]["crank_radius_m"], 0.025, abs_tol=1e-9)
    assert specs["geometry"]["tool_center_in_flange_m"] == [0.0, 0.0, 0.14555]
    assert specs["actuator"]["model"] == "DM-J4310P-2EC"
    assert math.isclose(specs["camera"]["bare_camera_mass_kg"], 0.072, abs_tol=1e-6)
    assert math.isclose(
        specs["materials_and_mass_budget"]["breakdown_kg"]["total_without_camera"],
        0.686,
        abs_tol=1e-4,
    )

    pts = get_sensor_spec("pts_2_0")
    assert pts["tactile"]["rows"] == 3 and pts["tactile"]["cols"] == 3
    assert math.isclose(pts["mass_kg"], 0.015, abs_tol=1e-6)
    assert len(pts["taxels_in_sensor_mount"]) == 9

    paxini = get_sensor_spec("paxini_m2824")
    assert paxini["tactile"]["mode"] == "pad_sites"
    assert paxini["tactile"]["count"] == 135
    assert len(paxini["taxels_in_sensor_mount"]) == 135
    assert math.isclose(paxini["thickness_to_tip_m"], 0.0037, abs_tol=1e-9)


def test_mass_budget_matches_686g_and_no_double_counting():
    """验证无相机整机严格匹配 686 g、带相机为 777 g 附近，且拆卸传感器无重复计重。"""
    m_no_cam = mujoco.MjModel.from_xml_path(str(get_mjcf_path("gripper_no_camera")))
    mass_no_cam = float(np.sum(m_no_cam.body_mass[1:]))
    assert math.isclose(mass_no_cam, 0.686, abs_tol=2e-4), f"got {mass_no_cam}"

    m_with_cam = mujoco.MjModel.from_xml_path(str(get_mjcf_path("gripper")))
    mass_with_cam = float(np.sum(m_with_cam.body_mass[1:]))
    expected_cam = get_specs()["camera"]["total_module_mass_kg"]
    assert math.isclose(mass_with_cam, 0.686 + expected_cam, abs_tol=2e-4)
    assert m_with_cam.camera("cam_d435i_color").id >= 0
    assert m_with_cam.camera("cam_d435i_depth").id >= 0

    m_chassis = build_gripper_spec(
        left_sensor=None, right_sensor=None, with_camera=False, fixed_base=True
    ).compile()
    mass_chassis = float(np.sum(m_chassis.body_mass[1:]))
    # 拆掉两只 15 g PTS 2.0 后精确减少 30 g（656 g），证明无重复计重
    assert math.isclose(mass_no_cam - mass_chassis, 0.030, abs_tol=1e-4)


def test_prebuilt_and_modular_mjcf_compile():
    """验证预生成 MJCF 及动态模组组合能编译并保留预期触点。"""
    for variant in ("gripper_chassis", "gripper", "gripper_no_camera"):
        model = mujoco.MjModel.from_xml_path(str(get_mjcf_path(variant)))
        assert model.nu == 1

    combos = [
        ("pts_2_0", "pts_2_0", True),
        ("pts_2_0", "flat_pad", False),
        ("flat_pad", "flat_pad", True),
        ("paxini_m2824", "paxini_m2824", True),
        ("pts_2_0", "paxini_m2824", False),
        (None, None, False),
    ]
    for left, right, cam in combos:
        spec = build_gripper_spec(
            left_sensor=left, right_sensor=right, with_camera=cam, fixed_base=True
        )
        m = spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        assert m.actuator("gripper_drive").id >= 0
        if left == "pts_2_0":
            for r in range(3):
                for c in range(3):
                    gid = m.geom(f"left_taxel_geom_{r}{c}").id
                    assert m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
                    assert math.isclose(float(m.geom_size[gid, 0]), 0.0028, abs_tol=1e-6)
                    assert m.site(f"left_taxel_{r}{c}").id >= 0
        if right == "flat_pad":
            assert m.geom("right_pad_collision").id >= 0
            assert m.site("right_pad_center").id >= 0
        for side, sensor_name in (("left", left), ("right", right)):
            if sensor_name == "paxini_m2824":
                gid = m.geom(f"{side}_pad_collision").id
                assert m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
                body_id = m.body(f"{side}_sensor_body").id
                body_geoms = np.flatnonzero(m.geom_bodyid == body_id)
                assert all(m.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE for g in body_geoms)
                assert m.site(f"{side}_pad_center").id >= 0
                assert m.site(f"{side}_taxel_001").id >= 0
                assert m.site(f"{side}_taxel_135").id >= 0


def test_taxel_poses_match_onshape_export():
    """逐点对照原始导出的球心位置与受力坐标系。"""
    raw_xml = get_data_root() / "tools" / "onshape_export" / "raw_gripper.xml"
    if not raw_xml.is_file():
        pytest.skip(f"缺少可选 Onshape 原始导出：{raw_xml}")
    raw = mujoco.MjModel.from_xml_path(str(raw_xml))
    modular = build_gripper_spec(fixed_base=True).compile()
    raw_data, modular_data = mujoco.MjData(raw), mujoco.MjData(modular)
    mujoco.mj_forward(raw, raw_data)
    mujoco.mj_forward(modular, modular_data)
    for side in ("left", "right"):
        for row in range(3):
            for col in range(3):
                name = f"{side}_taxel_{row}{col}"
                np.testing.assert_allclose(
                    raw_data.site(name).xpos, modular_data.site(name).xpos, atol=1e-7
                )
                np.testing.assert_allclose(
                    raw_data.site(name).xmat, modular_data.site(name).xmat, atol=1e-7
                )


def test_closed_loop_crank_slider_kinematics_sweep():
    """验证闭环曲柄滑块左右对称性与理论行程一致（误差 < 0.5 mm）。"""
    specs = get_specs()
    spec = build_gripper_spec(
        left_sensor="pts_2_0",
        right_sensor="pts_2_0",
        with_camera=False,
        fixed_base=True,
        backlash_rad=0.0,
        slide_frictionloss_n=0.0,
        drive_frictionloss_nm=0.0,
    )
    m = spec.compile()
    m.actuator_gainprm[0, 0] = 200.0
    m.actuator_biasprm[0, 1] = -200.0
    m.actuator_biastype[0] = mujoco.mjtBias.mjBIAS_AFFINE
    d = mujoco.MjData(m)

    q_left_adr = m.jnt_qposadr[m.joint("left_finger_slide").id]
    q_right_adr = m.jnt_qposadr[m.joint("right_finger_slide").id]
    q_drive_adr = m.jnt_qposadr[m.joint("gripper_drive").id]
    a0 = _crank_slider_aperture(0.0, specs)

    for target_q in (0.0, 0.4, 0.8, 1.2, 1.5):
        d.ctrl[0] = target_q
        for _ in range(800):
            mujoco.mj_step(m, d)
        ql = float(d.qpos[q_left_adr])
        qr = float(d.qpos[q_right_adr])
        qd = float(d.qpos[q_drive_adr])
        assert abs(ql - qr) < 3e-4
        expected = a0 - _crank_slider_aperture(qd, specs)
        assert abs((ql + qr) - expected) < 5e-4


def test_pts_2_0_tactile_normal_force_positive_on_compression():
    """验证指间夹紧方块时，左右所有接触触点在局部 site 坐标系下均输出正法向力 Fz > 0。"""
    spec = build_gripper_spec(
        left_sensor="pts_2_0",
        right_sensor="pts_2_0",
        with_camera=False,
        fixed_base=True,
        backlash_rad=0.0,
    )
    cube = spec.worldbody.add_body(name="test_box", pos=[0.0, 0.0, 0.14555])
    cube.add_geom(
        name="test_box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.020, 0.015, 0.015],
        rgba=[0.2, 0.7, 0.3, 1.0],
    )
    m = spec.compile()
    d = mujoco.MjData(m)
    d.ctrl[0] = 1.0
    for _ in range(800):
        mujoco.mj_step(m, d)

    taxel_sites = {}
    for side in ("left", "right"):
        for r in range(3):
            for c in range(3):
                gid = m.geom(f"{side}_taxel_geom_{r}{c}").id
                sid = m.site(f"{side}_taxel_{r}{c}").id
                taxel_sites[gid] = (side, sid)

    contact_forces = {"left": [], "right": []}
    cfrc = np.empty(6, dtype=float)
    for cid in range(d.ncon):
        con = d.contact[cid]
        for gid in (con.geom1, con.geom2):
            if gid not in taxel_sites:
                continue
            side, sid = taxel_sites[gid]
            mujoco.mj_contactForce(m, d, cid, cfrc)
            f_on_geom2_w = np.asarray(con.frame).reshape(3, 3).T @ cfrc[:3]
            f_on_pillar_w = f_on_geom2_w if gid == con.geom2 else -f_on_geom2_w
            f_local = np.asarray(d.site_xmat[sid]).reshape(3, 3).T @ f_on_pillar_w
            contact_forces[side].append(float(f_local[2]))

    assert len(contact_forces["left"]) >= 3
    assert len(contact_forces["right"]) >= 3
    assert all(fz > 0.05 for fz in contact_forces["left"])
    assert all(fz > 0.05 for fz in contact_forces["right"])


def test_paxini_m2824_pad_normal_force_positive_on_compression():
    """验证 paxini_m2824 盒式平垫夹紧方块时，接触发生于 pad_collision 且局部 Fz > 0。"""
    spec = build_gripper_spec(
        left_sensor="paxini_m2824",
        right_sensor="paxini_m2824",
        with_camera=False,
        fixed_base=True,
        backlash_rad=0.0,
    )
    cube = spec.worldbody.add_body(name="test_box", pos=[0.0, 0.0, 0.14555])
    cube.add_geom(
        name="test_box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.020, 0.015, 0.015],
        rgba=[0.2, 0.7, 0.3, 1.0],
    )
    m = spec.compile()
    d = mujoco.MjData(m)
    d.ctrl[0] = 1.0
    for _ in range(800):
        mujoco.mj_step(m, d)

    pad_sites = {
        m.geom(f"{side}_pad_collision").id: (side, m.site(f"{side}_pad_center").id)
        for side in ("left", "right")
    }
    contact_forces = {"left": [], "right": []}
    cfrc = np.empty(6, dtype=float)
    for cid in range(d.ncon):
        con = d.contact[cid]
        for gid in (con.geom1, con.geom2):
            if gid not in pad_sites:
                continue
            side, sid = pad_sites[gid]
            mujoco.mj_contactForce(m, d, cid, cfrc)
            f_on_geom2_w = np.asarray(con.frame).reshape(3, 3).T @ cfrc[:3]
            f_on_pad_w = f_on_geom2_w if gid == con.geom2 else -f_on_geom2_w
            f_local = np.asarray(d.site_xmat[sid]).reshape(3, 3).T @ f_on_pad_w
            contact_forces[side].append(float(f_local[2]))

    assert contact_forces["left"] and contact_forces["right"]
    assert sum(contact_forces["left"]) > 0.1
    assert sum(contact_forces["right"]) > 0.1


@pytest.mark.parametrize(
    ("left_sensor", "right_sensor"),
    [
        ("pts_2_0", "pts_2_0"),
        ("pts_2_0", "flat_pad"),
        ("flat_pad", "flat_pad"),
        ("pts_2_0", "paxini_m2824"),
        ("paxini_m2824", "paxini_m2824"),
    ],
)
def test_empty_grasp_does_not_jam_and_reopens(left_sensor, right_sensor):
    """验证空夹大扭矩闭合后对置指尖不反转穿模卡死，反向力矩可顺利重新张开。"""
    model = build_gripper_spec(
        left_sensor=left_sensor,
        right_sensor=right_sensor,
        with_camera=False,
        fixed_base=True,
    ).compile()
    for close_torque in (0.3, 4.0):
        data = mujoco.MjData(model)
        data.ctrl[0] = close_torque
        for _ in range(1000):
            mujoco.mj_step(model, data)
        assert data.ncon >= 1
        data.ctrl[0] = -4.0
        for _ in range(1000):
            mujoco.mj_step(model, data)
        q_eff = float(data.joint("gripper_drive").qpos[0] + data.joint("gripper_backlash").qpos[0])
        assert q_eff < 0.02


def test_unloaded_chassis_overshoot_stays_clear_of_opposing_slider():
    """验证空载底盘在 1 N·m 与满量程 4 N·m 急闭步进仿真中，曲柄轴套与对侧滑块保持正间隙。"""
    model = build_gripper_spec(
        left_sensor=None,
        right_sensor=None,
        with_camera=False,
        fixed_base=True,
    ).compile()
    opposing_pairs = []
    for rod_name, opp_finger_name in (
        ("rod_bearing_shim_subassembly", "right_finger"),
        ("rod_bearing_shim_subassembly_2", "left_finger"),
    ):
        rod_bid = model.body(rod_name).id
        opp_bid = model.body(opp_finger_name).id
        spacers = [
            gid
            for gid in range(model.ngeom)
            if model.geom_bodyid[gid] == rod_bid
            and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
            and model.mesh(model.geom_dataid[gid]).name == "link_spacer"
        ]
        blocks = [
            gid
            for gid in range(model.ngeom)
            if model.geom_bodyid[gid] == opp_bid
            and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
            and model.mesh(model.geom_dataid[gid]).name == "mgn9_block"
        ]
        assert len(spacers) == 2 and len(blocks) == 1
        for s_gid in spacers:
            opposing_pairs.append((s_gid, blocks[0]))

    fromto = np.empty(6, dtype=float)
    for torque in (1.0, 4.0):
        data = mujoco.MjData(model)
        data.ctrl[0] = torque
        min_clearance = float("inf")
        for _ in range(400):
            mujoco.mj_step(model, data)
            for s_gid, b_gid in opposing_pairs:
                dist = float(mujoco.mj_geomDistance(model, data, s_gid, b_gid, 1.0, fromto))
                min_clearance = min(min_clearance, dist)
        assert min_clearance > 0.0, (torque, min_clearance)


@pytest.mark.parametrize("timestep", [0.004, 0.005])
def test_attached_chassis_reopens_cleanly_at_non_default_timestep(timestep):
    """验证挂载到宿主场景且使用非默认步长（4/5 ms）时，空载底盘急闭后反向张开稳定无发散。"""
    scene = mujoco.MjSpec()
    scene.option.timestep = timestep
    carrier = scene.worldbody.add_body(name="carrier", pos=[0.0, 0.0, 0.5])
    scene.attach(
        build_gripper_spec(
            left_sensor=None,
            right_sensor=None,
            with_camera=False,
            fixed_base=True,
        ),
        prefix="gripper/",
        frame=carrier.add_frame(),
    )
    model = scene.compile()
    steps_per_phase = round(1.0 / timestep)
    for close_torque in (1.0, 4.0):
        data = mujoco.MjData(model)
        prev_time = -1.0
        data.ctrl[0] = close_torque
        for _ in range(steps_per_phase):
            mujoco.mj_step(model, data)
            assert data.time > prev_time
            assert np.all(np.isfinite(data.qacc))
            prev_time = float(data.time)
        data.ctrl[0] = -4.0
        for _ in range(steps_per_phase):
            mujoco.mj_step(model, data)
            assert data.time > prev_time
            assert np.all(np.isfinite(data.qacc))
            prev_time = float(data.time)
        assert math.isclose(float(data.time), 2.0, abs_tol=1e-6)
        q_eff = float(
            data.joint("gripper/gripper_drive").qpos[0]
            + data.joint("gripper/gripper_backlash").qpos[0]
        )
        assert q_eff < 0.02
