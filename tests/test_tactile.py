"""验证统一触觉接触对注册与 contact_geom / pad_sites 三轴力读取辅助函数。"""

from __future__ import annotations

from typing import get_type_hints

import mujoco
import numpy as np
import pytest

from adaptive_tactile_gripper_description import (
    TactileForceReader,
    add_tactile_contact_pairs,
    build_gripper_spec,
)


def _attach_gripper_and_box(
    *,
    left_sensor: str | None,
    right_sensor: str | None,
    box_size: tuple[float, float, float] = (0.016, 0.010, 0.010),
    box_pos: tuple[float, float, float] = (0.0, 0.0, 0.14555),
    box_contype: int = 0,
    box_conaffinity: int = 2,
    box_mass: float = 0.05,
    prefix: str = "gripper/",
) -> mujoco.MjSpec:
    scene = mujoco.MjSpec()
    scene.option.gravity = [0.0, 0.0, 0.0]
    scene.option.noslip_iterations = 5
    gripper = build_gripper_spec(
        left_sensor=left_sensor,
        right_sensor=right_sensor,
        with_camera=False,
        fixed_base=True,
        backlash_rad=0.0,
    )
    scene.attach(gripper, prefix=prefix, frame=scene.worldbody.add_frame())

    cube_body = scene.worldbody.add_body(name="cube_body", pos=list(box_pos))
    cube_body.add_joint(name="cube_free", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.05)
    cube_body.add_geom(
        name="cube/target_cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(box_size),
        mass=box_mass,
        contype=box_contype,
        conaffinity=box_conaffinity,
    )
    return scene


def _raw_pad_net_force_local(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pad_geom_name: str,
    ref_site_name: str,
    *,
    other_geom_id: int | None = None,
) -> tuple[np.ndarray, int]:
    pad_id = model.geom(pad_geom_name).id
    ref_rot = np.asarray(data.site_xmat[model.site(ref_site_name).id], dtype=np.float64).reshape(
        3, 3
    )
    net_world = np.zeros(3, dtype=np.float64)
    count = 0
    cfrc = np.empty(6, dtype=np.float64)
    for cid in range(data.ncon):
        con = data.contact[cid]
        if pad_id not in (con.geom1, con.geom2):
            continue
        other = con.geom2 if con.geom1 == pad_id else con.geom1
        if other_geom_id is not None and other != other_geom_id:
            continue
        count += 1
        mujoco.mj_contactForce(model, data, cid, cfrc)
        f_on_g2_w = np.asarray(con.frame, dtype=np.float64).reshape(3, 3).T @ cfrc[:3]
        f_on_pad_w = f_on_g2_w if con.geom2 == pad_id else -f_on_g2_w
        net_world += f_on_pad_w
    return ref_rot.T @ net_world, count


def test_tactile_type_annotations():
    """校验导出接口的公共类型标注可反射解析为 MuJoCo 类型。"""
    assert get_type_hints(add_tactile_contact_pairs)["scene"] is mujoco.MjSpec
    assert get_type_hints(TactileForceReader.__init__)["model"] is mujoco.MjModel
    assert get_type_hints(TactileForceReader.read)["data"] is mujoco.MjData


def test_add_tactile_contact_pairs_deduplicated_and_idempotent():
    """验证 contact_geom 逐球配对、pad_sites 严格单次配对及幂等跳过。"""
    scene = _attach_gripper_and_box(left_sensor="pts_2_0", right_sensor="paxini_m2824")

    # 物体屏蔽默认位掩码（contype=0, conaffinity=2）：未添加显式 <pair> 前闭合时与物体零接触。
    m_before = scene.compile()
    before_cube_gid = m_before.geom("cube/target_cube_geom").id
    d_before = mujoco.MjData(m_before)
    d_before.ctrl[0] = 0.25
    for _ in range(400):
        mujoco.mj_step(m_before, d_before)
    assert not any(before_cube_gid in (con.geom1, con.geom2) for con in d_before.contact)

    added = add_tactile_contact_pairs(
        scene,
        "cube/target_cube_geom",
        left_sensor="pts_2_0",
        right_sensor="paxini_m2824",
        prefix="gripper/",
    )
    assert len(added) == 10
    assert len(scene.pairs) == 10
    assert sum(name.startswith("tactile_object_left_") for name in added) == 9
    assert [name for name in added if name.startswith("tactile_object_right_")] == [
        "tactile_object_right_0"
    ]

    # 重复调用及无触觉组合不再新增重复 pair。
    assert (
        add_tactile_contact_pairs(
            scene,
            "cube/target_cube_geom",
            left_sensor="pts_2_0",
            right_sensor="paxini_m2824",
            prefix="gripper/",
        )
        == []
    )
    assert (
        add_tactile_contact_pairs(
            scene,
            "cube/target_cube_geom",
            left_sensor="flat_pad",
            right_sensor=None,
            prefix="gripper/",
        )
        == []
    )

    m_after = scene.compile()
    after_cube_gid = m_after.geom("cube/target_cube_geom").id
    d_after = mujoco.MjData(m_after)
    d_after.ctrl[0] = 0.25
    for _ in range(400):
        mujoco.mj_step(m_after, d_after)
    assert d_after.ncon == 13
    assert all(after_cube_gid in (con.geom1, con.geom2) for con in d_after.contact)
    touched = {
        m_after.geom(con.geom2 if con.geom1 == after_cube_gid else con.geom1).name
        for con in d_after.contact
    }
    assert "gripper/right_pad_collision" in touched
    assert any(name.startswith("gripper/left_taxel_geom_") for name in touched)


def test_paxini_pad_sites_three_axis_conservation_shear_and_torsion():
    """验证 paxini_m2824 在平面接触下：三维合力严格守恒、多测点响应、且保留剪切与扭矩偶。"""
    scene = _attach_gripper_and_box(left_sensor="paxini_m2824", right_sensor="paxini_m2824")
    added = add_tactile_contact_pairs(
        scene,
        "cube/target_cube_geom",
        left_sensor="paxini_m2824",
        right_sensor="paxini_m2824",
        prefix="gripper/",
    )
    assert len(added) == 2

    model = scene.compile()
    data = mujoco.MjData(model)
    reader = TactileForceReader(
        model,
        left_sensor="paxini_m2824",
        right_sensor="paxini_m2824",
        prefix="gripper/",
    )
    assert len(reader.site_ids("left")) == 135
    assert len(reader.site_ids("right")) == 135

    cube_bid = model.body("cube_body").id
    data.ctrl[0] = 0.25
    for step in range(800):
        # 稳定夹持后先施加切向推力，后段切换为绕闭合轴 X 的扭转力矩。
        if 300 <= step < 550:
            data.xfrc_applied[cube_bid, :] = [0.0, 0.5, 0.8, 0.0, 0.0, 0.0]
        elif step >= 550:
            data.xfrc_applied[cube_bid, :] = [0.0, 0.0, 0.0, 0.02, 0.0, 0.0]
        else:
            data.xfrc_applied[cube_bid, :] = 0.0
        mujoco.mj_step(model, data)

        if step in (549, 799):
            forces = reader.read(data)
            for side in ("left", "right"):
                arr = forces[side]
                assert arr is not None and arr.shape == (135, 3)
                assert np.all(arr[:, 2] >= -1e-12)
                # 相较于原生 <touch> 仅有角点 2 个测点非零，高斯核投影使周围阵列连续响应。
                assert np.count_nonzero(arr[:, 2] > 1e-3) >= 20
                expected_net, ncon = _raw_pad_net_force_local(
                    model, data, f"gripper/{side}_pad_collision", f"gripper/{side}_taxel_001"
                )
                assert ncon >= 1
                assert expected_net[2] > 0.5
                np.testing.assert_allclose(arr.sum(axis=0), expected_net, atol=1e-11)

            if step == 549:
                # 切向推力应在至少一侧（且两侧合力）体现明显切向剪切 [Fx, Fy]。
                left_shear = np.linalg.norm(forces["left"].sum(axis=0)[:2])
                right_shear = np.linalg.norm(forces["right"].sum(axis=0)[:2])
                assert left_shear + right_shear > 0.4

            if step == 799:
                # 绕接触法向施加扭矩时，分布式测点需重构出同号面内扭矩偶。
                for side in ("left", "right"):
                    arr = forces[side]
                    assert arr is not None
                    sids = np.array(reader.site_ids(side), dtype=np.intp)
                    center = np.asarray(
                        data.site_xpos[model.site(f"gripper/{side}_pad_center").id],
                        dtype=np.float64,
                    )
                    center_rot = np.asarray(
                        data.site_xmat[model.site(f"gripper/{side}_pad_center").id],
                        dtype=np.float64,
                    ).reshape(3, 3)
                    pos_local = (
                        np.asarray(data.site_xpos[sids], dtype=np.float64) - center
                    ) @ center_rot
                    # left/right site_z 指向垫背，施加绕世界 +X 扭矩时两侧力偶同向抵抗。
                    torque_z = float(
                        np.sum(pos_local[:, 0] * arr[:, 1] - pos_local[:, 1] * arr[:, 0])
                    )
                    assert abs(torque_z) > 1e-3


def test_paxini_spatial_localization_target_filter_and_read_grid():
    """验证点探针压力中心随触点移动、target_geoms 过滤有效，且 contact_geom read_grid 一致。"""
    scene = _attach_gripper_and_box(
        left_sensor="pts_2_0",
        right_sensor="paxini_m2824",
    )
    add_tactile_contact_pairs(
        scene,
        "cube/target_cube_geom",
        left_sensor="pts_2_0",
        right_sensor="paxini_m2824",
        prefix="gripper/",
    )
    model = scene.compile()
    data = mujoco.MjData(model)

    data.ctrl[0] = 0.25
    for _ in range(400):
        mujoco.mj_step(model, data)

    reader = TactileForceReader(
        model,
        left_sensor="pts_2_0",
        right_sensor="paxini_m2824",
        prefix="gripper/",
        target_geoms=["cube/target_cube_geom"],
    )
    forces = reader.read(data)
    grid = reader.read_grid(data, "left")
    assert grid.shape == (3, 3, 3)
    assert grid[2].sum() > 4.0
    assert forces["right"] is not None and forces["right"][:, 2].sum() > 4.0
    np.testing.assert_allclose(grid.transpose(1, 2, 0).reshape(9, 3), forces["left"], atol=1e-12)

    # 验证 contact_geom 逐球聚合与手动 mj_contactForce 精确一致。
    for r in range(3):
        for c in range(3):
            expected, _ = _raw_pad_net_force_local(
                model,
                data,
                f"gripper/left_taxel_geom_{r}{c}",
                f"gripper/left_taxel_{r}{c}",
                other_geom_id=model.geom("cube/target_cube_geom").id,
            )
            np.testing.assert_allclose(grid[:, r, c], expected, atol=1e-12)

    with pytest.raises(ValueError, match="规则行列网格"):
        reader.read_grid(data, "right")

    # 独立场景验证 paxini 点接触 CoP 随探针沿局部 X 移动（远端 vs 近端）。
    cop_x: list[float] = []
    for x_offset in (-0.008, 0.008):
        probe_scene = mujoco.MjSpec()
        probe_scene.option.gravity = [0.0, 0.0, 0.0]
        probe_scene.attach(
            build_gripper_spec(
                left_sensor="paxini_m2824", right_sensor=None, with_camera=False, fixed_base=True
            ),
            prefix="gripper/",
            frame=probe_scene.worldbody.add_frame(),
        )
        # 在 left_pad_center 平面沿 +X_mount / -X_mount 偏移，用半球探针压入左侧硅胶表面。
        probe = probe_scene.worldbody.add_body(
            name="probe_body", pos=[-0.036, 0.0, 0.14555 + x_offset]
        )
        probe.add_geom(
            name="probe_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.002, 0.0, 0.0],
            contype=0,
            conaffinity=2,
        )
        distractor = probe_scene.worldbody.add_body(
            name="distractor_body", pos=[-0.036, 0.005, 0.14555]
        )
        distractor.add_geom(
            name="distractor_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.002, 0.0, 0.0],
            contype=0,
            conaffinity=2,
        )
        add_tactile_contact_pairs(
            probe_scene,
            ["probe_geom", "distractor_geom"],
            left_sensor="paxini_m2824",
            right_sensor=None,
            prefix="gripper/",
        )
        pm = probe_scene.compile()
        pd = mujoco.MjData(pm)
        pd.ctrl[0] = 0.0
        for _ in range(150):
            mujoco.mj_step(pm, pd)

        filtered = TactileForceReader(
            pm,
            left_sensor="paxini_m2824",
            right_sensor=None,
            prefix="gripper/",
            target_geoms=["probe_geom"],
        )
        pf = filtered.read(pd)
        assert pf["right"] is None and pf["left"] is not None
        expected_probe_only, _ = _raw_pad_net_force_local(
            pm,
            pd,
            "gripper/left_pad_collision",
            "gripper/left_pad_center",
            other_geom_id=pm.geom("probe_geom").id,
        )
        np.testing.assert_allclose(pf["left"].sum(axis=0), expected_probe_only, atol=1e-12)
        sids = np.array(filtered.site_ids("left"), dtype=np.intp)
        fz = pf["left"][:, 2]
        assert fz.sum() > 0.1
        cop_world = np.sum(pd.site_xpos[sids] * fz[:, None], axis=0) / fz.sum()
        cop_x.append(float(cop_world[2] - 0.14555))

    assert cop_x[0] < -0.005
    assert cop_x[1] > 0.005


def test_tactile_helper_validation_errors():
    """验证非法参数、缺省名称与无触觉侧读取报错。"""
    scene = _attach_gripper_and_box(left_sensor="flat_pad", right_sensor=None)
    with pytest.raises(ValueError, match="condim"):
        add_tactile_contact_pairs(scene, "cube/target_cube_geom", condim=2)
    with pytest.raises(ValueError, match="friction_coefficient"):
        add_tactile_contact_pairs(scene, "cube/target_cube_geom", friction_coefficient=-0.1)
    with pytest.raises(ValueError, match="object_geom"):
        add_tactile_contact_pairs(scene, [])

    model = scene.compile()
    data = mujoco.MjData(model)
    with pytest.raises(ValueError, match="pad_sigma_m"):
        TactileForceReader(model, left_sensor="flat_pad", right_sensor=None, pad_sigma_m=0.0)
    with pytest.raises(ValueError, match="target_geoms"):
        TactileForceReader(
            model, left_sensor="flat_pad", right_sensor=None, target_geoms=[model.ngeom]
        )

    reader = TactileForceReader(model, left_sensor="flat_pad", right_sensor=None, prefix="gripper/")
    assert reader.read(data) == {"left": None, "right": None}
    assert reader.site_ids("left") == ()
    with pytest.raises(ValueError, match="规则行列网格"):
        reader.read_grid(data, "left")
