"""验证统一触觉接触对注册与 contact_geom / pad_sites 三轴力读取辅助函数。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, get_type_hints

import mujoco
import numpy as np
import pytest

from adaptive_tactile_gripper_description import (
    TactileForceReader,
    add_tactile_contact_pairs,
    build_gripper_spec,
    tactile,
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
    assert get_type_hints(TactileForceReader.__init__)["target_geoms"] == (
        str | int | np.integer[Any] | Sequence[str | int | np.integer[Any]] | None
    )


def _probe_pad(monkeypatch, xy, condim, *, pad_on_geom2=False):
    points = [(-0.004, -0.003), (0.004, 0.003), (0.0, 0.0), (0.004, -0.003), (-0.004, 0.003)]
    taxels = [{"id": i, "site_name": f"taxel_{i}"} for i in range(len(points))]
    # The probe overlaps the plane by 0.2 mm. All sites share the pad normal.
    quat = "1 0 0 0" if pad_on_geom2 else "0 1 0 0"
    sites = "".join(
        f'<site name="taxel_{i}" pos="{x} {y} 0" quat="{quat}"/>' for i, (x, y) in enumerate(points)
    )
    plane_name, sphere_name = ("probe_geom", "pad") if pad_on_geom2 else ("pad", "probe_geom")
    model = mujoco.MjModel.from_xml_string(f"""
      <mujoco>
        <option gravity="0 0 0" cone="elliptic"/>
        <worldbody>
          <geom name="{plane_name}" type="plane" size=".03 .03 .001"
                condim="{condim}" friction=".8 .02 .01"/>
          {"" if pad_on_geom2 else sites}
          <body pos="{xy[0]} {xy[1]} .0018">
            <freejoint/>
            <geom name="{sphere_name}" type="sphere" size=".002" mass=".05"/>
            {sites if pad_on_geom2 else ""}
          </body>
        </worldbody>
      </mujoco>
    """)
    monkeypatch.setattr(
        tactile,
        "get_tactile_layout",
        lambda *args, **kwargs: {
            "left": {"mode": "pad_sites", "pad_geom_name": "pad", "taxels": taxels},
            "right": None,
        },
    )
    return model, mujoco.MjData(model)


@pytest.mark.parametrize(("condim", "pad_on_geom2"), [(4, False), (4, True), (6, False)])
def test_pad_reconstructs_contact_twist_without_changing_net_force(
    monkeypatch, condim, pad_on_geom2
):
    model, data = _probe_pad(monkeypatch, (0.0013, -0.0007), condim, pad_on_geom2=pad_on_geom2)
    spin = np.array([2.0, 1.0, 5.0])
    # Cancel the sliding velocity of the spinning probe at the contact.
    data.qvel[:3] = -np.cross(spin, [0.0, 0.0, -0.0019])
    data.qvel[3:] = spin
    mujoco.mj_forward(model, data)
    assert data.ncon == 1
    contact = data.contact[0]
    assert contact.dim == condim
    assert contact.geom[int(pad_on_geom2)] == model.geom("pad").id
    sign = 1 if pad_on_geom2 else -1
    wrench = np.zeros(6)
    mujoco.mj_contactForce(model, data, 0, wrench)
    assert abs(wrench[3]) > 1e-6
    if condim == 6:
        assert np.linalg.norm(wrench[4:]) > 1e-6
    contact_frame = contact.frame.reshape(3, 3)
    raw_force = sign * contact_frame.T @ wrench[:3]
    raw_twist = sign * contact_frame[0] * wrench[3]

    reader = TactileForceReader(model, right_sensor=None, target_geoms="probe_geom")
    forces = reader.read(data)["left"]
    assert forces is not None
    assert np.all(forces[:, 2] >= 0.0)
    positions = data.site_xpos[list(reader.site_ids("left"))]
    rotation = data.site_xmat[reader.site_ids("left")[0]].reshape(3, 3)
    world_forces = forces @ rotation.T
    np.testing.assert_allclose(world_forces.sum(axis=0), raw_force, atol=1e-12)
    # A matching centroid gives the distributed contact force zero moment
    # about the projected contact, even when rolling induces shear.
    origin = np.array([*contact.pos[:2], positions[0, 2]])
    moment = np.cross(positions - origin, world_forces).sum(axis=0)
    np.testing.assert_allclose(moment, raw_twist, atol=1e-12)

    narrow_reader = TactileForceReader(model, right_sensor=None, pad_sigma_m=1e-200)
    narrow_forces = narrow_reader.read(data)["left"]
    assert narrow_forces is not None
    narrow_world = narrow_forces @ rotation.T
    np.testing.assert_allclose(narrow_world.sum(axis=0), raw_force, atol=1e-12)
    np.testing.assert_allclose(
        np.cross(positions - origin, narrow_world).sum(axis=0), 0.0, atol=1e-12
    )


@pytest.mark.parametrize(
    ("probe_xy", "expected_xy"),
    [
        ((0.0013, -0.0007), (0.0013, -0.0007)),
        ((0.0039999, -0.0029999), (0.0039999, -0.0029999)),
        ((0.004, 0.003), (0.004, 0.003)),
        ((0.005, 0.004), (0.004, 0.003)),
        ((0.005, 0.001), (0.004, 0.001)),
    ],
)
def test_pad_pressure_center_matches_interior_and_clamps_to_hull(
    monkeypatch, probe_xy, expected_xy
):
    model, data = _probe_pad(monkeypatch, probe_xy, 3)
    mujoco.mj_forward(model, data)
    assert data.ncon == 1
    reader = TactileForceReader(model, right_sensor=None)
    forces = reader.read(data)["left"]
    assert forces is not None
    fz = forces[:, 2]
    assert np.all(fz >= 0)
    assert fz.sum() > 0
    sites = data.site_xpos[list(reader.site_ids("left"))]
    np.testing.assert_allclose(fz @ sites[:, :2] / fz.sum(), expected_xy, atol=1e-12)
    wrench = np.zeros(6)
    mujoco.mj_contactForce(model, data, 0, wrench)
    np.testing.assert_allclose(forces.sum(axis=0), [0, 0, wrench[0]], atol=1e-12)


@pytest.fixture
def shuffled_grid(monkeypatch):
    scene = mujoco.MjSpec.from_string("""
      <mujoco>
        <option gravity="0 0 0"/>
        <worldbody>
          <geom name="probe_geom" type="plane" size=".03 .03 .001"/>
        </worldbody>
      </mujoco>
    """)
    cells = [(1, 2), (0, 1), (1, 0), (0, 2), (0, 0), (1, 1)]
    taxels = []
    for i, (row, col) in enumerate(cells):
        body = scene.worldbody.add_body(pos=[0.006 * col, 0.006 * row, 0.0019 - i * 0.00005])
        body.add_freejoint()
        geom = body.add_geom(
            name=f"geom_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.002, 0, 0], mass=0.01
        )
        site = body.add_site(name=f"site_{i}")
        taxels.append({"row": row, "col": col, "geom_name": geom.name, "site_name": site.name})
    monkeypatch.setattr(
        tactile,
        "get_tactile_layout",
        lambda *args, **kwargs: {
            "left": {"mode": "contact_geom", "rows": 2, "cols": 3, "taxels": taxels},
            "right": None,
        },
    )
    return scene.compile(), taxels


def test_read_grid_uses_declared_coordinates(shuffled_grid):
    model, taxels = shuffled_grid
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    reader = TactileForceReader(model, right_sensor=None)
    forces = reader.read(data)["left"]
    assert forces is not None
    grid = reader.read_grid(data, "left")
    assert grid.shape == (3, 2, 3)
    assert np.all(grid[2] > 0)
    assert np.unique(grid[2]).size == 6
    for idx, taxel in enumerate(taxels):
        expected, count = _raw_pad_net_force_local(
            model, data, taxel["geom_name"], taxel["site_name"]
        )
        assert count == 1
        np.testing.assert_allclose(forces[idx], expected, atol=1e-12)
        np.testing.assert_allclose(grid[:, taxel["row"], taxel["col"]], expected, atol=1e-12)


@pytest.mark.parametrize(
    "coords",
    [None, (0, 1), (-1, 0), (2, 0), (0, -1), (1, 3), (1, 2.0), (True, 2)],
)
def test_read_grid_rejects_invalid_coverage(shuffled_grid, coords):
    model, taxels = shuffled_grid
    if coords is None:
        taxels.pop()
    else:
        taxels[0]["row"], taxels[0]["col"] = coords
    with pytest.raises(ValueError, match="row"):
        TactileForceReader(model, right_sensor=None)


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
            target_geoms="probe_geom",
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
        probe_gid = pm.geom("probe_geom").id
        for target in (
            probe_gid,
            np.int64(probe_gid),
            np.int32(probe_gid),
            ["probe_geom", np.int64(probe_gid)],
        ):
            equivalent = TactileForceReader(
                pm,
                left_sensor="paxini_m2824",
                right_sensor=None,
                prefix="gripper/",
                target_geoms=target,
            )
            np.testing.assert_array_equal(equivalent.read(pd)["left"], pf["left"])
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
    for bad_target in (True, [False]):
        with pytest.raises(TypeError, match="target_geoms"):
            TactileForceReader(
                model, left_sensor="flat_pad", right_sensor=None, target_geoms=bad_target
            )

    reader = TactileForceReader(model, left_sensor="flat_pad", right_sensor=None, prefix="gripper/")
    assert reader.read(data) == {"left": None, "right": None}
    assert reader.site_ids("left") == ()
    with pytest.raises(ValueError, match="规则行列网格"):
        reader.read_grid(data, "left")


def test_rigid_pad_reuses_cached_hull_across_reads(monkeypatch):
    """验证刚性平垫在 __init__ 预计算凸包，read() 逐帧调用不再重复构建凸包。"""
    scene = _attach_gripper_and_box(left_sensor="paxini_m2824", right_sensor="paxini_m2824")
    add_tactile_contact_pairs(
        scene,
        "cube/target_cube_geom",
        left_sensor="paxini_m2824",
        right_sensor="paxini_m2824",
        prefix="gripper/",
    )
    model = scene.compile()
    data = mujoco.MjData(model)
    reader = TactileForceReader(
        model,
        left_sensor="paxini_m2824",
        right_sensor="paxini_m2824",
        prefix="gripper/",
    )

    def _fail_if_rebuilt(_xy):
        raise AssertionError("刚性平垫的凸包不应在 read() 期间重复计算")

    monkeypatch.setattr(tactile, "_convex_hull_indices", _fail_if_rebuilt)
    data.ctrl[0] = 0.25
    for _ in range(400):
        mujoco.mj_step(model, data)
    forces = reader.read(data)
    for side in ("left", "right"):
        arr = forces[side]
        assert arr is not None and arr.shape == (135, 3)
        expected, ncon = _raw_pad_net_force_local(
            model, data, f"gripper/{side}_pad_collision", f"gripper/{side}_taxel_001"
        )
        assert ncon >= 1 and expected[2] > 1.0
        np.testing.assert_allclose(arr.sum(axis=0), expected, atol=1e-11)


def test_rigid_pad_refreshes_cache_when_site_transforms_change():
    """验证标定修改 model.site_pos / model.site_quat 后，read() 自动刷新缓存并保持合力与符号一致。"""
    scene = _attach_gripper_and_box(left_sensor="paxini_m2824", right_sensor="paxini_m2824")
    add_tactile_contact_pairs(
        scene,
        "cube/target_cube_geom",
        left_sensor="paxini_m2824",
        right_sensor="paxini_m2824",
        prefix="gripper/",
    )
    model = scene.compile()
    data = mujoco.MjData(model)
    reader = TactileForceReader(
        model,
        left_sensor="paxini_m2824",
        right_sensor="paxini_m2824",
        prefix="gripper/",
    )
    data.ctrl[0] = 0.25
    for _ in range(400):
        mujoco.mj_step(model, data)

    before = reader.read(data)["left"]
    assert before is not None
    peak_idx = int(np.argmax(before[1:, 2])) + 1
    assert before[peak_idx, 2] > 1e-3

    # 同时翻转首个测点（taxel_001）与受压峰值测点的局部坐标系，不应误屏蔽前感应面接触。
    sid0 = reader.site_ids("left")[0]
    peak_sid = reader.site_ids("left")[peak_idx]
    model.site_quat[sid0] = [1.0, 0.0, 0.0, 0.0]
    model.site_quat[peak_sid] = [1.0, 0.0, 0.0, 0.0]
    model.site_pos[peak_sid, 0] += 0.0005
    mujoco.mj_forward(model, data)

    after = reader.read(data)["left"]
    assert after is not None
    # 翻转该测点法向后，其局部 Fz 符号须反转为负，未翻转测点保持正值，世界合力与 MuJoCo 严格一致。
    assert after[peak_idx, 2] < -1e-3
    unflipped = np.ones(135, dtype=bool)
    unflipped[[0, peak_idx]] = False
    assert np.all(after[unflipped, 2] >= -1e-12)
    assert after[unflipped, 2].sum() > 1.0
    sids = np.array(reader.site_ids("left"), dtype=np.intp)
    world_sum = np.einsum(
        "nij,nj->ni", np.asarray(data.site_xmat[sids]).reshape(-1, 3, 3), after
    ).sum(axis=0)
    pad_rot = np.asarray(
        data.site_xmat[model.site("gripper/left_pad_center").id], dtype=np.float64
    ).reshape(3, 3)
    expected_local, _ = _raw_pad_net_force_local(
        model, data, "gripper/left_pad_collision", "gripper/left_pad_center"
    )
    np.testing.assert_allclose(pad_rot.T @ world_sum, expected_local, atol=1e-11)


@pytest.mark.parametrize(
    ("xy_mount", "sigma_m"),
    [
        ((-0.013, 0.009), 0.0002),
        ((0.0143082, 0.001), 1e-200),
        ((0.015, 0.0), 1e-200),
    ],
)
def test_paxini_small_sigma_keeps_finite_exact_cop(xy_mount, sigma_m):
    """验证极小 pad_sigma_m（含有限下溢与边界外全 -inf 先验）仍返回有限、归一化且精确压力中心的读数。"""
    scene = mujoco.MjSpec()
    scene.option.gravity = [0.0, 0.0, 0.0]
    scene.attach(
        build_gripper_spec(
            left_sensor="paxini_m2824", right_sensor=None, with_camera=False, fixed_base=True
        ),
        prefix="gripper/",
        frame=scene.worldbody.add_frame(),
    )
    ref_model = scene.compile()
    ref_data = mujoco.MjData(ref_model)
    mujoco.mj_forward(ref_model, ref_data)
    mount_pos = ref_data.body("gripper/left_sensor_body").xpos.copy()
    mount_rot = ref_data.body("gripper/left_sensor_body").xmat.reshape(3, 3)
    target_mount = np.array([xy_mount[0], xy_mount[1], 0.0037])
    probe_pos = mount_pos + mount_rot @ (target_mount + np.array([0.0, 0.0, 0.0018]))
    probe = scene.worldbody.add_body(name="small_sigma_probe", pos=probe_pos.tolist())
    probe.add_geom(
        name="small_sigma_probe_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.002, 0.0, 0.0],
        contype=0,
        conaffinity=2,
    )
    add_tactile_contact_pairs(
        scene,
        "small_sigma_probe_geom",
        left_sensor="paxini_m2824",
        right_sensor=None,
        prefix="gripper/",
    )
    model = scene.compile()
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)

    reader = TactileForceReader(
        model,
        left_sensor="paxini_m2824",
        right_sensor=None,
        prefix="gripper/",
        pad_sigma_m=sigma_m,
        target_geoms="small_sigma_probe_geom",
    )
    forces = reader.read(data)["left"]
    assert forces is not None
    assert np.all(np.isfinite(forces))
    fz = forces[:, 2]
    assert np.all(fz >= -1e-12)
    assert fz.sum() > 0.05
    expected_local, _ = _raw_pad_net_force_local(
        model,
        data,
        "gripper/left_pad_collision",
        "gripper/left_pad_center",
        other_geom_id=model.geom("small_sigma_probe_geom").id,
    )
    np.testing.assert_allclose(forces.sum(axis=0), expected_local, atol=1e-11)
    sids = np.array(reader.site_ids("left"), dtype=np.intp)
    cop_world = np.sum(data.site_xpos[sids] * fz[:, None], axis=0) / fz.sum()
    cop_mount = (cop_world - data.body("gripper/left_sensor_body").xpos) @ data.body(
        "gripper/left_sensor_body"
    ).xmat.reshape(3, 3)
    if xy_mount[0] <= 0.0143082:
        np.testing.assert_allclose(cop_mount[:2], xy_mount, atol=5e-5)
    else:
        assert cop_mount[0] == pytest.approx(0.0143082, abs=5e-5)
        assert abs(cop_mount[1] - xy_mount[1]) < 5e-5


@pytest.mark.parametrize("taxel_id", [39, 78])
def test_paxini_edge_taxel_receives_local_load(taxel_id):
    """验证压在共线及微内缩边缘测点（taxel 39 / 78）正上方时，受力集中在当地测点且压力中心一致。"""
    probe_scene = mujoco.MjSpec()
    probe_scene.option.gravity = [0.0, 0.0, 0.0]
    probe_scene.attach(
        build_gripper_spec(
            left_sensor="paxini_m2824", right_sensor=None, with_camera=False, fixed_base=True
        ),
        prefix="gripper/",
        frame=probe_scene.worldbody.add_frame(),
    )
    ref_model = probe_scene.compile()
    ref_data = mujoco.MjData(ref_model)
    mujoco.mj_forward(ref_model, ref_data)
    site_name = f"gripper/left_taxel_{taxel_id:03d}"
    t_pos = ref_data.site(site_name).xpos.copy()
    t_rot = ref_data.site(site_name).xmat.reshape(3, 3)
    probe_pos = t_pos - 0.0018 * t_rot[:, 2]
    probe = probe_scene.worldbody.add_body(name="edge_probe", pos=probe_pos.tolist())
    probe.add_geom(
        name="edge_probe_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.002, 0.0, 0.0],
        contype=0,
        conaffinity=2,
    )
    add_tactile_contact_pairs(
        probe_scene,
        "edge_probe_geom",
        left_sensor="paxini_m2824",
        right_sensor=None,
        prefix="gripper/",
    )
    model = probe_scene.compile()
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)

    reader = TactileForceReader(
        model,
        left_sensor="paxini_m2824",
        right_sensor=None,
        prefix="gripper/",
        target_geoms="edge_probe_geom",
    )
    forces = reader.read(data)["left"]
    assert forces is not None
    fz = forces[:, 2]
    assert fz.sum() > 0.05
    idx = next(i for i, t in enumerate(reader.layout["left"]["taxels"]) if t["id"] == taxel_id)
    assert int(np.argmax(fz)) == idx
    assert fz[idx] > 0.25 * fz.sum()
    assert float(model.site_size[model.site(site_name).id, 0]) < 0.0011
    sids = np.array(reader.site_ids("left"), dtype=np.intp)
    cop_world = np.sum(data.site_xpos[sids] * fz[:, None], axis=0) / fz.sum()
    np.testing.assert_allclose(cop_world, data.site(site_name).xpos, atol=5e-5)


@pytest.mark.parametrize("contact_side", ["rear_lip", "side_wall"])
def test_paxini_excludes_side_and_rear_pad_contacts(contact_side):
    """验证作用于 pad_collision 背面裸露唇边或侧壁的接触不会误报到前感应面阵列。"""
    scene = mujoco.MjSpec()
    scene.option.gravity = [0.0, 0.0, 0.0]
    scene.attach(
        build_gripper_spec(
            left_sensor="paxini_m2824", right_sensor=None, with_camera=False, fixed_base=True
        ),
        prefix="gripper/",
        frame=scene.worldbody.add_frame(),
    )
    ref_model = scene.compile()
    ref_data = mujoco.MjData(ref_model)
    mujoco.mj_forward(ref_model, ref_data)
    pad_gid = ref_model.geom("gripper/left_pad_collision").id
    pad_pos = ref_data.geom_xpos[pad_gid].copy()
    pad_rot = ref_data.geom_xmat[pad_gid].reshape(3, 3)
    half = ref_model.geom_size[pad_gid]
    if contact_side == "rear_lip":
        local_offset = np.array([0.0125, 0.0, -(half[2] + 0.0016)])
    else:
        local_offset = np.array([0.0, half[1] + 0.0016, 0.0])
    probe = scene.worldbody.add_body(
        name="side_rear_probe", pos=(pad_pos + pad_rot @ local_offset).tolist()
    )
    probe.add_geom(
        name="side_rear_probe_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.002, 0.0, 0.0],
        contype=0,
        conaffinity=2,
    )
    add_tactile_contact_pairs(
        scene,
        "side_rear_probe_geom",
        left_sensor="paxini_m2824",
        right_sensor=None,
        prefix="gripper/",
    )
    model = scene.compile()
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)

    probe_gid = model.geom("side_rear_probe_geom").id
    assert any(probe_gid in (c.geom1, c.geom2) for c in data.contact)
    reader = TactileForceReader(
        model,
        left_sensor="paxini_m2824",
        right_sensor=None,
        prefix="gripper/",
        target_geoms="side_rear_probe_geom",
    )
    forces = reader.read(data)["left"]
    assert forces is not None
    np.testing.assert_array_equal(forces, np.zeros((135, 3)))


def test_paxini_deep_front_penetration_keeps_reading_load():
    """验证物体前表面深度压入超过单倍垫厚（> 6 mm）时，仍按垫面接触点判定为前感应面接触。"""
    scene = mujoco.MjSpec()
    scene.option.gravity = [0.0, 0.0, 0.0]
    scene.attach(
        build_gripper_spec(
            left_sensor="paxini_m2824", right_sensor=None, with_camera=False, fixed_base=True
        ),
        prefix="gripper/",
        frame=scene.worldbody.add_frame(),
    )
    ref_model = scene.compile()
    ref_data = mujoco.MjData(ref_model)
    mujoco.mj_forward(ref_model, ref_data)
    face_pos = ref_data.site("gripper/left_pad_center").xpos.copy()
    face_rot = ref_data.site("gripper/left_pad_center").xmat.reshape(3, 3)
    # 方块半厚 15 mm，中心沿局部 -Z（垫前外侧）仅放 8.5 mm，使前表面深度嵌入约 6.5 mm（> 垫厚 6.0 mm）。
    box_pos = face_pos - 0.0085 * face_rot[:, 2]
    box = scene.worldbody.add_body(name="deep_box", pos=box_pos.tolist())
    box.add_geom(
        name="deep_box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.015, 0.008, 0.008],
        contype=0,
        conaffinity=2,
    )
    add_tactile_contact_pairs(
        scene,
        "deep_box_geom",
        left_sensor="paxini_m2824",
        right_sensor=None,
        prefix="gripper/",
    )
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # 确认接触穿透深度大于整块垫厚（6.0 mm），使接触中点落到垫块中心之后。
    box_gid = model.geom("deep_box_geom").id
    dists = [c.dist for c in data.contact if box_gid in (c.geom1, c.geom2)]
    assert dists and min(dists) < -0.0061

    reader = TactileForceReader(
        model,
        left_sensor="paxini_m2824",
        right_sensor=None,
        prefix="gripper/",
        target_geoms="deep_box_geom",
    )
    forces = reader.read(data)["left"]
    assert forces is not None
    expected_local, _ = _raw_pad_net_force_local(
        model,
        data,
        "gripper/left_pad_collision",
        "gripper/left_pad_center",
        other_geom_id=box_gid,
    )
    assert expected_local[2] > 1.0
    np.testing.assert_allclose(forces.sum(axis=0), expected_local, atol=1e-11)
