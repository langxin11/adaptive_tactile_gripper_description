"""验证 MuJoCo、Jazzy Xacro 与下游消费接口的一致性。"""

from __future__ import annotations

import hashlib
import math
import os
import runpy
import shutil
import subprocess
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path

import mujoco
import numpy as np
import pytest

from adaptive_tactile_gripper_description import (
    build_gripper_spec,
    get_closure_jacobian,
    get_data_root,
    get_joint_positions,
    get_mjcf_path,
    get_specs,
    get_stroke_limits,
    get_tactile_layout,
)
from adaptive_tactile_gripper_description.kinematics import drive_angle_for_closure

NS = "gripper"
SIDES = ("left", "right")
MODULES = ("pts_2_0", "flat_pad", "none")
CONFIGURATIONS = list(product(MODULES, MODULES, (False, True)))


@pytest.fixture(scope="module")
def expand_xacro(tmp_path_factory):
    """用临时 ament 索引展开源码，结果不依赖系统中同名包的安装版本。"""
    jazzy = Path("/opt/ros/jazzy")
    xacro = jazzy / "bin/xacro"
    if not xacro.is_file():
        executable = shutil.which("xacro")
        if executable is None:
            pytest.skip("未找到 Jazzy Xacro 或 PATH 中的 xacro")
        xacro = Path(executable)
    package = "adaptive_tactile_gripper_description"
    prefix = tmp_path_factory.mktemp("ament")
    index = prefix / "share/ament_index/resource_index/packages"
    index.mkdir(parents=True)
    (index / package).touch()
    root = get_data_root()
    (prefix / "share" / package).symlink_to(root, target_is_directory=True)

    srdf_wrapper = prefix / "test.srdf.xacro"
    srdf_wrapper.write_text("""<robot name="adaptive_tactile_gripper" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:arg name="left_sensor" default="pts_2_0"/>
  <xacro:arg name="right_sensor" default="pts_2_0"/>
  <xacro:arg name="with_camera" default="true"/>
  <xacro:arg name="use_nominal_extrinsics" default="true"/>
  <xacro:include filename="$(find adaptive_tactile_gripper_description)/srdf/gripper.srdf.xacro"/>
  <xacro:adaptive_tactile_gripper_srdf ns="gripper" left_sensor="$(arg left_sensor)"
    right_sensor="$(arg right_sensor)" with_camera="$(arg with_camera)"
    use_nominal_extrinsics="$(arg use_nominal_extrinsics)"/>
</robot>""", encoding="utf-8")
    env = os.environ.copy()
    env["AMENT_PREFIX_PATH"] = os.pathsep.join(filter(None, (str(prefix), env.get("AMENT_PREFIX_PATH"))))
    if xacro == jazzy / "bin/xacro":
        jazzy_python = jazzy / "lib/python3.12/site-packages"
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(jazzy_python), env.get("PYTHONPATH"))))

    def expand(*, srdf=False, **args):
        path = srdf_wrapper if srdf else root / "urdf/gripper_standalone.urdf.xacro"
        cmd = [str(xacro), str(path), f"ns:={NS}", *(f"{key}:={value}" for key, value in args.items())]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        return ET.fromstring(result.stdout)

    return expand


def _quat_matrix(quat):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(quat))
    return result.reshape(3, 3)


def _homogeneous(position, rotation):
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation).reshape(3, 3)
    result[:3, 3] = position
    return result


def _origin(element):
    origin = element.find("origin")
    if origin is None:
        return np.eye(4)
    roll, pitch, yaw = map(float, origin.get("rpy").split())
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    rotation = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return _homogeneous(list(map(float, origin.get("xyz").split())), rotation)


def _urdf_poses(urdf, positions):
    poses = {f"{NS}_base": np.eye(4)}
    joints = {joint.get("name"): joint for joint in urdf.findall("joint")}
    reciprocals = {joint.find("child").get("link"): name for name, joint in joints.items()}

    def joint_position(name):
        mimic = joints[name].find("mimic")
        if mimic is not None:
            return joint_position(mimic.get("joint")) * float(mimic.get("multiplier")) + float(mimic.get("offset"))
        return positions.get(name.removeprefix(f"{NS}_"), 0.0)

    def link_pose(name):
        if name not in poses:
            joint_name = reciprocals[name]
            joint = joints[joint_name]
            transform = link_pose(joint.find("parent").get("link")) @ _origin(joint)
            kind = joint.get("type")
            if kind in ("prismatic", "revolute"):
                q = joint_position(joint_name)
                axis = np.array([float(value) for value in joint.find("axis").get("xyz").split()])
                if kind == "prismatic":
                    transform[:3, 3] += transform[:3, :3] @ (axis * q)
                else:
                    rotation = _quat_matrix(np.r_[math.cos(q / 2), axis * math.sin(q / 2)])
                    transform[:3, :3] = transform[:3, :3] @ rotation
            poses[name] = transform
        return poses[name]

    return {link.get("name"): link_pose(link.get("name")) for link in urdf.findall("link")}


def _forward(model, q):
    data = mujoco.MjData(model)
    for name, pos in get_joint_positions(q).items():
        data.joint(name).qpos[0] = pos
    mujoco.mj_forward(model, data)
    return data


def _composite(parts):
    mass = sum(m for m, _, _ in parts)
    com = sum(m * pos for m, pos, _ in parts) / mass
    inertia = np.zeros((3, 3))
    for m, pos, tensor in parts:
        delta = pos - com
        inertia += tensor + m * (delta @ delta * np.eye(3) - np.outer(delta, delta))
    return mass, com, inertia


@pytest.mark.parametrize("closure", [0.0, 0.02])
def test_mjcf_urdf_world_poses_and_inertias(expand_xacro, closure):
    """逐体核对网格、碰撞体、参考帧、球心和合成质量属性。"""
    urdf = expand_xacro()
    q = drive_angle_for_closure(closure)
    poses = _urdf_poses(urdf, get_joint_positions(q))
    model = build_gripper_spec(fixed_base=True).compile()
    data = _forward(model, q)
    prebuilt = mujoco.MjModel.from_xml_path(str(get_mjcf_path()))
    prebuilt_data = _forward(prebuilt, q)
    mjcf_names = {
        "base": "base", "motor": "motor",
        "left_rod": "rod_bearing_shim_subassembly", "right_rod": "rod_bearing_shim_subassembly_2",
        "left_finger": "left_finger", "right_finger": "right_finger",
        "left_sensor_link": "left_sensor_body", "right_sensor_link": "right_sensor_body",
        "camera_mount_link": "cam_camera_assembly",
    }
    mjcf_parts, urdf_parts = [], []
    for name, mjname in mjcf_names.items():
        link = urdf.find(f"link[@name='{NS}_{name}']")
        link_pose = poses[link.get("name")]
        body_id = model.body(mjname).id
        body = data.body(body_id)
        np.testing.assert_allclose(link_pose, _homogeneous(body.xpos, body.xmat), atol=1e-8)

        element = link.find("inertial")
        mass = float(element.find("mass").get("value"))
        urdf_inertial = link_pose @ _origin(element)
        inertia = element.find("inertia")
        urdf_tensor = np.array([[float(inertia.get(f"i{min(a,b)}{max(a,b)}")) for b in "xyz"] for a in "xyz"])
        rotation = urdf_inertial[:3, :3]
        urdf_tensor = rotation @ urdf_tensor @ rotation.T
        rotation = body.ximat.reshape(3, 3)
        mjcf_tensor = rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
        assert mass == pytest.approx(model.body_mass[body_id], abs=1e-12)
        np.testing.assert_allclose(urdf_inertial[:3, 3], body.xipos, atol=1e-8)
        np.testing.assert_allclose(urdf_tensor, mjcf_tensor, atol=1e-10)
        prebuilt_body = prebuilt_data.body(mjname)
        np.testing.assert_allclose(_homogeneous(prebuilt_body.xpos, prebuilt_body.xmat), link_pose, atol=2e-6)
        np.testing.assert_allclose(prebuilt_body.xipos, body.xipos, atol=2e-7)
        assert mass == pytest.approx(prebuilt.body(mjname).mass[0], rel=2e-6)
        prebuilt_rotation = prebuilt_body.ximat.reshape(3, 3)
        prebuilt_tensor = prebuilt_rotation @ np.diag(prebuilt.body(mjname).inertia) @ prebuilt_rotation.T
        np.testing.assert_allclose(prebuilt_tensor, urdf_tensor, atol=2e-9)
        urdf_parts.append((mass, urdf_inertial[:3, 3], urdf_tensor))
        mjcf_parts.append((model.body_mass[body_id], body.xipos, mjcf_tensor))

        geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        assert len(geom_ids) == len(link.findall("visual")) + len(link.findall("collision"))
        for geom_id, element in zip(geom_ids, link.findall("visual") + link.findall("collision"), strict=True):
            geom = data.geom(int(geom_id))
            position, rotation = geom.xpos, geom.xmat.reshape(3, 3)
            shape = element.find("geometry")[0]
            dims = model.geom_size[geom_id]
            kind = model.geom_type[geom_id]
            if shape.tag == "mesh":
                assert kind == mujoco.mjtGeom.mjGEOM_MESH
                mesh_id = model.geom_dataid[geom_id]
                rotation = rotation @ _quat_matrix(model.mesh_quat[mesh_id]).T
                position = position - rotation @ model.mesh_pos[mesh_id]
                assert model.mesh(mesh_id).name.endswith(Path(shape.get("filename")).stem)
            elif shape.tag == "box":
                assert kind == mujoco.mjtGeom.mjGEOM_BOX
                np.testing.assert_allclose(2 * dims, list(map(float, shape.get("size").split())), atol=1e-9)
            elif shape.tag == "cylinder":
                assert kind == mujoco.mjtGeom.mjGEOM_CYLINDER
                np.testing.assert_allclose([dims[0], 2 * dims[1]], [float(shape.get("radius")), float(shape.get("length"))])
            else:
                assert shape.tag == "sphere" and kind == mujoco.mjtGeom.mjGEOM_SPHERE
                assert dims[0] == pytest.approx(float(shape.get("radius")))
            world_pose = link_pose @ _origin(element)
            np.testing.assert_allclose(world_pose[:3, 3], position, atol=2e-7)
            tolerance = 1e-5 if shape.tag == "mesh" else 2e-8
            np.testing.assert_allclose(world_pose[:3, :3], rotation, atol=tolerance)
            if shape.tag == "sphere":
                np.testing.assert_allclose(prebuilt_data.geom(model.geom(int(geom_id)).name).xpos, position, atol=2e-7)

    assert len(mjcf_parts) == model.nbody - 1
    urdf_mass, urdf_com, urdf_inertia = _composite(urdf_parts)
    mjcf_mass, mjcf_com, mjcf_inertia = _composite(mjcf_parts)
    assert urdf_mass == pytest.approx(mjcf_mass, abs=1e-12)
    np.testing.assert_allclose(urdf_com, mjcf_com, atol=1e-8)
    np.testing.assert_allclose(urdf_inertia, mjcf_inertia, atol=1e-10)

    assert len(urdf.findall(".//collision/geometry/sphere")) == 18
    np.testing.assert_allclose(poses[f"{NS}_right_sensor_mount"][:3, 3], [0.03975 - closure, 0.0, 0.14555], atol=1e-10)
    np.testing.assert_allclose(poses[f"{NS}_tcp"], _homogeneous(data.site("tcp").xpos, data.site("tcp").xmat), atol=1e-8)
    for side in SIDES:
        tip = data.site(f"{side}_fingertip")
        np.testing.assert_allclose(poses[f"{NS}_{side}_fingertip"], _homogeneous(tip.xpos, tip.xmat), atol=1e-8)
    for site in (model.site(i) for i in range(model.nsite)):
        if site.name.startswith("cam_d435i_"):
            frame = data.site(site.id)
            np.testing.assert_allclose(
                poses[f"{NS}_{site.name.removeprefix('cam_')}"],
                _homogeneous(frame.xpos, frame.xmat), atol=1e-8,
            )
    if closure == 0:
        motor = urdf.find(f"link[@name='{NS}_motor']")
        visual = next(v for v in motor.findall("visual") if v.find("geometry/mesh").get("filename").endswith("/motor.stl"))
        assert (poses[motor.get("name")] @ _origin(visual))[2, 3] == pytest.approx(0.012, abs=1e-12)


def test_d435i_inertia_long_axis():
    """从生成的模组张量分离相机本体，检查长轴主惯量和包围盒上界。"""
    root = get_data_root()
    builder_path = root / "tools/build_description.py"
    if not builder_path.is_file():
        pytest.skip("缺少用于分离相机支架惯量的源码构建脚本")
    builder = runpy.run_path(str(builder_path))
    specs = get_specs()
    cam = specs["camera"]
    module = mujoco.MjModel.from_xml_path(str(root / "mjcf/accessories/realsense_d435i.xml"))
    solid = module.mesh("d435i_solid")

    # STL 为二进制三角网格，顶点包围盒用于计算质量分布方差的物理上界。
    stl = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("flags", "<u2")])
    vertices = np.fromfile(root / "meshes/d435i_solid.stl", dtype=stl, offset=84)["vertices"].reshape(-1, 3)
    extents = np.ptp(vertices, axis=0)
    assert np.argmax(extents) == 0
    upper_bound = cam["bare_camera_mass_kg"] * (extents[1] ** 2 + extents[2] ** 2) / 4
    assert upper_bound < 2.255e-5

    # MJCF 网格编译会记录原网格到重心网格的变换。
    visual_id = int(np.flatnonzero(module.geom_dataid == solid.id)[0])
    stl_rotation = _quat_matrix(module.geom_quat[visual_id]) @ _quat_matrix(module.mesh_quat[solid.id]).T
    stl_pos = module.geom_pos[visual_id] - stl_rotation @ module.mesh_pos[solid.id]
    bare_com = stl_pos + stl_rotation @ cam["bare_com_in_stl_m"]

    body = module.body("camera_assembly")
    module_rotation = _quat_matrix(body.iquat)
    module_tensor = module_rotation @ np.diag(body.inertia) @ module_rotation.T
    mount_mesh = ET.parse(root / "mjcf/accessories/realsense_d435i.xml").find(".//geom[@mesh='camera_mount']")
    density = specs["materials_and_mass_budget"]["effective_placf_density_kg_m3"]
    mount = builder["_part_from_mesh"](
        "camera_mount", root / "meshes/camera_mount.stl",
        list(map(float, mount_mesh.get("pos").split())), list(map(float, mount_mesh.get("quat").split())),
        density_kg_m3=density,
    )
    fasteners = builder["_point_part"]("camera_fasteners", builder["CAMERA_FASTENERS_MASS_KG"], [0, -0.064, 0.002])
    parts = (
        (cam["bare_camera_mass_kg"], bare_com),
        (mount.mass_kg, mount.com_m),
        (fasteners.mass_kg, fasteners.com_m),
    )
    for mass, com in parts:
        delta = com - body.ipos
        module_tensor -= mass * (delta @ delta * np.eye(3) - np.outer(delta, delta))
    bare_tensor = stl_rotation.T @ (module_tensor - mount.inertia_com_kg_m2 - fasteners.inertia_com_kg_m2) @ stl_rotation
    assert bare_tensor[0, 0] < upper_bound
    principal, axes = np.linalg.eigh(bare_tensor)
    np.testing.assert_allclose(principal, sorted(cam["bare_principal_inertia_kg_m2"]), atol=1e-9)
    assert abs(axes[0, 0]) > 0.99


@pytest.mark.parametrize("variant", ["gripper_chassis", "gripper_no_camera", "gripper"])
def test_only_drive_has_armature(variant):
    """校验驱动保留原参数，销轴、滑块和回差广义自由度的附加惯量为零。"""
    model = mujoco.MjModel.from_xml_path(str(get_mjcf_path(variant)))
    assert model.joint("gripper_drive").armature[0] == pytest.approx(0.002074755, abs=1e-12)
    np.testing.assert_allclose(model.joint("gripper_drive").range, [0, math.pi / 2], atol=1e-8)
    drive_dof = model.jnt_dofadr[model.joint("gripper_drive").id]
    assert np.flatnonzero(model.dof_armature).tolist() == [drive_dof]


@pytest.mark.parametrize("variant", ["gripper", "gripper_no_camera"])
def test_attach_prebuilt_like_tactile_grasp(variant):
    """按下游方式仅删除 freejoint，验证能挂到非世界层的移动载体下。"""
    path = get_mjcf_path(variant)
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str(path.parent / compiler.get("meshdir")))
    base = root.find(".//body[@name='base']")
    freejoints = base.findall("freejoint")
    assert [joint.attrib for joint in freejoints] == [{"name": "base_freejoint"}]
    for freejoint in freejoints:
        base.remove(freejoint)
    gripper = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    scene = mujoco.MjSpec()
    body = scene.worldbody.add_body(name="carrier", mass=1, fullinertia=[0.01, 0.01, 0.01, 0, 0, 0])
    body.add_joint(name="lift", type=mujoco.mjtJoint.mjJNT_SLIDE)
    scene.attach(gripper, prefix="gripper/", frame=body.add_frame())
    model = scene.compile()
    assert model.body("gripper/base").parentid[0] == model.body("carrier").id
    assert model.actuator("gripper/gripper_drive").id == 0
    assert model.site("gripper/tcp").id >= 0
    for module in get_tactile_layout(prefix="gripper/").values():
        for taxel in module["taxels"]:
            assert model.geom(taxel["geom_name"]).id >= 0
            assert model.site(taxel["site_name"]).id >= 0


@pytest.mark.parametrize("left,right,camera", CONFIGURATIONS)
def test_module_stroke_and_collision_exclusions(left, right, camera):
    """全行程检查结构碰撞，保留模组到达实际接触端点后的指间接触。"""
    limits = get_stroke_limits(left, right)
    assert limits["protocol_range_rad"] == (0.0, 1.7)
    assert limits["mechanism_range_rad"] == (0.0, math.pi / 2)
    assert limits["mechanism_single_finger_stroke_m"] == pytest.approx(0.035355339)
    contact_q = limits["first_contact_q_rad"]
    if "none" not in (left, right):
        assert contact_q == pytest.approx(1.450404, abs=1e-6)
        assert limits["first_contact_single_finger_closure_m"] == pytest.approx(0.03315)
    else:
        assert contact_q is None
    model = build_gripper_spec(left_sensor=left, right_sensor=right, with_camera=camera, fixed_base=True).compile()
    for q in np.linspace(0.0, math.pi / 2, 101):
        data = _forward(model, q)
        contacts = [(model.geom(c.geom1).name, model.geom(c.geom2).name) for c in data.contact]
        if contact_q is None or q < contact_q:
            assert not contacts, (left, right, camera, q, contacts)
        else:
            assert contacts
            assert all("taxel_geom_" in g or g.endswith("_pad_collision") for pair in contacts for g in pair)


def test_pts_housing_collides_with_environment():
    """确认外物仍能碰到铝壳，且壳体侧面接触不会进入触觉通道。"""
    spec = build_gripper_spec(fixed_base=True, with_camera=False)
    model = spec.compile()
    housing = _forward(model, 0.0).geom("left_sensor_housing_collision")
    probe_pos = housing.xpos + housing.xmat.reshape(3, 3) @ [0, 0.0125, 0]
    spec.worldbody.add_geom(name="probe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], pos=probe_pos)
    model = spec.compile()
    probe = model.geom("probe").id
    touched = {
        model.geom(c.geom2 if c.geom1 == probe else c.geom1).name
        for c in _forward(model, 0.0).contact
        if probe in (c.geom1, c.geom2)
    }
    assert "left_sensor_housing_collision" in touched
    assert all("taxel_geom_" not in name for name in touched)


@pytest.mark.parametrize("body_name,link_suffix", [
    ("left_finger", "left_finger"),
    ("right_finger", "right_finger"),
    ("base", "base"),
    ("cam_camera_assembly", "camera_mount_link"),
])
def test_external_probe_at_mesh_extrema(expand_xacro, body_name, link_suffix):
    """检查全部网格顶点位于 URDF 包络，并用自由探针抽测 MJCF 接触。"""
    spec = build_gripper_spec(fixed_base=True)
    probe_body = spec.worldbody.add_body(name="extrema_probe", pos=[0, 0, 1])
    probe_body.add_freejoint(name="extrema_probe_joint")
    probe_body.add_geom(name="extrema_probe_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0])
    model = spec.compile()
    data = _forward(model, 0.0)
    body_id = model.body(body_name).id
    probe_id = model.geom("extrema_probe_geom").id
    probe_pos = data.joint("extrema_probe_joint").qpos[:3]

    urdf = expand_xacro()
    link_name = f"{NS}_{link_suffix}"
    link_pose = _urdf_poses(urdf, {})[link_name]
    collisions = [
        (link_pose @ _origin(collision), collision.find("geometry")[0])
        for collision in urdf.findall(f"link[@name='{link_name}']/collision")
    ]

    mesh_geoms = np.flatnonzero((model.geom_bodyid == body_id) & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH))
    assert mesh_geoms.size
    for geom_id in mesh_geoms:
        geom = data.geom(int(geom_id))
        mesh = model.mesh(model.geom_dataid[geom_id])
        start = mesh.vertadr[0]
        vertices = model.mesh_vert[start : start + mesh.vertnum[0]] @ geom.xmat.reshape(3, 3).T + geom.xpos
        covered = np.zeros(len(vertices), dtype=bool)
        tolerance = 1e-8
        for pose, shape in collisions:
            local = (vertices - pose[:3, 3]) @ pose[:3, :3]
            if shape.tag == "box":
                half_size = np.fromstring(shape.get("size"), sep=" ") / 2
                covered |= np.all(np.abs(local) <= half_size + tolerance, axis=1)
            else:
                assert shape.tag == "cylinder"
                radius, half_length = float(shape.get("radius")), float(shape.get("length")) / 2
                covered |= (np.linalg.norm(local[:, :2], axis=1) <= radius + tolerance) & (
                    np.abs(local[:, 2]) <= half_length + tolerance
                )
        assert covered.all(), (body_name, mesh.name, vertices[~covered][0])

        extrema = np.r_[vertices.argmin(axis=0), vertices.argmax(axis=0)]
        if mesh.name == "finger_brace":
            # 加测复审发现漏检的支撑片顶点及其右侧对称点。
            reported = np.array([-0.076432, 0.017, 0.0947])
            if body_name == "right_finger":
                reported[:2] *= -1
            extrema = np.append(extrema, np.linalg.norm(vertices - reported, axis=1).argmin())
        for point in vertices[np.unique(extrema)]:
            probe_pos[:] = point
            mujoco.mj_forward(model, data)
            assert data.ncon >= 1, (body_name, mesh.name, point)
            assert any(
                model.geom_bodyid[contact.geom2 if contact.geom1 == probe_id else contact.geom1] == body_id
                for contact in data.contact
                if probe_id in (contact.geom1, contact.geom2)
            ), (body_name, mesh.name, point)


def test_traceable_payload_and_tactile_layout():
    """校验原始导出和网格摘要、静态构型诊断及触觉布局。"""
    specs = get_specs()
    root = get_data_root()
    raw_xml = root / "tools/onshape_export/raw_gripper.xml"
    if raw_xml.is_file():
        assert hashlib.sha256(raw_xml.read_bytes()).hexdigest() == specs["source"]["raw_mjcf_sha256"], (
            "raw_gripper.xml 已更新，请重新运行 tools/build_description.py"
        )
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "meshes").glob("*.stl")}
    assert specs["source"]["mesh_sha256"] == hashes
    payloads = specs["franka_desk_payload"]
    assert payloads["method"] == "analytic_closed_loop_mj_forward"
    for variant, suffix in (("gripper", "with_camera"), ("gripper_no_camera", "without_camera")):
        model = mujoco.MjModel.from_xml_path(str(get_mjcf_path(variant)))
        for state in ("open", "nominal_grasp"):
            payload = payloads[f"{suffix}_{state}"]
            assert payload["q_backlash_rad"] == payload["drive_error_rad"] == 0
            assert payload["max_angular_speed_rad_s"] == payload["max_slide_speed_m_s"] == 0
            assert payload["max_closure_error_m"] < 1e-6
            data = _forward(model, payload["q_drive_rad"])
            parts = [(float(model.body(i).mass[0]), data.body(i).xipos,
                      data.body(i).ximat.reshape(3, 3) @ np.diag(model.body(i).inertia) @ data.body(i).ximat.reshape(3, 3).T)
                     for i in range(1, model.nbody)]
            mass, com, inertia = _composite(parts)
            assert payload["mass_kg"] == pytest.approx(mass, abs=5e-7)
            np.testing.assert_allclose(payload["F_x_Cee_m"], com, atol=5e-7)
            np.testing.assert_allclose(payload["F_I_ee_kg_m2"], inertia, atol=5e-9)

    layout = get_tactile_layout("flat_pad", "pts_2_0", prefix="gripper/")
    assert layout["left"] is None
    assert get_tactile_layout(None, "none") == {"left": None, "right": None}
    assert layout["right"]["site_quat_in_sensor_mount_wxyz"] == [0, 1, 0, 0]
    assert layout["right"]["taxels"][4]["geom_name"] == "gripper/right_taxel_geom_11"
    assert layout["right"]["taxels"][4]["site_name"] == "gripper/right_taxel_11"
    assert get_closure_jacobian(0.0) == pytest.approx(0.035355339059)
    q, step = 0.75, 1e-5
    numeric = (get_joint_positions(q + step)["left_finger_slide"]
               - get_joint_positions(q - step)["left_finger_slide"]) / step
    assert get_closure_jacobian(q) == pytest.approx(numeric)


@pytest.mark.parametrize("left,right,camera", CONFIGURATIONS)
def test_srdf_members_and_nominal_camera_switch(expand_xacro, left, right, camera):
    """展开各组合并校验 SRDF 根层、组成员、免碰引用与可选相机帧。"""
    for nominal in ((False, True) if camera else (False,)):
        args = dict(left_sensor=left, right_sensor=right, with_camera=str(camera).lower(),
                    use_nominal_extrinsics=str(nominal).lower())
        urdf, srdf = expand_xacro(**args), expand_xacro(srdf=True, **args)
        links = {link.get("name") for link in urdf.findall("link")}
        joints = {joint.get("name") for joint in urdf.findall("joint")}
        group = srdf.find("group")
        motion = srdf.find(f"group[@name='{NS}_motion']")
        assert [joint.get("name") for joint in motion.findall("joint")] == [f"{NS}_left_finger_slide"]
        assert {link.get("name") for link in group.findall("link")} == links
        assert {joint.get("name") for joint in group.findall("joint")} == joints
        assert not group.findall("passive_joint")
        passive = {joint.get("name") for joint in srdf.findall("passive_joint")}
        assert passive == {f"{NS}_{name}" for name in (
            "right_finger_slide", "gripper_drive", "left_link_crank_pin", "right_link_crank_pin",
        )}
        assert passive <= joints
        for element in srdf.findall("disable_collisions"):
            pair = {element.get("link1"), element.get("link2")}
            assert pair <= links | {"panda_link7", "panda_link8"}
            assert pair != {f"{NS}_left_sensor_link", f"{NS}_right_sensor_link"}
        camera_links = {link.removeprefix(f"{NS}_d435i_") for link in links if link.startswith(f"{NS}_d435i_")}
        if camera and nominal:
            assert {"accel_frame", "accel_optical_frame", "gyro_frame", "gyro_optical_frame"} <= camera_links
        else:
            assert camera_links == ({"bottom_screw_frame", "link"} if camera else set())


@pytest.mark.parametrize("sensors", [("pts_2_0", "pts_2_0"), ("pts_2_0", "flat_pad"), ("flat_pad", "flat_pad")])
def test_close_gap_with_safety_distance(expand_xacro, sensors):
    """检查左右所有实体沿闭合轴的间隙，包含后向外扩的壳体与指背。"""
    args = dict(zip(("left_sensor", "right_sensor"), sensors, strict=True))
    close = expand_xacro(srdf=True, **args).find(f"group_state[@name='close'][@group='{NS}']")
    positions = {joint.get("name").removeprefix(f"{NS}_"): float(joint.get("value"))
                 for joint in close.findall("joint")}
    for margin in (0.0, 0.02):
        urdf = expand_xacro(safety_distance=margin, **args)
        poses = _urdf_poses(urdf, positions)
        inner = {}
        for side, sign in (("left", 1), ("right", -1)):
            extents = []
            for name in (f"{NS}_{side}_finger", f"{NS}_{side}_sensor_link"):
                for collision in urdf.findall(f"link[@name='{name}']/collision"):
                    pose = poses[name] @ _origin(collision)
                    shape = collision.find("geometry")[0]
                    if shape.tag == "sphere":
                        radius = float(shape.get("radius"))
                    else:
                        half_dims = np.array(list(map(float, shape.get("size").split()))) / 2
                        radius = np.abs(pose[0, :3]) @ half_dims
                    extents.append(sign * pose[0, 3] + radius)
            inner[side] = max(extents)
        assert -sum(inner.values()) == pytest.approx(0.0023, abs=1e-10)
