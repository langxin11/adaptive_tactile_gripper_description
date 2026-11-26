"""验证夹爪的语义材质、跨格式配色和切换预设时的物理不变性。"""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from adaptive_tactile_gripper_description import build_gripper_spec, get_data_root, list_appearances
from adaptive_tactile_gripper_description.appearance import Appearance


def _library():
    return yaml.safe_load((get_data_root() / "config/appearance.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("preset", ["realistic", "cad"])
def test_compiled_visual_materials(preset):
    """用包含全部材质类别的混装夹爪，核对渲染场景最终解析出的颜色与高光。"""
    library = _library()
    assert set(list_appearances()) == {"realistic", "cad"}
    seen = set()
    for left_sensor, right_sensor, with_camera in (
        ("pts_2_0", "flat_pad", True),
        ("paxini_m2824", None, False),
    ):
        model = build_gripper_spec(
            appearance=preset,
            left_sensor=left_sensor,
            right_sensor=right_sensor,
            with_camera=with_camera,
        ).compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scene = mujoco.MjvScene(model, maxgeom=256)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        mujoco.mjv_updateScene(
            model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        for geom in scene.geoms[: scene.ngeom]:
            if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or model.geom_group[geom.objid] != 2:
                continue
            gid = geom.objid
            mesh = (
                model.mesh(model.geom_dataid[gid]).name
                if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
                else model.geom(gid).name
            )
            for prefix in ("left_", "right_", "cam_"):
                mesh = mesh.removeprefix(prefix)
            category = library["mesh_classes"][mesh]
            material = library["materials"][category]
            seen.add(mesh)
            assert geom.matid >= 0
            assert model.material(geom.matid).name in [
                prefix + category for prefix in ("", "left_", "right_", "cam_")
            ]
            color = library["presets"][preset]["mesh_rgba"].get(mesh, material["rgba"])
            np.testing.assert_array_equal(geom.rgba, np.asarray(color, dtype=np.float32))
            for field in ("specular", "shininess", "reflectance"):
                assert getattr(model, "mat_" + field)[geom.matid] == pytest.approx(material[field])
    assert seen == set(library["mesh_classes"])


def test_appearance_preserves_physics():
    """两个预设和随包外观的质量、局部惯性系、关节及碰撞几何逐元素一致。"""
    models = [
        build_gripper_spec(appearance=preset, right_sensor="flat_pad").compile()
        for preset in (None, "cad", "realistic")
    ]
    original = models[0]
    collision = original.geom_group == 3
    assert collision.any()
    for model in models[1:]:
        for field in (
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_iquat",
            "body_pos",
            "body_quat",
            "geom_group",
            "jnt_pos",
            "jnt_axis",
            "jnt_range",
        ):
            np.testing.assert_array_equal(getattr(original, field), getattr(model, field))
        for field in dir(original):
            if field.startswith("geom_") and isinstance(getattr(original, field), np.ndarray):
                np.testing.assert_array_equal(
                    getattr(original, field)[collision],
                    getattr(model, field)[collision],
                    err_msg=field,
                )


def test_invalid_appearance_and_unmapped_mesh():
    """未知预设与缺失视觉映射应给出可定位的错误。"""
    with pytest.raises(ValueError, match="未知外观预设"):
        build_gripper_spec(appearance="unknown")
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><geom group="2" type="mesh" mesh="new_part"/></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="new_part"):
        Appearance.from_library(_library(), "realistic").apply(spec)


@pytest.mark.parametrize(
    "module",
    [
        "gripper_chassis",
        "sensors/pts_2_0",
        "sensors/flat_pad",
        "sensors/paxini_m2824",
        "accessories/realsense_d435i",
    ],
)
def test_xacro_matches_asset_colors(module):
    """对照每个模组的原语和网格，验证随包 Xacro 颜色与 MJCF 一致。"""
    root = get_data_root()
    mjcf = ET.parse(root / "mjcf" / (module + ".xml"))
    xacro = ET.parse(root / "urdf" / (module + ".xacro"))
    colors = {mat.attrib["name"]: mat.attrib["rgba"] for mat in mjcf.findall("asset/material")}
    expected = {}
    for geom in mjcf.findall(".//worldbody//geom"):
        if geom.get("class") == "visual" or geom.get("group") == "2":
            mesh = geom.get("mesh", geom.get("name"))
            expected[mesh] = geom.get("rgba", colors[geom.attrib["material"]])
    actual = {}
    for visual in xacro.iter("visual"):
        mesh = visual.find("geometry/mesh")
        key = Path(mesh.attrib["filename"]).stem if mesh is not None else "pad_visual"
        prefix = "${prefix}_" if module.startswith("sensors/") else "${ns}_"
        assert visual.find("material").attrib["name"].startswith(prefix)
        color = visual.find("material/color").attrib["rgba"]
        assert color == expected[key]
        actual[key] = color
    assert actual == expected


@pytest.mark.parametrize("directory", ["sensors", "accessories"])
def test_discovered_module_requires_mapping(tmp_path, directory):
    """自定义类继承视觉组，未映射的网格须定位到文件，校验保留原文。"""
    import runpy

    build = runpy.run_path(str(get_data_root() / "tools/build_description.py"))
    path = tmp_path / directory / "custom.xml"
    path.parent.mkdir()
    source = """<mujoco><asset><mesh name="custom_mesh" file="housing.stl"/></asset>
    <default><default class="surface"><geom group="2" type="mesh"/></default></default>
    <worldbody><body childclass="surface"><geom mesh="custom_mesh"/></body></worldbody></mujoco>"""
    path.write_text(source)
    palette = Appearance.from_library(_library(), "realistic")
    with pytest.raises(ValueError, match=r"custom\.xml.*custom_mesh"):
        build["_validate_module_appearances"](tmp_path, palette)
    palette.mesh_classes["custom_mesh"] = "pts_housing"
    build["_validate_module_appearances"](tmp_path, palette)
    assert path.read_text() == source


def test_two_grippers_preserve_host_appearance():
    """双前缀挂载时，逐实例验证材质归属、闭环数量与宿主照明。"""
    host = mujoco.MjSpec.from_string("""<mujoco><visual><quality shadowsize="1024" offsamples="2"/>
      <headlight ambient=".11 .22 .33"/></visual><asset><material name="steel" rgba="1 0 0 1"/>
      <texture name="sky" type="skybox" builtin="gradient" width="8" height="48"/></asset>
      <worldbody><geom name="floor" type="plane" size="1 1 .01" material="steel"/></worldbody></mujoco>""")
    for i, (prefix, preset) in enumerate((("A/", "realistic"), ("B/", "cad"))):
        gripper = build_gripper_spec(appearance=preset, fixed_base=True)
        host.attach(gripper, prefix=prefix, frame=host.worldbody.add_frame(pos=[0.3 * i, 0, 0]))
    model = host.compile()
    assert model.neq == 8 and model.ntex == 1
    assert model.vis.quality.shadowsize == 1024 and model.vis.quality.offsamples == 2
    np.testing.assert_allclose(model.vis.headlight.ambient, [0.11, 0.22, 0.33])
    np.testing.assert_array_equal(model.mat_rgba[model.geom("floor").matid[0]], [1, 0, 0, 1])
    for gid in np.flatnonzero(model.geom_group == 2):
        prefix = model.body(model.geom_bodyid[gid]).name.split("/")[0] + "/"
        assert prefix in ("A/", "B/")
        assert model.geom_matid[gid] >= 0
        assert model.material(model.geom_matid[gid]).name.startswith(prefix)
    assert model.material("A/steel").id != model.material("B/steel").id


def test_default_build_rejects_custom_module(tmp_path):
    """原始导出可用时，覆盖默认构建入口以及既有扩展文件的只读语义。"""
    import subprocess
    import sys

    root = get_data_root()
    if not (root / "tools/onshape_export/raw_gripper.xml").is_file():
        pytest.skip("完整重建需要本地 Onshape 原始导出")
    module = tmp_path / "mjcf/sensors/custom.xml"
    module.parent.mkdir(parents=True)
    source = """<mujoco><compiler meshdir="../../meshes"/>
      <asset><mesh name="unmapped_housing" file="housing.stl"/></asset>
      <worldbody><body name="sensor_body"><geom type="mesh" group="2" mesh="unmapped_housing"/>
      </body></worldbody></mujoco>"""
    module.write_text(source)
    result = subprocess.run(
        [sys.executable, str(root / "tools/build_description.py"), "--out-root", str(tmp_path)],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert str(module) in result.stderr and "unmapped_housing" in result.stderr
    assert module.read_text() == source


def test_extended_module_preserves_colors(tmp_path, monkeypatch):
    import subprocess
    import sys

    from adaptive_tactile_gripper_description import assembly

    root = get_data_root()
    if not (root / "tools/onshape_export/raw_gripper.xml").is_file():
        pytest.skip("完整重建需要本地 Onshape 原始导出")
    path = tmp_path / "config/appearance.yaml"
    path.parent.mkdir()
    library = _library()
    for legacy_key in ("paxini_m2824", "paxini_m2824_adapter"):
        library["mesh_classes"].pop(legacy_key, None)
        library["presets"]["cad"]["mesh_rgba"].pop(legacy_key, None)
    library["mesh_classes"]["custom"] = "pts_housing"
    cad = [0.1, 0.2, 0.9, 1]
    library["presets"]["cad"]["mesh_rgba"]["custom"] = cad
    path.write_text(yaml.safe_dump(library))
    module = tmp_path / "mjcf/sensors/custom.xml"
    module.parent.mkdir(parents=True)
    module.write_text("""<mujoco><compiler meshdir="../../meshes"/>
        <asset><mesh name="custom" file="housing.stl"/></asset>
        <default><default class="finish"><geom group="2" rgba="1 0 1 1"/></default></default>
        <worldbody><body name="sensor_body" childclass="finish">
        <geom name="surface" type="mesh" mesh="custom"/>
        <geom name="contact" group="3" type="sphere" size=".003" rgba="0 1 0 1"/>
        </body></worldbody></mujoco>""")
    monkeypatch.setattr(assembly, "get_data_root", lambda: tmp_path)
    for _ in range(2):
        subprocess.run(
            [sys.executable, str(root / "tools/build_description.py"), "--out-root", str(tmp_path)],
            check=True,
        )
        saved = yaml.safe_load(path.read_text())
        assert saved["presets"]["cad"]["mesh_rgba"]["custom"] == cad
        for builtin_key in ("paxini_m2824", "paxini_m2824_adapter"):
            assert builtin_key in saved["mesh_classes"]
            assert builtin_key in saved["presets"]["cad"]["mesh_rgba"]
        assert (
            build_gripper_spec(left_sensor="paxini_m2824", appearance="cad").compile() is not None
        )
        for preset, expected in (
            ("realistic", library["materials"]["pts_housing"]["rgba"]),
            ("cad", cad),
        ):
            model = build_gripper_spec(left_sensor="custom", appearance=preset).compile()
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            scene = mujoco.MjvScene(model, maxgeom=256)
            mujoco.mjv_updateScene(
                model,
                data,
                mujoco.MjvOption(),
                None,
                mujoco.MjvCamera(),
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )
            gid = model.geom("left_surface").id
            visual = next(
                g
                for g in scene.geoms[: scene.ngeom]
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == gid
            )
            assert model.material(visual.matid).name == "left_pts_housing"
            np.testing.assert_allclose(visual.rgba, expected)
            contact = model.geom("left_contact")
            assert contact.matid.item() == -1 and contact.contype.item() == 1
            np.testing.assert_array_equal(contact.rgba, [0, 1, 0, 1])


@pytest.mark.parametrize("preset", [None, "realistic", "cad"])
def test_replicated_visuals_preserve_contact_material(tmp_path, monkeypatch, preset):
    """对照展开后的两个网格和共享材质的碰撞球，读取场景最终材质与颜色。"""
    from adaptive_tactile_gripper_description import assembly

    root = get_data_root()
    sensors = tmp_path / "mjcf/sensors"
    sensors.mkdir(parents=True)
    for path in ("meshes", "config", "mjcf/gripper_chassis.xml"):
        (tmp_path / path).symlink_to(root / path)
    (sensors / "array.xml").write_text("""<mujoco>
        <compiler meshdir="../../meshes"/>
        <asset><mesh name="housing" file="housing.stl"/>
          <material name="pts_housing" rgba="0 1 0 1"/></asset>
        <default>
          <default class="surface"><geom group="2" material="pts_housing" rgba="1 0 1 1"/></default>
          <default class="contact"><geom group="3" material="pts_housing" contype="4" conaffinity="9"/></default>
        </default>
        <worldbody><body name="sensor_body" childclass="surface">
          <replicate count="2" offset=".015 0 0"><geom name="surface2" type="mesh" mesh="housing"/></replicate>
          <geom name="guard" class="contact" type="sphere" size=".003"/>
        </body></worldbody></mujoco>""")
    monkeypatch.setattr(assembly, "get_data_root", lambda: tmp_path)
    options = {} if preset is None else {"appearance": preset}
    model = build_gripper_spec(
        left_sensor="array", right_sensor=None, with_camera=False, **options
    ).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=256)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1
    mujoco.mjv_updateScene(
        model, data, opt, None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    rendered = {
        model.geom(g.objid).name: g
        for g in scene.geoms[: scene.ngeom]
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM
    }
    colors = {
        None: [1, 0, 1, 1],
        "realistic": _library()["materials"]["pts_housing"]["rgba"],
        "cad": _library()["presets"]["cad"]["mesh_rgba"]["housing"],
    }
    for name in ("left_surface20", "left_surface21"):
        visual = rendered[name]
        material_name = "left_pts_housing" if preset is None else "left_pts_housing_visual"
        assert model.material(visual.matid).name == material_name
        np.testing.assert_array_equal(visual.rgba, np.asarray(colors[preset], dtype=np.float32))
    guard = rendered["left_guard"]
    assert model.material(guard.matid).name == "left_pts_housing"
    np.testing.assert_array_equal(guard.rgba, [0, 1, 0, 1])
    assert model.geom_group[guard.objid] == 3
    assert (model.geom_contype[guard.objid], model.geom_conaffinity[guard.objid]) == (4, 9)
