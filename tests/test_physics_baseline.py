"""用 335f584 的分区摘要守护全部随包 MJCF 的物理描述。"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from adaptive_tactile_gripper_description import get_data_root

BASELINE = json.loads(Path(__file__).with_name("physics_335f584.json").read_text())


def physics_summary(source):
    """保留层级、顺序及全部物理属性，忽略排版、视觉配色和平滑标志。"""
    root = ET.fromstring(source)
    asset = root.find("asset")
    if asset is not None:
        for material in asset.findall("material"):
            asset.remove(material)
        for mesh in asset.findall("mesh"):
            mesh.attrib.pop("smoothnormal", None)
        if not len(asset):
            root.remove(asset)
    for geom in root.findall(".//worldbody//geom"):
        if geom.get("group") == "2" or geom.get("class", "").split("_")[-1] == "visual":
            geom.attrib.pop("rgba", None)
            geom.attrib.pop("material", None)
    result = []
    for section in root:
        # 显式保存闭环参数，回归时可直接看到对象、锚点或求解参数的差值。
        if section.tag == "equality":
            value = [dict(type=node.tag, **node.attrib) for node in section]
        else:
            canonical = ET.canonicalize(ET.tostring(section), strip_text=True)
            value = hashlib.sha256(canonical.encode()).hexdigest()
        result.append([section.tag, value])
    return result


@pytest.mark.parametrize("path", sorted(BASELINE))
def test_physics_matches_release(path):
    actual = physics_summary((get_data_root() / path).read_bytes())
    assert actual == BASELINE[path]


def test_extra_equality_section(tmp_path):
    import mujoco

    path = "mjcf/gripper_no_camera.xml"
    root = ET.parse(get_data_root() / path).getroot()
    root.find("compiler").set("meshdir", str(get_data_root() / "meshes"))
    extra = ET.fromstring('<equality><joint joint1="gripper_drive"/></equality>')
    root.insert(list(root).index(root.find("equality")), extra)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert model.neq == 5
    root.find("compiler").set(
        "meshdir", ET.parse(get_data_root() / path).find("compiler").attrib["meshdir"]
    )
    assert physics_summary(ET.tostring(root)) != BASELINE[path]
