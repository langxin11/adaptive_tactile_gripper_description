"""验证命令行配置参数与双打包版本。"""

import re
import tomllib
import xml.etree.ElementTree as ET

from adaptive_tactile_gripper_description import __version__, get_data_root
from adaptive_tactile_gripper_description.cli import main


def test_version():
    root = get_data_root()
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    assert (
        metadata["project"]["version"]
        == ET.parse(root / "package.xml").findtext("version")
        == __version__
    )


def test_info(capsys):
    assert main(["info"]) == 0
    text = capsys.readouterr().out
    for value in ("776.955 g", "145.550", "35.355 mm", "1.450404 rad", "-20.000", "-5.000"):
        assert value in text
    assert main(["info", "--no-camera"]) == 0
    text = capsys.readouterr().out
    assert "686.000 g" in text and "光心" not in text


def test_integration_toml_examples():
    guide = (get_data_root() / "docs/integration.md").read_text(encoding="utf-8")
    examples = re.findall(r"^```toml\n(.*?)^```$", guide, re.MULTILINE | re.DOTALL)
    assert examples
    for example in examples:
        tomllib.loads(example)
