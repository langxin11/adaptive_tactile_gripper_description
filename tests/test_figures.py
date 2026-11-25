"""出版图的数据和无显示导出回归。"""

import os
import subprocess
import sys

import numpy as np
from PIL import Image

from adaptive_tactile_gripper_description import get_data_root


def test_headless_figures(tmp_path, monkeypatch):
    tools = get_data_root() / "tools"
    monkeypatch.syspath_prepend(str(tools))
    import render_figures as figures

    frames, _, optics = figures.frame_data()
    assert np.isclose(frames["color"][0][0], -20)
    assert np.isclose(frames["depth"][0][0], -5)
    assert np.all(optics["color"]["fov"] < optics["depth"]["fov"])
    np.testing.assert_allclose(optics["color"]["offsets"], [8.4, 15.7], atol=0.05)
    assert abs(optics["color"]["angle"] - 17.7) < 0.05
    main, camera = figures.mass_data()
    np.testing.assert_allclose(main, [330, 183.65, 89, 53.35, 30])
    np.testing.assert_allclose(camera, [72, 18.955])
    d = figures.kinematics_data()
    assert np.isclose(d["gap"][0], 66.3)
    assert np.isclose(d["gap"][d["q"] == d["contact"]].item(), 0, atol=1e-8)
    np.testing.assert_allclose(
        np.gradient(2 * d["closure"], d["q"])[1:-1], d["jac"][1:-1], rtol=1e-4
    )
    np.testing.assert_allclose(d["forces"][0] * d["jac"] / 1000, 3.5)
    np.testing.assert_allclose([sum(row) for row in figures.mass_data()], [686, 90.955])

    env = dict(os.environ, MPLBACKEND="Agg", MUJOCO_GL="egl", MPLCONFIGDIR=str(tmp_path / "mpl"))
    env.pop("DISPLAY", None)
    subprocess.run(
        [sys.executable, str(tools / "render_figures.py"), "--out-dir", str(tmp_path)],
        env=env,
        check=True,
        timeout=90,
    )
    expected = [f"{name}.png" for name in ("frames", "kinematics", "mass_budget")]
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == expected
    for name in expected:
        with Image.open(tmp_path / name) as im:
            np.testing.assert_allclose(im.info["dpi"], [300, 300], atol=0.01)
