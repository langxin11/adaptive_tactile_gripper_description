"""从命令行查看、构建与绘制夹爪描述。"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import mujoco

from .assembly import (
    build_gripper_spec,
    get_data_root,
    get_stroke_limits,
    list_appearances,
    list_sensors,
)

_mj: Any = mujoco


def _info(left: str, right: str, with_camera: bool) -> None:
    model = build_gripper_spec(
        left_sensor=left, right_sensor=right, with_camera=with_camera, fixed_base=True
    ).compile()
    data = _mj.MjData(model)
    _mj.mj_forward(model, data)
    print(f"整机质量 {model.body_mass.sum() * 1000:.3f} g")
    sites = {"TCP": "tcp"}
    if with_camera:
        sites.update(
            {
                "彩色光心": "cam_d435i_color_optical_frame",
                "深度光心": "cam_d435i_depth_optical_frame",
            }
        )
    for label, name in sites.items():
        xyz = ", ".join(f"{x:.3f}" for x in data.site(name).xpos * 1000)
        print(f"{label} ({xyz}) mm")
    limits = get_stroke_limits(left, right)
    print(f"机构单指行程 {limits['mechanism_single_finger_stroke_m'] * 1000:.3f} mm")
    angle = limits["first_contact_q_rad"]
    print(f"首次接触角 {angle:.6f} rad" if angle is not None else "机构行程内无首次接触")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info", help="显示所选配置的机构参数")
    for side in ("left", "right"):
        info.add_argument(f"--{side}-sensor", choices=[*list_sensors(), "none"], default="pts_2_0")
    info.add_argument("--no-camera", action="store_true", help="卸下相机")
    commands.add_parser("appearances", help="列出外观预设")
    commands.add_parser("build", help="从源码仓库的 Onshape 导出构建资产")
    render = commands.add_parser("render", help="导出三张出版图的 300 dpi PNG")
    render.add_argument("--out-dir", type=Path, help="图表输出目录")
    args = parser.parse_args(argv)
    root = get_data_root()
    if args.command == "info":
        _info(args.left_sensor, args.right_sensor, not args.no_camera)
    elif args.command == "appearances":
        print("\n".join(list_appearances()))
    elif args.command == "build":
        script = root / "tools/build_description.py"
        if not script.is_file():
            parser.error("构建需要源码仓库及 Onshape 原始导出；请在源码仓库中运行 build。")
        return subprocess.call([sys.executable, str(script)])
    else:
        if any(importlib.util.find_spec(name) is None for name in ("matplotlib", "PIL")):
            parser.error("出图需要安装 matplotlib 和 pillow。")
        out = args.out_dir or (
            root / "docs/figures" if (root / "pyproject.toml").is_file() else Path.cwd() / "figures"
        )
        return subprocess.call(
            [sys.executable, str(root / "tools/render_figures.py"), "--out-dir", str(out)]
        )
    return 0
