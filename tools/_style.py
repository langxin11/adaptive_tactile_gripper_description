"""夹爪图版的中文字体、真机色和单栏／双栏出版尺寸。"""

from importlib import import_module
from pathlib import Path

import matplotlib as mpl
import yaml
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import ImageFont

try:
    maniplot = import_module("maniplot")
except ModuleNotFoundError as error:
    if error.name != "maniplot":
        raise
    maniplot = None

WIDTH_IN = {"single": 3.50, "double": 7.16}
DPI = 300
XYZ_COLORS = ("#D55E00", "#009E73", "#005596")
FONT_FAMILIES = ("Noto Sans CJK SC", "Microsoft YaHei", "Noto Sans CJK JP", "WenQuanYi Zen Hei")
_fonts = font_manager.fontManager.ttflist
FONT = next((name for name in FONT_FAMILIES if any(f.name == name for f in _fonts)), "DejaVu Sans")

_library = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "config/appearance.yaml").read_text()
)
COLORS = {name: value["rgba"] for name, value in _library["materials"].items()}
COLORS.update(_library["presets"]["realistic"]["mesh_rgba"])


def panel_font(size):
    """按中文字体集合的字面索引加载预览图题字。"""
    path = font_manager.findfont(font_manager.FontProperties(family=[FONT]))
    entry = next(f for f in _fonts if f.fname == path and f.name == FONT)
    return ImageFont.truetype(path, size, index=getattr(entry, "index", 0))


def use_style():
    """按中文论文规范排版，采用任务指定的中文无衬线字体。"""
    if maniplot:
        maniplot.use_style("cn_thesis")
    mpl.rcParams.update(
        {
            "font.family": [FONT],
            "font.size": 9,
            "mathtext.fontset": "stix",
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.linewidth": 0.75,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "xtick.top": False,
            "ytick.right": False,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.unicode_minus": False,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "figure.constrained_layout.use": False,
            "figure.autolayout": False,
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 150,
            "axes.edgecolor": "#222222",
            "grid.alpha": 0.65,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "legend.borderpad": 0.35,
            "legend.handlelength": 1.8,
            "savefig.dpi": DPI,
        }
    )


def label_panels(axes):
    """将分图编号与居中的栏题置于同一行。"""
    if maniplot:
        maniplot.add_panel_labels(axes, loc=(0.08, 1.025), fontsize=9)
    else:
        for i, ax in enumerate(axes):
            ax.text(
                0.08,
                1.025,
                f"({chr(97 + i)})",
                transform=ax.transAxes,
                weight="bold",
                fontsize=9,
                ha="right",
                va="bottom",
            )


def save_figure(fig, path):
    """按约定物理栏宽导出 300 dpi 的 PNG 图表。"""
    # 在四周为紧边界导出保留 0.06 英寸，保持约定的物理栏宽。
    w, h = fig.get_size_inches()
    margin = 0.06
    frame = Rectangle(
        (margin / w, margin / h),
        1 - 2 * margin / w,
        1 - 2 * margin / h,
        transform=fig.transFigure,
        fill=False,
        edgecolor="none",
    )
    fig.add_artist(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=DPI, bbox_inches="tight", pad_inches=margin)
