"""读取外观预设，并将网格配色和表面参数应用到单个 MJCF 模组。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import yaml

if TYPE_CHECKING:
    type MjsGeom = Any
    type MjSpec = Any
else:
    MjsGeom = mujoco.MjsGeom
    MjSpec = mujoco.MjSpec


def read_appearance(path: Path) -> dict:
    """读取描述包中的语义材质库。"""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def is_visual(geom: MjsGeom) -> bool:
    """按解析了默认类的视觉组和接触掩码识别视觉件。"""
    return geom.group == 2 or (geom.contype == 0 and geom.conaffinity == 0)


@dataclass
class Appearance:
    """已选定的外观，按规范网格名在底盘与附件之间共享。"""

    materials: dict
    mesh_classes: dict[str, str]
    smooth_meshes: set[str]
    mesh_rgba: dict[str, list[float]]
    base_rgba: list[float] | None

    @classmethod
    def from_library(cls, library: dict, name: str) -> Appearance:
        """解析预设，其中逐件颜色优先于类别颜色。"""
        presets = library["presets"]
        if name not in presets:
            raise ValueError(f"未知外观预设 {name!r}，可选：{list(presets)}")
        preset = presets[name]
        return cls(
            library["materials"],
            library["mesh_classes"],
            set(library["smooth_meshes"]),
            preset.get("mesh_rgba", {}),
            preset.get("base_rgba"),
        )

    def category(self, mesh: str) -> str:
        """查找视觉网格或原语已登记的材质类别。"""
        category = self.mesh_classes.get(mesh)
        if category is None or category not in self.materials:
            raise ValueError(f"视觉几何 {mesh!r} 缺少有效的材质类别映射")
        return category

    def material_attributes(self, category: str) -> dict:
        """返回预设下该类别的材质底色与表面参数。"""
        attributes = self.materials[category]
        return {
            "rgba": self.base_rgba or attributes["rgba"],
            **{key: attributes[key] for key in ("specular", "shininess", "reflectance")},
        }

    def apply(self, spec: MjSpec) -> None:
        """为解析并展开后的视觉几何设置外观，保留碰撞几何引用的材质。"""
        protected = {geom.material for geom in spec.geoms if not is_visual(geom)}
        reserved = {material.name for material in spec.materials} | self.materials.keys()
        names = {}
        for geom in spec.geoms:
            if not is_visual(geom):
                continue
            mesh = geom.meshname or geom.name
            category = self.category(mesh)
            if category not in names:
                name = category
                if category in protected:
                    name = f"{category}_visual"
                    while name in reserved:
                        name += "_"
                reserved.add(name)
                material = spec.material(name)
                if material is None:
                    material = spec.add_material(name=name)
                for key, value in self.material_attributes(category).items():
                    setattr(material, key, value)
                names[category] = name
            geom.material = names[category]
            # 中性灰让 MuJoCo 使用材质颜色，覆盖默认类继承的逐件颜色。
            geom.rgba = self.mesh_rgba.get(mesh, [0.5, 0.5, 0.5, 1])
