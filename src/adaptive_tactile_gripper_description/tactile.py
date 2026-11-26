"""MuJoCo 触觉接触对注册与兼容 ``contact_geom`` / ``pad_sites`` 的三轴力读取器。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from .assembly import get_sensor_spec, get_tactile_layout

_mj: Any = mujoco
if TYPE_CHECKING:
    type MjData = Any
    type MjModel = Any
    type MjSpec = Any
else:
    MjData = mujoco.MjData
    MjModel = mujoco.MjModel
    MjSpec = mujoco.MjSpec

_VALID_CONDIM = (1, 3, 4, 6)
_TORSIONAL_FRICTION = 0.02
_ROLLING_FRICTION = 0.001
_MIN_COUPLE_VARIANCE_M2 = 1e-16


def add_tactile_contact_pairs(
    scene: MjSpec,
    object_geom: str | Sequence[str],
    *,
    left_sensor: str | None = "pts_2_0",
    right_sensor: str | None = "pts_2_0",
    prefix: str = "",
    pair_prefix: str = "tactile_object",
    condim: int = 3,
    friction_coefficient: float | None = None,
    solref: Sequence[float] | None = None,
    solimp: Sequence[float] | None = None,
) -> list[str]:
    """为所选指尖触觉模组与目标物体几何体注册显式 MuJoCo 接触 ``<pair>``。

    - ``contact_geom``（如 ``pts_2_0``）：为每颗触觉碰撞球各注册 1 条 ``<pair>``；
    - ``pad_sites``（如 ``paxini_m2824``）：仅对整块 ``pad_geom_name`` 注册 **1 条** ``<pair>``，避免重复配对导致接触倍增；
    - 无触觉侧（如 ``flat_pad`` 或 ``None``）：自动跳过。
    已存在于 ``scene.pairs`` 的几何体组合会自动跳过以保证幂等。返回本次新增的 ``<pair>`` 名称列表。
    """
    if condim not in _VALID_CONDIM:
        raise ValueError(f"condim 必须属于 {_VALID_CONDIM}，当前为 {condim!r}")
    if friction_coefficient is not None and friction_coefficient < 0.0:
        raise ValueError(f"friction_coefficient 不能为负数：{friction_coefficient!r}")

    object_geoms = (
        [object_geom] if isinstance(object_geom, str) else list(dict.fromkeys(object_geom))
    )
    if not object_geoms or any(not name for name in object_geoms):
        raise ValueError("object_geom 必须包含至少一个非空几何体名称")

    layout = get_tactile_layout(left_sensor, right_sensor, prefix=prefix)
    existing_pairs = {frozenset((pair.geomname1, pair.geomname2)) for pair in scene.pairs}
    existing_names = {pair.name for pair in scene.pairs if pair.name}
    added_names: list[str] = []

    for side, sensor_name in (("left", left_sensor), ("right", right_sensor)):
        side_layout = layout[side]
        if side_layout is None or sensor_name is None or sensor_name == "none":
            continue
        mode = side_layout["mode"]
        if mode == "contact_geom":
            sensor_geoms = [taxel["geom_name"] for taxel in side_layout["taxels"]]
        elif mode == "pad_sites":
            sensor_geoms = [side_layout["pad_geom_name"]]
        else:
            raise ValueError(f"未知触觉布局模式 {mode!r}")

        contact_cfg = get_sensor_spec(sensor_name)["contact"]
        mu = float(
            friction_coefficient
            if friction_coefficient is not None
            else contact_cfg["friction_coefficient"]
        )
        pair_solref = [float(x) for x in (solref if solref is not None else contact_cfg["solref"])]
        pair_solimp = [float(x) for x in (solimp if solimp is not None else contact_cfg["solimp"])]
        pair_friction = [
            mu,
            mu,
            _TORSIONAL_FRICTION,
            _ROLLING_FRICTION,
            _ROLLING_FRICTION,
        ]

        for obj_idx, obj_geom in enumerate(object_geoms):
            for geom_idx, sensor_geom in enumerate(sensor_geoms):
                pair_key = frozenset((sensor_geom, obj_geom))
                if pair_key in existing_pairs:
                    continue
                base_name = (
                    f"{pair_prefix}_{side}_{geom_idx}"
                    if len(object_geoms) == 1
                    else f"{pair_prefix}_{side}_{obj_idx}_{geom_idx}"
                )
                pair_name = base_name
                suffix = 1
                while pair_name in existing_names:
                    pair_name = f"{base_name}_{suffix}"
                    suffix += 1
                scene.add_pair(
                    name=pair_name,
                    geomname1=sensor_geom,
                    geomname2=obj_geom,
                    condim=condim,
                    solref=pair_solref,
                    solimp=pair_solimp,
                    friction=pair_friction,
                )
                existing_pairs.add(pair_key)
                existing_names.add(pair_name)
                added_names.append(pair_name)
    return added_names


@dataclass(frozen=True, slots=True)
class _SideSpec:
    mode: str
    site_ids: np.ndarray
    grid_indices: np.ndarray | None
    geom_to_index: dict[int, int]
    pad_geom_id: int | None


def _grid_indices(layout: dict[str, Any]) -> np.ndarray:
    rows, cols = layout["rows"], layout["cols"]
    if type(rows) is not int or type(cols) is not int or rows < 1 or cols < 1:
        raise ValueError("触觉网格 rows 和 cols 必须为正整数")
    taxels = layout["taxels"]
    if len(taxels) != rows * cols:
        raise ValueError("触觉测点必须覆盖 rows × cols 网格")
    indices = np.full((rows, cols), -1, dtype=np.intp)
    for idx, taxel in enumerate(taxels):
        row, col = taxel.get("row"), taxel.get("col")
        if (
            type(row) is not int
            or type(col) is not int
            or not (0 <= row < rows and 0 <= col < cols)
        ):
            raise ValueError(f"触觉网格 row/col 无效或越界：{(row, col)!r}")
        if indices[row, col] != -1:
            raise ValueError(f"触觉网格 row/col 重复：{(row, col)!r}")
        indices[row, col] = idx
    return indices


def _convex_hull_indices(xy: np.ndarray) -> np.ndarray:
    """Return the counterclockwise hull of the projected taxels."""
    points, indices = np.unique(xy, axis=0, return_index=True)
    if len(points) <= 2:
        return indices

    sorted_xy = points.tolist()

    def half_hull(order: range) -> list[int]:
        half: list[int] = []
        for i in order:
            x, y = sorted_xy[i]
            while len(half) >= 2:
                ax, ay = sorted_xy[half[-2]]
                bx, by = sorted_xy[half[-1]]
                if (bx - ax) * (y - ay) - (by - ay) * (x - ax) > 0:
                    break
                half.pop()
            half.append(i)
        return half

    lo = half_hull(range(len(points)))
    hi = half_hull(range(len(points) - 1, -1, -1))
    return indices[lo[:-1] + hi[:-1]]


def _match_pad_moment(
    weights: np.ndarray,
    xy: np.ndarray,
    target: np.ndarray,
    hull: np.ndarray,
) -> np.ndarray:
    """Move the Gaussian centroid to the target or its closest point on the hull."""
    if len(hull) == 1:
        return weights
    vertices = xy[hull]
    edges = np.roll(vertices, -1, axis=0) - vertices
    edge_sq = np.einsum("ij,ij->i", edges, edges)
    inward = np.column_stack((-edges[:, 1], edges[:, 0]))
    target_rel = target - vertices
    if len(hull) == 2 or np.any(np.einsum("ij,ij->i", target_rel, inward) < 0.0):
        fractions = np.clip(np.einsum("ij,ij->i", target_rel, edges) / edge_sq, 0.0, 1.0)
        residual = target_rel - fractions[:, None] * edges
        edge_idx = int(np.argmin(np.einsum("ij,ij->i", residual, residual)))
        endpoint_fraction = fractions[edge_idx]
        blend = 1.0
    else:
        mean = weights @ xy
        direction = target - mean
        if np.linalg.norm(direction) <= 1e-12:
            return weights
        # Extend the centroid-to-target ray to the hull. Mixing in the force
        # at that intersection corrects the moment with nonnegative weights.
        approach = -(inward @ direction)
        mean_dist = np.einsum("ij,ij->i", mean - vertices, inward)
        steps = np.full(len(hull), np.inf)
        np.divide(mean_dist, approach, out=steps, where=approach > 0.0)
        edge_idx = int(np.argmin(steps))
        step = max(1.0, steps[edge_idx])
        blend = 1.0 / step
        intersection = (mean - vertices[edge_idx]) + step * direction
        endpoint_fraction = np.clip(intersection @ edges[edge_idx] / edge_sq[edge_idx], 0.0, 1.0)

    matched = (1.0 - blend) * weights
    matched[hull[edge_idx]] += blend * (1.0 - endpoint_fraction)
    matched[hull[(edge_idx + 1) % len(hull)]] += blend * endpoint_fraction
    return matched


class TactileForceReader:
    """统一的 MuJoCo 指尖三轴触觉力读取器，兼容 ``contact_geom`` 与 ``pad_sites``。

    ``read(data)`` 按侧返回形状为 ``(N, 3)`` 的浮点数组（列顺序 ``[Fx, Fy, Fz]``，单位 N，表达在各测点
    ``site`` 局部坐标系下），测点顺序与 ``layout[side]["taxels"]`` 一致，无触觉的一侧返回 ``None``。

    对 ``pad_sites``，用高斯权重分配接触力，并在首个测点的切平面内校正权重质心。
    凸包内重构投影压力中心及平面力臂产生的一阶力矩，凸包外将接触点投影至最近的凸包边界。
    权重非负且和为一，三维合力在同一坐标系内守恒（至浮点精度）。
    ``condim=4/6`` 的接触法向扭矩在垫面法向上的分量由高斯加权的零合力切向力偶重构，
    加权力臂方差不超过 1e-16 m² 时省略该力偶。``condim=6`` 的两个滚动力矩分量省略。
    对法向一致的平垫，纯压缩时各测点 ``Fz >= 0``。

    ``target_geoms`` 可为单个 geom 名称、ID 或两者的序列，省略时读取所有接触对象。
    """

    def __init__(
        self,
        model: MjModel,
        *,
        left_sensor: str | None = "pts_2_0",
        right_sensor: str | None = "pts_2_0",
        prefix: str = "",
        pad_sigma_m: float = 0.0025,
        target_geoms: str | int | Sequence[str | int] | None = None,
    ) -> None:
        if pad_sigma_m <= 0.0 or not np.isfinite(pad_sigma_m):
            raise ValueError(f"pad_sigma_m 必须为有限正数，当前为 {pad_sigma_m!r}")
        self._model = model
        self._pad_sigma_m = float(pad_sigma_m)
        self._layout = get_tactile_layout(left_sensor, right_sensor, prefix=prefix)
        self._target_geom_ids: frozenset[int] | None = (
            self._resolve_target_geoms(model, target_geoms) if target_geoms is not None else None
        )

        self._sides: dict[str, _SideSpec | None] = {}
        self._geom_owners: dict[int, list[str]] = {}
        for side in ("left", "right"):
            side_layout = self._layout[side]
            if side_layout is None:
                self._sides[side] = None
                continue
            mode = side_layout["mode"]
            grid_indices = _grid_indices(side_layout) if mode == "contact_geom" else None
            site_ids = np.array(
                [
                    self._require_id(model, _mj.mjtObj.mjOBJ_SITE, t["site_name"])
                    for t in side_layout["taxels"]
                ],
                dtype=np.intp,
            )
            if mode == "contact_geom":
                geom_to_index: dict[int, int] = {}
                for idx, taxel in enumerate(side_layout["taxels"]):
                    gid = self._require_id(model, _mj.mjtObj.mjOBJ_GEOM, taxel["geom_name"])
                    geom_to_index[gid] = idx
                    self._geom_owners.setdefault(gid, []).append(side)
                self._sides[side] = _SideSpec(
                    mode=mode,
                    site_ids=site_ids,
                    grid_indices=grid_indices,
                    geom_to_index=geom_to_index,
                    pad_geom_id=None,
                )
            elif mode == "pad_sites":
                pad_gid = self._require_id(
                    model, _mj.mjtObj.mjOBJ_GEOM, side_layout["pad_geom_name"]
                )
                self._geom_owners.setdefault(pad_gid, []).append(side)
                self._sides[side] = _SideSpec(
                    mode=mode,
                    site_ids=site_ids,
                    grid_indices=None,
                    geom_to_index={},
                    pad_geom_id=pad_gid,
                )
            else:
                raise ValueError(f"未知触觉布局模式 {mode!r}")

    @staticmethod
    def _require_id(model: MjModel, obj_type: Any, name: str) -> int:
        obj_id = int(_mj.mj_name2id(model, obj_type, name))
        if obj_id < 0:
            kind = "site" if obj_type == _mj.mjtObj.mjOBJ_SITE else "geom"
            raise ValueError(f"MuJoCo 模型缺少触觉 {kind} {name!r}")
        return obj_id

    @classmethod
    def _resolve_target_geoms(
        cls, model: MjModel, target_geoms: str | int | Sequence[str | int]
    ) -> frozenset[int]:
        resolved: set[int] = set()
        items = (target_geoms,) if isinstance(target_geoms, (str, int)) else target_geoms
        for item in items:
            if isinstance(item, str):
                resolved.add(cls._require_id(model, _mj.mjtObj.mjOBJ_GEOM, item))
            else:
                gid = int(item)
                if gid < 0 or gid >= int(model.ngeom):
                    raise ValueError(f"target_geoms 包含越界 geom ID：{gid}")
                resolved.add(gid)
        return frozenset(resolved)

    @property
    def layout(self) -> dict[str, dict[str, Any] | None]:
        """返回解析后的左右触觉布局字典。"""
        return self._layout

    def site_ids(self, side: str) -> tuple[int, ...]:
        """返回指定侧（``"left"`` 或 ``"right"``）的测点 ``site`` ID 元组；无触觉时返回空元组。"""
        if side not in self._sides:
            raise ValueError(f"side 必须为 'left' 或 'right'，当前为 {side!r}")
        spec = self._sides[side]
        return () if spec is None else tuple(int(x) for x in spec.site_ids)

    def read(self, data: MjData) -> dict[str, np.ndarray | None]:
        """读取当前帧左右指尖的 ``(N, 3)`` 局部三轴触觉力 ``[Fx, Fy, Fz]``（单位 N）。"""
        result: dict[str, np.ndarray | None] = {}
        pad_contacts: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
            "left": [],
            "right": [],
        }
        for side, spec in self._sides.items():
            result[side] = (
                None if spec is None else np.zeros((len(spec.site_ids), 3), dtype=np.float64)
            )

        cfrc = np.empty(6, dtype=np.float64)
        for cid in range(int(data.ncon)):
            con = data.contact[cid]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 not in self._geom_owners and g2 not in self._geom_owners:
                continue
            _mj.mj_contactForce(self._model, data, cid, cfrc)
            if not np.any(cfrc[:4]):
                continue
            contact_frame = np.asarray(con.frame, dtype=np.float64).reshape(3, 3)
            f_on_g2_world = contact_frame.T @ cfrc[:3]
            twist_on_g2_world = cfrc[3] * contact_frame[0]
            for sensor_gid, other_gid, sign in ((g1, g2, -1.0), (g2, g1, 1.0)):
                owners = self._geom_owners.get(sensor_gid)
                if not owners:
                    continue
                if self._target_geom_ids is not None and other_gid not in self._target_geom_ids:
                    continue
                force_world = sign * f_on_g2_world
                for side in owners:
                    spec = self._sides[side]
                    assert spec is not None
                    if spec.mode == "contact_geom":
                        idx = spec.geom_to_index[sensor_gid]
                        sid = int(spec.site_ids[idx])
                        rot = np.asarray(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
                        out = result[side]
                        assert out is not None
                        out[idx] += rot.T @ force_world
                    else:
                        pad_contacts[side].append(
                            (
                                np.asarray(con.pos, dtype=np.float64).copy(),
                                force_world,
                                sign * twist_on_g2_world,
                            )
                        )

        for side, contacts in pad_contacts.items():
            if not contacts:
                continue
            spec = self._sides[side]
            assert spec is not None
            site_pos = np.asarray(data.site_xpos[spec.site_ids], dtype=np.float64)
            site_rot = np.asarray(data.site_xmat[spec.site_ids], dtype=np.float64).reshape(-1, 3, 3)
            pad_rot = site_rot[0]
            tangent = pad_rot[:, :2]
            normal = pad_rot[:, 2]
            xy = (site_pos - site_pos[0]) @ tangent
            hull = _convex_hull_indices(xy)
            taxel_forces = np.zeros_like(site_pos)
            for pos, force, twist in contacts:
                delta = site_pos - pos
                sq_dist = np.einsum("ij,ij->i", delta, delta)
                # Scaling the distance avoids squaring a tiny sigma. Overflow
                # in the exponent gives zero weight; the closest site has weight 1.
                distance = np.sqrt(sq_dist - sq_dist.min())
                with np.errstate(over="ignore", under="ignore"):
                    gaussian = np.exp(-0.5 * (distance / self._pad_sigma_m) ** 2)
                gaussian /= gaussian.sum()
                weights = _match_pad_moment(gaussian, xy, (pos - site_pos[0]) @ tangent, hull)
                taxel_forces += weights[:, None] * (force @ pad_rot)

                twist_z = twist @ normal
                if twist_z != 0.0:
                    # Rotate the centered moment arms by 90 degrees to get
                    # zero net shear and a moment proportional to the variance.
                    offset = xy - gaussian @ xy
                    weighted_offset = gaussian[:, None] * offset
                    variance = np.sum(weighted_offset * offset)
                    if variance > _MIN_COUPLE_VARIANCE_M2:
                        couple = np.column_stack((-weighted_offset[:, 1], weighted_offset[:, 0]))
                        taxel_forces[:, :2] += (twist_z / variance) * couple
            if np.any(site_rot != pad_rot):
                taxel_forces = np.einsum("nij,ni->nj", site_rot, taxel_forces @ pad_rot.T)
            result[side] = taxel_forces

        return result

    def read_grid(self, data: MjData, side: str) -> np.ndarray:
        """按测点声明的 ``row`` 和 ``col`` 返回 ``(3, rows, cols)`` 的网格力张量。"""
        if side not in self._sides:
            raise ValueError(f"side 必须为 'left' 或 'right'，当前为 {side!r}")
        spec = self._sides[side]
        if spec is None or spec.grid_indices is None:
            raise ValueError(f"{side!r} 侧模组不是规则行列网格，请改用 read(data)[{side!r}]")
        forces = self.read(data)[side]
        assert forces is not None
        return forces[spec.grid_indices].transpose(2, 0, 1)
