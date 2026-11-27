"""MuJoCo 触觉接触对注册、三轴阵列力读取器与六轴接触力旋量（Wrench）读取器。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

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
_FRONT_FACE_NORMAL_MIN_DOT = 0.5
_FRONT_FACE_PLANE_TOL_M = 5e-5
_DSBL_NATIVECCD = int(getattr(_mj.mjtDisableBit, "mjDSBL_NATIVECCD", 0))


def _require_box_pad_nativeccd(model: MjModel) -> None:
    if _DSBL_NATIVECCD and (int(model.opt.disableflags) & _DSBL_NATIVECCD):
        raise ValueError(
            "pad_sites 盒式平垫的前感应面接触过滤依赖 MuJoCo nativeccd 接触几何，"
            "不支持在模型选项中禁用 nativeccd（disableflags 含 nativeccd=disable）"
        )


def _require_contact_default(sensor_name: str, contact_cfg: dict[str, Any] | None, key: str) -> Any:
    if contact_cfg is None or key not in contact_cfg or contact_cfg[key] is None:
        raise ValueError(
            f"传感器 {sensor_name!r} 缺少默认接触参数 contact.{key}，请在 YAML 中补充或显式传入 {key}"
        )
    return contact_cfg[key]


def add_tactile_contact_pairs(
    scene: MjSpec,
    object_geom: str | Sequence[str],
    *,
    left_sensor: str | None = "pts_2_0",
    right_sensor: str | None = "pts_2_0",
    prefix: str = "",
    pair_prefix: str = "tactile_object",
    condim: int | None = None,
    friction_coefficient: float | None = None,
    torsional_friction: float | None = None,
    rolling_friction: float | None = None,
    solref: Sequence[float] | None = None,
    solimp: Sequence[float] | None = None,
) -> list[str]:
    """为所选指尖触觉模组与目标物体几何体注册显式 MuJoCo 接触 ``<pair>``。

    - ``contact_geom``（如 ``pts_2_0``）：为每颗触觉碰撞球各注册 1 条 ``<pair>``；
    - ``pad_sites``（如 ``paxini_m2824``）与 ``pad_wrench``（如 ``soft_dome``）：仅对整块
      ``pad_geom_name`` 注册 **1 条** ``<pair>``，避免重复配对导致接触倍增；
    - 无触觉侧（如 ``flat_pad`` 或 ``None``）：自动跳过。
    ``condim`` 省略时优先采用模组 ``contact.condim``（如 ``soft_dome`` 默认 ``4``），未声明时回退为 ``3``。
    已存在于 ``scene.pairs`` 的几何体组合会自动跳过以保证幂等。返回本次新增的 ``<pair>`` 名称列表。
    """
    if condim is not None and (isinstance(condim, bool) or condim not in _VALID_CONDIM):
        raise ValueError(f"condim 必须属于 {_VALID_CONDIM}，当前为 {condim!r}")
    if friction_coefficient is not None and friction_coefficient < 0.0:
        raise ValueError(f"friction_coefficient 不能为负数：{friction_coefficient!r}")
    if torsional_friction is not None and torsional_friction < 0.0:
        raise ValueError(f"torsional_friction 不能为负数：{torsional_friction!r}")
    if rolling_friction is not None and rolling_friction < 0.0:
        raise ValueError(f"rolling_friction 不能为负数：{rolling_friction!r}")

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
        elif mode in ("pad_sites", "pad_wrench"):
            sensor_geoms = [side_layout["pad_geom_name"]]
        else:
            raise ValueError(f"未知触觉布局模式 {mode!r}")

        pending_pairs: list[tuple[int, int, str, str]] = [
            (obj_idx, geom_idx, sensor_geom, obj_geom)
            for obj_idx, obj_geom in enumerate(object_geoms)
            for geom_idx, sensor_geom in enumerate(sensor_geoms)
            if frozenset((sensor_geom, obj_geom)) not in existing_pairs
        ]
        if not pending_pairs:
            continue

        contact_cfg: dict[str, Any] | None = None
        if any(
            param is None
            for param in (
                condim,
                friction_coefficient,
                torsional_friction,
                rolling_friction,
                solref,
                solimp,
            )
        ):
            raw_contact = get_sensor_spec(sensor_name).get("contact")
            if isinstance(raw_contact, dict):
                contact_cfg = raw_contact

        raw_condim = (
            condim
            if condim is not None
            else (
                contact_cfg.get("condim", 3)
                if contact_cfg is not None and contact_cfg.get("condim") is not None
                else 3
            )
        )
        if isinstance(raw_condim, bool) or raw_condim not in _VALID_CONDIM:
            raise ValueError(f"condim 必须属于 {_VALID_CONDIM}，当前为 {raw_condim!r}")
        pair_condim = int(raw_condim)

        mu = float(
            friction_coefficient
            if friction_coefficient is not None
            else _require_contact_default(sensor_name, contact_cfg, "friction_coefficient")
        )
        if mu < 0.0:
            raise ValueError(f"friction_coefficient 不能为负数：{mu!r}")
        pair_torsional = float(
            torsional_friction
            if torsional_friction is not None
            else (
                contact_cfg["torsional_friction"]
                if contact_cfg is not None and contact_cfg.get("torsional_friction") is not None
                else _TORSIONAL_FRICTION
            )
        )
        if pair_torsional < 0.0:
            raise ValueError(f"torsional_friction 不能为负数：{pair_torsional!r}")
        pair_rolling = float(
            rolling_friction
            if rolling_friction is not None
            else (
                contact_cfg["rolling_friction"]
                if contact_cfg is not None and contact_cfg.get("rolling_friction") is not None
                else _ROLLING_FRICTION
            )
        )
        if pair_rolling < 0.0:
            raise ValueError(f"rolling_friction 不能为负数：{pair_rolling!r}")
        pair_solref = [
            float(x)
            for x in (
                solref
                if solref is not None
                else _require_contact_default(sensor_name, contact_cfg, "solref")
            )
        ]
        pair_solimp = [
            float(x)
            for x in (
                solimp
                if solimp is not None
                else _require_contact_default(sensor_name, contact_cfg, "solimp")
            )
        ]
        pair_friction = [
            mu,
            mu,
            pair_torsional,
            pair_rolling,
            pair_rolling,
        ]

        for obj_idx, geom_idx, sensor_geom, obj_geom in pending_pairs:
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
                condim=pair_condim,
                solref=pair_solref,
                solimp=pair_solimp,
                friction=pair_friction,
            )
            existing_pairs.add(pair_key)
            existing_names.add(pair_name)
            added_names.append(pair_name)
    return added_names


@dataclass(frozen=True, slots=True)
class _HullGeometry:
    seg_starts: np.ndarray
    seg_ends: np.ndarray
    vertices: np.ndarray
    edges: np.ndarray
    edge_sq: np.ndarray
    inward_unit: np.ndarray
    scale: float
    is_collinear: bool


@dataclass(frozen=True, slots=True)
class _RigidPadCache:
    pos_snapshot: np.ndarray
    quat_snapshot: np.ndarray
    geom_pos_snapshot: np.ndarray
    geom_quat_snapshot: np.ndarray
    geom_size_snapshot: np.ndarray
    geom_type_snapshot: int
    pad_rot_in_geom: np.ndarray
    face_origin_in_geom: np.ndarray
    local_pos: np.ndarray
    xy: np.ndarray
    hull: _HullGeometry
    same_orientation: bool
    rel_rot: np.ndarray | None


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


def _convex_hull_indices(xy: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return the counterclockwise hull (preserving collinear edge points) and collinearity flag."""
    points, indices = np.unique(xy, axis=0, return_index=True)
    if len(points) <= 2:
        return indices, True

    sorted_xy = points.tolist()
    scale = float(np.ptp(points, axis=0).max())
    eps = 1e-12 * scale * scale

    def half_hull(order: range) -> list[int]:
        half: list[int] = []
        for i in order:
            x, y = sorted_xy[i]
            while len(half) >= 2:
                ax, ay = sorted_xy[half[-2]]
                bx, by = sorted_xy[half[-1]]
                if (bx - ax) * (y - ay) - (by - ay) * (x - ax) >= -eps:
                    break
                half.pop()
            half.append(i)
        return half

    lo = half_hull(range(len(points)))
    hi = half_hull(range(len(points) - 1, -1, -1))
    if len(lo) == len(points) and len(hi) == len(points):
        return indices[lo], True
    return indices[lo[:-1] + hi[:-1]], False


def _build_hull_geometry(xy: np.ndarray) -> _HullGeometry:
    hull, is_collinear = _convex_hull_indices(xy)
    scale = max(float(np.ptp(xy, axis=0).max()), 1e-6) if len(xy) else 1.0
    if len(hull) <= 1:
        empty = np.empty((0, 2), dtype=np.float64)
        empty_idx = np.empty(0, dtype=np.intp)
        return _HullGeometry(
            empty_idx,
            empty_idx,
            empty,
            empty,
            np.empty(0, dtype=np.float64),
            empty,
            scale,
            True,
        )
    if is_collinear:
        seg_starts = hull[:-1]
        seg_ends = hull[1:]
    else:
        seg_starts = hull
        seg_ends = np.roll(hull, -1)
    vertices = xy[seg_starts]
    edges = xy[seg_ends] - vertices
    edge_sq = np.einsum("ij,ij->i", edges, edges)
    edge_len = np.sqrt(edge_sq)
    inward_unit = np.column_stack((-edges[:, 1], edges[:, 0])) / edge_len[:, None]
    return _HullGeometry(
        seg_starts, seg_ends, vertices, edges, edge_sq, inward_unit, scale, is_collinear
    )


def _quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
    mat = np.empty(9, dtype=np.float64)
    _mj.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    return mat.reshape(3, 3)


def _orthonormal_basis_from_normal(
    z_axis: np.ndarray, hint_x: np.ndarray | None = None
) -> np.ndarray:
    z = np.asarray(z_axis, dtype=np.float64)
    z = z / float(np.linalg.norm(z))
    if hint_x is None:
        hint_x = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = hint_x - float(hint_x @ z) * z
    x_norm = float(np.linalg.norm(x))
    if x_norm <= 1e-12:
        fallback = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
        x = fallback - float(fallback @ z) * z
        x_norm = float(np.linalg.norm(x))
    x /= x_norm
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def _pad_frame_in_geom(
    model: MjModel,
    pad_gid: int,
    site_pos_in_geom: np.ndarray,
    site_rot_in_geom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive sensing tangent frame & inward normal from the pad collision geom."""
    gtype = int(model.geom_type[pad_gid])
    if gtype == int(_mj.mjtGeom.mjGEOM_BOX):
        half = np.asarray(model.geom_size[pad_gid], dtype=np.float64)
        centroid = site_pos_in_geom.mean(axis=0)
        normals = site_rot_in_geom[:, :, 2]
        mean_normal = normals.mean(axis=0)
        axis_alignment = np.abs(normals).mean(axis=0)
        axis = -1
        face_sign = 1.0
        if float(axis_alignment.max()) > 1e-6:
            axis = int(np.argmax(axis_alignment))
            if abs(float(centroid[axis])) > 1e-6 * max(float(half[axis]), 1e-12):
                face_sign = 1.0 if centroid[axis] >= 0.0 else -1.0
            elif abs(float(mean_normal[axis])) > 1e-6:
                face_sign = -1.0 if mean_normal[axis] >= 0.0 else 1.0
        else:
            ratios = np.abs(centroid) / np.maximum(half, 1e-12)
            if float(ratios.max()) > 1e-3:
                axis = int(np.argmax(ratios))
                face_sign = 1.0 if centroid[axis] >= 0.0 else -1.0
        if axis >= 0:
            z_local = np.zeros(3, dtype=np.float64)
            z_local[axis] = -face_sign
            hint = np.zeros(3, dtype=np.float64)
            hint[(axis + 1) % 3 if axis != 2 else 0] = 1.0
            pad_rot = _orthonormal_basis_from_normal(z_local, hint)
            face_origin = np.zeros(3, dtype=np.float64)
            face_origin[axis] = face_sign * float(half[axis])
            return pad_rot, face_origin
    if gtype == int(_mj.mjtGeom.mjGEOM_PLANE):
        return np.diag([1.0, -1.0, -1.0]), np.zeros(3, dtype=np.float64)
    mean_normal = site_rot_in_geom[:, :, 2].mean(axis=0)
    if float(np.linalg.norm(mean_normal)) <= 1e-12:
        mean_normal = site_rot_in_geom[0, :, 2]
    pad_rot = _orthonormal_basis_from_normal(mean_normal, site_rot_in_geom[0, :, 0])
    return pad_rot, np.zeros(3, dtype=np.float64)


def _build_rigid_pad_cache(
    model: MjModel, pad_gid: int, site_ids: np.ndarray
) -> _RigidPadCache | None:
    if len(site_ids) == 0:
        return None
    pad_body = int(model.geom_bodyid[pad_gid])
    body_ids = np.asarray(model.site_bodyid[site_ids], dtype=np.intp)
    if not bool(np.all(body_ids == pad_body)):
        return None
    pos_body = np.asarray(model.site_pos[site_ids], dtype=np.float64).copy()
    quats = np.asarray(model.site_quat[site_ids], dtype=np.float64).copy()
    geom_pos = np.asarray(model.geom_pos[pad_gid], dtype=np.float64).copy()
    geom_quat = np.asarray(model.geom_quat[pad_gid], dtype=np.float64).copy()
    geom_size = np.asarray(model.geom_size[pad_gid], dtype=np.float64).copy()
    geom_type = int(model.geom_type[pad_gid])
    geom_rot = _quat_wxyz_to_mat(geom_quat)
    rot_body = np.stack([_quat_wxyz_to_mat(q) for q in quats], axis=0)
    site_pos_in_geom = (pos_body - geom_pos) @ geom_rot
    site_rot_in_geom = np.einsum("ij,njk->nik", geom_rot.T, rot_body)
    pad_rot_in_geom, face_origin_in_geom = _pad_frame_in_geom(
        model, pad_gid, site_pos_in_geom, site_rot_in_geom
    )
    local_pos = (site_pos_in_geom - face_origin_in_geom) @ pad_rot_in_geom
    xy = local_pos[:, :2].copy()
    rel_rot = np.einsum("ij,njk->nik", pad_rot_in_geom.T, site_rot_in_geom)
    same_orientation = bool(np.allclose(rel_rot, np.eye(3, dtype=np.float64), atol=1e-12))
    return _RigidPadCache(
        pos_snapshot=pos_body,
        quat_snapshot=quats,
        geom_pos_snapshot=geom_pos,
        geom_quat_snapshot=geom_quat,
        geom_size_snapshot=geom_size,
        geom_type_snapshot=geom_type,
        pad_rot_in_geom=pad_rot_in_geom,
        face_origin_in_geom=face_origin_in_geom,
        local_pos=local_pos,
        xy=xy,
        hull=_build_hull_geometry(xy),
        same_orientation=same_orientation,
        rel_rot=None if same_orientation else rel_rot,
    )


def _refresh_rigid_pad_cache(
    model: MjModel,
    pad_gid: int,
    site_ids: np.ndarray,
    cache: _RigidPadCache | None,
) -> _RigidPadCache | None:
    if cache is None:
        return None
    cur_pos = np.asarray(model.site_pos[site_ids], dtype=np.float64)
    cur_quat = np.asarray(model.site_quat[site_ids], dtype=np.float64)
    cur_gpos = np.asarray(model.geom_pos[pad_gid], dtype=np.float64)
    cur_gquat = np.asarray(model.geom_quat[pad_gid], dtype=np.float64)
    cur_gsize = np.asarray(model.geom_size[pad_gid], dtype=np.float64)
    cur_gtype = int(model.geom_type[pad_gid])
    if (
        np.array_equal(cur_pos, cache.pos_snapshot)
        and np.array_equal(cur_quat, cache.quat_snapshot)
        and np.array_equal(cur_gpos, cache.geom_pos_snapshot)
        and np.array_equal(cur_gquat, cache.geom_quat_snapshot)
        and np.array_equal(cur_gsize, cache.geom_size_snapshot)
        and cur_gtype == cache.geom_type_snapshot
    ):
        return cache
    return _build_rigid_pad_cache(model, pad_gid, site_ids)


def _solve_2d_tilt(
    lp: np.ndarray, u0: np.ndarray, u1: np.ndarray, tol_u: float
) -> tuple[np.ndarray, float]:
    if not np.isfinite(float(lp.max())):
        return np.full(len(lp), np.nan, dtype=np.float64), float("inf")
    l0, l1 = 0.0, 0.0

    def eval_dual(a0: float, a1: float) -> tuple[float, np.ndarray]:
        z = lp + a0 * u0 + a1 * u1
        z_max = float(z.max())
        if not np.isfinite(z_max):
            return float("inf"), np.full(len(lp), np.nan, dtype=np.float64)
        exp_z = np.exp(z - z_max)
        s = float(exp_z.sum())
        if s <= 0.0 or not np.isfinite(s):
            return float("inf"), np.full(len(lp), np.nan, dtype=np.float64)
        return z_max + float(np.log(s)), exp_z / s

    phi, w_sub = eval_dual(l0, l1)
    if not np.isfinite(phi):
        return w_sub, float("inf")
    for _ in range(40):
        g0 = float(w_sub @ u0)
        g1 = float(w_sub @ u1)
        if not (np.isfinite(g0) and np.isfinite(g1)):
            return w_sub, float("inf")
        if g0 * g0 + g1 * g1 <= tol_u * tol_u:
            return w_sub, float(np.hypot(g0, g1))
        d0 = u0 - g0
        d1 = u1 - g1
        wd0 = w_sub * d0
        h00 = float(wd0 @ d0)
        h01 = float(wd0 @ d1)
        h11 = float((w_sub * d1) @ d1)
        tr = h00 + h11
        if tr <= 1e-30:
            break
        reg = max(1e-12 * tr, 1e-24)
        a = h00 + reg
        c = h11 + reg
        det = a * c - h01 * h01
        if det <= 1e-36:
            break
        s0 = -(c * g0 - h01 * g1) / det
        s1 = -(-h01 * g0 + a * g1) / det
        snorm = float(np.hypot(s0, s1))
        if snorm > 200.0:
            s0 *= 200.0 / snorm
            s1 *= 200.0 / snorm
        slope = g0 * s0 + g1 * s1
        if slope >= 0.0:
            break
        alpha = 1.0
        accepted = False
        for _ in range(20):
            phi_new, w_new = eval_dual(l0 + alpha * s0, l1 + alpha * s1)
            if np.isfinite(phi_new) and phi_new <= phi + 1e-4 * alpha * slope:
                l0 += alpha * s0
                l1 += alpha * s1
                phi, w_sub = phi_new, w_new
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    err = float(np.hypot(float(w_sub @ u0), float(w_sub @ u1)))
    return w_sub, err if np.isfinite(err) else float("inf")


def _ray_hull_fallback(
    weights: np.ndarray, xy: np.ndarray, target: np.ndarray, hull_geom: _HullGeometry
) -> np.ndarray:
    """Exact nonnegative convex-combination fallback matching target on the hull."""
    vertices = hull_geom.vertices
    edges = hull_geom.edges
    edge_sq = hull_geom.edge_sq
    inward = hull_geom.inward_unit
    mean = weights @ xy
    direction = target - mean
    if float(np.linalg.norm(direction)) <= 1e-12:
        return weights
    if hull_geom.is_collinear:
        hit = target
        blend = 1.0
    else:
        approach = -(inward @ direction)
        mean_dist = np.maximum(np.einsum("ij,ij->i", mean - vertices, inward), 0.0)
        steps = np.full(len(vertices), np.inf)
        np.divide(mean_dist, approach, out=steps, where=approach > 0.0)
        step = max(1.0, float(np.min(steps)))
        blend = 1.0 / step
        hit = mean + step * direction
    hit_rel = hit - vertices
    fractions = np.clip(np.einsum("ij,ij->i", hit_rel, edges) / edge_sq, 0.0, 1.0)
    residual = hit_rel - fractions[:, None] * edges
    edge_idx = int(np.argmin(np.einsum("ij,ij->i", residual, residual)))
    endpoint_fraction = float(fractions[edge_idx])
    matched = (1.0 - blend) * weights
    matched[hull_geom.seg_starts[edge_idx]] += blend * (1.0 - endpoint_fraction)
    matched[hull_geom.seg_ends[edge_idx]] += blend * endpoint_fraction
    return matched


def _match_pad_moment(
    log_prior: np.ndarray,
    xy: np.ndarray,
    target: np.ndarray,
    hull_geom: _HullGeometry,
) -> np.ndarray:
    """Compute local nonnegative KL-tilted weights matching target (or its hull projection)."""
    if len(hull_geom.seg_starts) == 0:
        z_max = float(log_prior.max())
        if not np.isfinite(z_max):
            return np.full(len(xy), 1.0 / len(xy), dtype=np.float64)
        exp_z = np.exp(log_prior - z_max)
        return exp_z / float(exp_z.sum())

    vertices = hull_geom.vertices
    edges = hull_geom.edges
    edge_sq = hull_geom.edge_sq
    inward_unit = hull_geom.inward_unit
    scale = hull_geom.scale

    target_rel = target - vertices
    signed = np.einsum("ij,ij->i", target_rel, inward_unit)
    if hull_geom.is_collinear or np.any(signed < 0.0):
        fractions = np.clip(np.einsum("ij,ij->i", target_rel, edges) / edge_sq, 0.0, 1.0)
        residual = target_rel - fractions[:, None] * edges
        edge_idx = int(np.argmin(np.einsum("ij,ij->i", residual, residual)))
        target = vertices[edge_idx] + float(fractions[edge_idx]) * edges[edge_idx]
        signed = np.einsum("ij,ij->i", target - vertices, inward_unit)

    active_mask = np.ones(len(xy), dtype=bool)
    if not hull_geom.is_collinear:
        face_tol = 1e-11 * scale
        active_edges = np.flatnonzero(signed <= face_tol)
        if active_edges.size:
            for e_idx in active_edges:
                d_face = (xy - vertices[e_idx]) @ inward_unit[e_idx]
                active_mask &= d_face <= face_tol
            if not np.any(active_mask):
                active_mask[:] = True

    active_idx = np.flatnonzero(active_mask)
    if len(active_idx) == 1:
        matched = np.zeros(len(xy), dtype=np.float64)
        matched[active_idx[0]] = 1.0
        return matched

    u = (xy[active_idx] - target) / scale
    u0, u1 = u[:, 0], u[:, 1]
    u_sq = u0 * u0 + u1 * u1
    tol_u = 1e-11 / scale
    retry_tol_u = 1e-8 / scale

    w_sub, err_u = _solve_2d_tilt(log_prior[active_idx], u0, u1, tol_u)
    if not (np.isfinite(err_u) and err_u <= retry_tol_u and bool(np.all(np.isfinite(w_sub)))):
        # Retry with a well-conditioned local Gaussian prior that cannot underflow,
        # then a uniform maximum-entropy prior if the local solve still stops early.
        for fallback_lp in (-20.0 * u_sq, np.zeros_like(u0)):
            cand_w, cand_err = _solve_2d_tilt(fallback_lp, u0, u1, tol_u)
            if np.isfinite(cand_err) and bool(np.all(np.isfinite(cand_w))):
                if not np.isfinite(err_u) or cand_err < err_u:
                    w_sub, err_u = cand_w, cand_err
                if err_u <= retry_tol_u:
                    break

    matched = np.zeros(len(xy), dtype=np.float64)
    if np.isfinite(err_u) and bool(np.all(np.isfinite(w_sub))):
        matched[active_idx] = w_sub
    else:
        matched[active_idx] = 1.0 / len(active_idx)
    if not (np.isfinite(err_u) and err_u <= 1e-7 / scale):
        matched = _ray_hull_fallback(matched, xy, target, hull_geom)
    return matched


class TactileForceReader:
    """统一的 MuJoCo 指尖三轴触觉力读取器，兼容 ``contact_geom`` 与 ``pad_sites``。

    ``read(data)`` 按侧返回形状为 ``(N, 3)`` 的浮点数组（列顺序 ``[Fx, Fy, Fz]``，单位 N，表达在各测点
    ``site`` 局部坐标系下），测点顺序与 ``layout[side]["taxels"]`` 一致，无触觉的一侧返回 ``None``。

    对 ``pad_sites``，基于垫块碰撞体自身坐标帧仅统计作用于前感应面的接触（排除垫块侧面与背面接触），
    以高斯权重为先验，在前感应面切平面内通过局部非负指数倾斜（含病态先验安全回退）校正权重质心。
    凸包内重构投影压力中心及平面力臂产生的一阶力矩，凸包外将接触点投影至最近的凸包边界。
    权重非负且和为一，前感应面三维合力在同一坐标系内守恒（至浮点精度）。
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
        target_geoms: str
        | int
        | np.integer[Any]
        | Sequence[str | int | np.integer[Any]]
        | None = None,
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
        self._rigid_pads: dict[str, _RigidPadCache | None] = {"left": None, "right": None}
        self._geom_owners: dict[int, list[str]] = {}
        self._left_sensor = left_sensor
        self._right_sensor = right_sensor
        self._prefix = prefix
        self._target_geoms_arg = target_geoms
        self._wrench_readers: dict[str, TactileWrenchReader] = {}
        for side in ("left", "right"):
            side_layout = self._layout[side]
            if side_layout is None:
                self._sides[side] = None
                continue
            mode = side_layout["mode"]
            grid_indices = _grid_indices(side_layout) if mode == "contact_geom" else None
            if mode == "pad_wrench":
                site_ids = np.array(
                    [self._require_id(model, _mj.mjtObj.mjOBJ_SITE, side_layout["site_name"])],
                    dtype=np.intp,
                )
            else:
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
                if int(model.geom_type[pad_gid]) == int(_mj.mjtGeom.mjGEOM_BOX):
                    _require_box_pad_nativeccd(model)
                self._geom_owners.setdefault(pad_gid, []).append(side)
                self._sides[side] = _SideSpec(
                    mode=mode,
                    site_ids=site_ids,
                    grid_indices=None,
                    geom_to_index={},
                    pad_geom_id=pad_gid,
                )
                self._rigid_pads[side] = _build_rigid_pad_cache(model, pad_gid, site_ids)
            elif mode == "pad_wrench":
                pad_gid = self._require_id(
                    model, _mj.mjtObj.mjOBJ_GEOM, side_layout["pad_geom_name"]
                )
                self._geom_owners.setdefault(pad_gid, []).append(side)
                self._sides[side] = _SideSpec(
                    mode=mode,
                    site_ids=site_ids,
                    grid_indices=None,
                    geom_to_index={pad_gid: 0},
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
        cls,
        model: MjModel,
        target_geoms: str | int | np.integer[Any] | Sequence[str | int | np.integer[Any]],
    ) -> frozenset[int]:
        resolved: set[int] = set()
        if isinstance(target_geoms, bool):
            raise TypeError(f"target_geoms 仅支持 geom 名称或整数 ID，收到 {target_geoms!r}")
        items = (
            (target_geoms,) if isinstance(target_geoms, (str, int, np.integer)) else target_geoms
        )
        for item in items:
            if isinstance(item, str):
                resolved.add(cls._require_id(model, _mj.mjtObj.mjOBJ_GEOM, item))
            elif isinstance(item, (int, np.integer)) and not isinstance(item, bool):
                gid = int(item)
                if gid < 0 or gid >= int(model.ngeom):
                    raise ValueError(f"target_geoms 包含越界 geom ID：{gid}")
                resolved.add(gid)
            else:
                raise TypeError(f"target_geoms 仅支持 geom 名称或整数 ID，收到 {item!r}")
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
        pad_contacts: dict[
            str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]
        ] = {
            "left": [],
            "right": [],
        }
        for side, spec in self._sides.items():
            if (
                spec is not None
                and spec.mode == "pad_sites"
                and spec.pad_geom_id is not None
                and int(self._model.geom_type[spec.pad_geom_id]) == int(_mj.mjtGeom.mjGEOM_BOX)
            ):
                _require_box_pad_nativeccd(self._model)
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
                    if spec.mode in ("contact_geom", "pad_wrench"):
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
                                sign * contact_frame[0],
                                float(con.dist),
                            )
                        )

        for side, contacts in pad_contacts.items():
            if not contacts:
                continue
            spec = self._sides[side]
            assert spec is not None
            pad_gid = spec.pad_geom_id
            assert pad_gid is not None
            is_box_pad = int(self._model.geom_type[pad_gid]) == int(_mj.mjtGeom.mjGEOM_BOX)
            rigid = _refresh_rigid_pad_cache(
                self._model, pad_gid, spec.site_ids, self._rigid_pads[side]
            )
            self._rigid_pads[side] = rigid
            if rigid is not None:
                geom_pos_w = np.asarray(data.geom_xpos[pad_gid], dtype=np.float64)
                geom_rot_w = np.asarray(data.geom_xmat[pad_gid], dtype=np.float64).reshape(3, 3)
                origin = geom_pos_w + geom_rot_w @ rigid.face_origin_in_geom
                pad_rot = geom_rot_w @ rigid.pad_rot_in_geom
                normal = pad_rot[:, 2]
                local_pos = rigid.local_pos
                xy = rigid.xy
                hull_geom = rigid.hull
                has_box_front_face = is_box_pad and bool(np.any(rigid.face_origin_in_geom != 0.0))
                taxel_forces = np.zeros_like(local_pos)
                for pos, force, twist, normal_into_pad, con_dist in contacts:
                    if float(normal_into_pad @ normal) <= _FRONT_FACE_NORMAL_MIN_DOT:
                        continue
                    pos_on_pad = pos + 0.5 * con_dist * normal_into_pad
                    if is_box_pad:
                        if has_box_front_face:
                            if abs(float((pos_on_pad - origin) @ normal)) > _FRONT_FACE_PLANE_TOL_M:
                                continue
                        elif float((pos_on_pad - geom_pos_w) @ normal) > 0.0:
                            continue
                    pos_local = (pos - origin) @ pad_rot
                    delta = local_pos - pos_local
                    sq_dist = np.einsum("ij,ij->i", delta, delta)
                    distance = np.sqrt(sq_dist - sq_dist.min())
                    with np.errstate(over="ignore", under="ignore"):
                        log_prior = -0.5 * (distance / self._pad_sigma_m) ** 2
                    weights = _match_pad_moment(log_prior, xy, pos_local[:2], hull_geom)
                    taxel_forces += weights[:, None] * (force @ pad_rot)

                    twist_z = twist @ normal
                    if twist_z != 0.0:
                        with np.errstate(over="ignore", under="ignore"):
                            gaussian = np.exp(log_prior)
                        gaussian /= gaussian.sum()
                        offset = xy - gaussian @ xy
                        weighted_offset = gaussian[:, None] * offset
                        variance = np.sum(weighted_offset * offset)
                        if variance > _MIN_COUPLE_VARIANCE_M2:
                            couple = np.column_stack(
                                (-weighted_offset[:, 1], weighted_offset[:, 0])
                            )
                            taxel_forces[:, :2] += (twist_z / variance) * couple
                if not rigid.same_orientation:
                    assert rigid.rel_rot is not None
                    taxel_forces = np.einsum("nij,ni->nj", rigid.rel_rot, taxel_forces)
                result[side] = taxel_forces
                continue

            geom_pos_w = np.asarray(data.geom_xpos[pad_gid], dtype=np.float64)
            geom_rot_w = np.asarray(data.geom_xmat[pad_gid], dtype=np.float64).reshape(3, 3)
            site_pos = np.asarray(data.site_xpos[spec.site_ids], dtype=np.float64)
            site_rot = np.asarray(data.site_xmat[spec.site_ids], dtype=np.float64).reshape(-1, 3, 3)
            site_pos_in_geom = (site_pos - geom_pos_w) @ geom_rot_w
            site_rot_in_geom = np.einsum("ij,njk->nik", geom_rot_w.T, site_rot)
            pad_rot_in_geom, face_origin_in_geom = _pad_frame_in_geom(
                self._model, pad_gid, site_pos_in_geom, site_rot_in_geom
            )
            origin = geom_pos_w + geom_rot_w @ face_origin_in_geom
            pad_rot = geom_rot_w @ pad_rot_in_geom
            tangent = pad_rot[:, :2]
            normal = pad_rot[:, 2]
            xy = (site_pos - origin) @ tangent
            hull_geom = _build_hull_geometry(xy)
            has_box_front_face = is_box_pad and bool(np.any(face_origin_in_geom != 0.0))
            taxel_forces = np.zeros_like(site_pos)
            for pos, force, twist, normal_into_pad, con_dist in contacts:
                if float(normal_into_pad @ normal) <= _FRONT_FACE_NORMAL_MIN_DOT:
                    continue
                pos_on_pad = pos + 0.5 * con_dist * normal_into_pad
                if is_box_pad:
                    if has_box_front_face:
                        if abs(float((pos_on_pad - origin) @ normal)) > _FRONT_FACE_PLANE_TOL_M:
                            continue
                    elif float((pos_on_pad - geom_pos_w) @ normal) > 0.0:
                        continue
                pos_local = (pos - origin) @ pad_rot
                delta = site_pos - pos
                sq_dist = np.einsum("ij,ij->i", delta, delta)
                # Scaling the distance avoids squaring a tiny sigma. Overflow
                # in the exponent gives zero weight; the closest site has weight 1.
                distance = np.sqrt(sq_dist - sq_dist.min())
                with np.errstate(over="ignore", under="ignore"):
                    log_prior = -0.5 * (distance / self._pad_sigma_m) ** 2
                weights = _match_pad_moment(log_prior, xy, pos_local[:2], hull_geom)
                taxel_forces += weights[:, None] * (force @ pad_rot)

                twist_z = twist @ normal
                if twist_z != 0.0:
                    # Rotate the centered moment arms by 90 degrees to get
                    # zero net shear and a moment proportional to the variance.
                    with np.errstate(over="ignore", under="ignore"):
                        gaussian = np.exp(log_prior)
                    gaussian /= gaussian.sum()
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

    def read_wrench(
        self,
        data: MjData,
        *,
        frame: Literal["pad_center", "ft_sensor"] = "pad_center",
    ) -> dict[str, np.ndarray | None]:
        """读取当前帧左右指尖的 ``(6,)`` 六轴接触力旋量 ``[Fx, Fy, Fz, Tx, Ty, Tz]``。"""
        reader = self._wrench_readers.get(frame)
        if reader is None:
            reader = TactileWrenchReader(
                self._model,
                left_sensor=self._left_sensor,
                right_sensor=self._right_sensor,
                prefix=self._prefix,
                frame=frame,
                target_geoms=self._target_geoms_arg,
            )
            self._wrench_readers[frame] = reader
        return reader.read(data)

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


@dataclass(frozen=True, slots=True)
class _WrenchSideSpec:
    mode: str
    ref_site_id: int
    geom_ids: frozenset[int]
    box_pad_geom_id: int | None
    box_site_ids: np.ndarray | None


class TactileWrenchReader:
    """统一的 MuJoCo 指尖六轴接触力/力矩（Wrench）读取器。

    ``read(data)`` 按侧返回形状为 ``(6,)`` 的浮点数组 ``[Fx, Fy, Fz, Tx, Ty, Tz]``
    （前 3 项为力，单位 N；后 3 项为力矩，单位 N·m），表达在所选参考 ``site`` 局部坐标系下：
    - 受压时法向力 ``Fz > 0``；
    - ``Tz`` 为绕内法向的抗扭摩擦力矩（启用 ``condim=4/6`` 时直接包含原生扭转摩擦，以及多接触点的面内力偶）；
    - ``Tx, Ty`` 为偏心接触点相对于参考原点的力臂弯矩及 ``condim=6`` 滚动阻力矩；
    - 无传感的一侧（如 ``flat_pad`` 或 ``None``）返回 ``None``。

    ``frame`` 参数控制力矩参考系：
    - ``"pad_center"``（默认）：以接触面顶点/中心为参考原点，直接对齐 Limit Surface 载荷定义；
    - ``"ft_sensor"``：以 ``sensor_mount`` 底座量测中心为参考原点（要求模组声明 ``base_site_name``，
      如 ``soft_dome``），包含指尖厚度方向的切向剪切力臂弯矩。
    """

    def __init__(
        self,
        model: MjModel,
        *,
        left_sensor: str | None = "soft_dome",
        right_sensor: str | None = "soft_dome",
        prefix: str = "",
        frame: Literal["pad_center", "ft_sensor"] = "pad_center",
        target_geoms: str
        | int
        | np.integer[Any]
        | Sequence[str | int | np.integer[Any]]
        | None = None,
    ) -> None:
        if frame not in ("pad_center", "ft_sensor"):
            raise ValueError(f"frame 必须为 'pad_center' 或 'ft_sensor'，当前为 {frame!r}")
        self._model = model
        self._frame = frame
        self._layout = get_tactile_layout(left_sensor, right_sensor, prefix=prefix)
        self._target_geom_ids: frozenset[int] | None = (
            TactileForceReader._resolve_target_geoms(model, target_geoms)
            if target_geoms is not None
            else None
        )
        self._sides: dict[str, _WrenchSideSpec | None] = {}
        self._box_caches: dict[str, _RigidPadCache | None] = {"left": None, "right": None}
        self._geom_owners: dict[int, list[str]] = {}

        for side in ("left", "right"):
            side_layout = self._layout[side]
            if side_layout is None:
                self._sides[side] = None
                continue
            mode = side_layout["mode"]
            if mode == "pad_wrench":
                if frame == "ft_sensor":
                    site_name = side_layout.get("base_site_name")
                    if not site_name:
                        raise ValueError(
                            f"{side!r} 侧模组未声明底座测量系 base_site_name，请改用 frame='pad_center'"
                        )
                else:
                    site_name = side_layout["site_name"]
                ref_sid = TactileForceReader._require_id(model, _mj.mjtObj.mjOBJ_SITE, site_name)
                pad_gid = TactileForceReader._require_id(
                    model, _mj.mjtObj.mjOBJ_GEOM, side_layout["pad_geom_name"]
                )
                is_box = int(model.geom_type[pad_gid]) == int(_mj.mjtGeom.mjGEOM_BOX)
                if is_box:
                    _require_box_pad_nativeccd(model)
                self._geom_owners.setdefault(pad_gid, []).append(side)
                pad_sids = np.array([ref_sid], dtype=np.intp)
                self._sides[side] = _WrenchSideSpec(
                    mode=mode,
                    ref_site_id=ref_sid,
                    geom_ids=frozenset((pad_gid,)),
                    box_pad_geom_id=pad_gid,
                    box_site_ids=pad_sids,
                )
                self._box_caches[side] = _build_rigid_pad_cache(model, pad_gid, pad_sids)
            elif mode == "pad_sites":
                if frame == "ft_sensor":
                    raise ValueError(
                        f"{side!r} 侧模组（{mode}）未声明底座测量系，请使用 frame='pad_center'"
                    )
                pad_gid = TactileForceReader._require_id(
                    model, _mj.mjtObj.mjOBJ_GEOM, side_layout["pad_geom_name"]
                )
                center_name = f"{prefix}{side}_pad_center"
                ref_sid = int(_mj.mj_name2id(model, _mj.mjtObj.mjOBJ_SITE, center_name))
                site_ids = np.array(
                    [
                        TactileForceReader._require_id(model, _mj.mjtObj.mjOBJ_SITE, t["site_name"])
                        for t in side_layout["taxels"]
                    ],
                    dtype=np.intp,
                )
                if ref_sid < 0:
                    ref_sid = int(site_ids[len(site_ids) // 2])
                is_box = int(model.geom_type[pad_gid]) == int(_mj.mjtGeom.mjGEOM_BOX)
                if is_box:
                    _require_box_pad_nativeccd(model)
                self._geom_owners.setdefault(pad_gid, []).append(side)
                self._sides[side] = _WrenchSideSpec(
                    mode=mode,
                    ref_site_id=ref_sid,
                    geom_ids=frozenset((pad_gid,)),
                    box_pad_geom_id=pad_gid,
                    box_site_ids=site_ids,
                )
                self._box_caches[side] = _build_rigid_pad_cache(model, pad_gid, site_ids)
            elif mode == "contact_geom":
                if frame == "ft_sensor":
                    raise ValueError(
                        f"{side!r} 侧模组（{mode}）未声明底座测量系，请使用 frame='pad_center'"
                    )
                grid = _grid_indices(side_layout)
                center_idx = int(grid[grid.shape[0] // 2, grid.shape[1] // 2])
                ref_sid = TactileForceReader._require_id(
                    model, _mj.mjtObj.mjOBJ_SITE, side_layout["taxels"][center_idx]["site_name"]
                )
                gids: list[int] = []
                for taxel in side_layout["taxels"]:
                    gid = TactileForceReader._require_id(
                        model, _mj.mjtObj.mjOBJ_GEOM, taxel["geom_name"]
                    )
                    gids.append(gid)
                    self._geom_owners.setdefault(gid, []).append(side)
                self._sides[side] = _WrenchSideSpec(
                    mode=mode,
                    ref_site_id=ref_sid,
                    geom_ids=frozenset(gids),
                    box_pad_geom_id=None,
                    box_site_ids=None,
                )
            else:
                raise ValueError(f"未知触觉布局模式 {mode!r}")

    @property
    def layout(self) -> dict[str, dict[str, Any] | None]:
        """返回解析后的左右触觉布局字典。"""
        return self._layout

    @property
    def frame(self) -> str:
        """返回当前选用的力矩参考系名称（``"pad_center"`` 或 ``"ft_sensor"``）。"""
        return self._frame

    def site_id(self, side: str) -> int | None:
        """返回指定侧（``"left"`` 或 ``"right"``）的六轴参考 ``site`` ID；无传感时返回 ``None``。"""
        if side not in self._sides:
            raise ValueError(f"side 必须为 'left' 或 'right'，当前为 {side!r}")
        spec = self._sides[side]
        return None if spec is None else int(spec.ref_site_id)

    def _accept_box_front_contact(
        self,
        side: str,
        spec: _WrenchSideSpec,
        data: MjData,
        pos: np.ndarray,
        normal_into_pad: np.ndarray,
        con_dist: float,
    ) -> bool:
        pad_gid = spec.box_pad_geom_id
        if pad_gid is None or int(self._model.geom_type[pad_gid]) != int(_mj.mjtGeom.mjGEOM_BOX):
            return True
        _require_box_pad_nativeccd(self._model)
        geom_pos_w = np.asarray(data.geom_xpos[pad_gid], dtype=np.float64)
        geom_rot_w = np.asarray(data.geom_xmat[pad_gid], dtype=np.float64).reshape(3, 3)
        cache = self._box_caches[side]
        if cache is not None:
            pad_rot_in_geom = cache.pad_rot_in_geom
            face_origin_in_geom = cache.face_origin_in_geom
        else:
            assert spec.box_site_ids is not None
            site_pos = np.asarray(data.site_xpos[spec.box_site_ids], dtype=np.float64)
            site_rot = np.asarray(data.site_xmat[spec.box_site_ids], dtype=np.float64).reshape(
                -1, 3, 3
            )
            pad_rot_in_geom, face_origin_in_geom = _pad_frame_in_geom(
                self._model,
                pad_gid,
                (site_pos - geom_pos_w) @ geom_rot_w,
                np.einsum("ij,njk->nik", geom_rot_w.T, site_rot),
            )
        origin = geom_pos_w + geom_rot_w @ face_origin_in_geom
        normal = (geom_rot_w @ pad_rot_in_geom)[:, 2]
        if float(normal_into_pad @ normal) <= _FRONT_FACE_NORMAL_MIN_DOT:
            return False
        pos_on_pad = pos + 0.5 * con_dist * normal_into_pad
        if bool(np.any(face_origin_in_geom != 0.0)):
            return abs(float((pos_on_pad - origin) @ normal)) <= _FRONT_FACE_PLANE_TOL_M
        return float((pos_on_pad - geom_pos_w) @ normal) <= 0.0

    def read(self, data: MjData) -> dict[str, np.ndarray | None]:
        """读取当前帧左右指尖的 ``(6,)`` 局部接触力旋量 ``[Fx, Fy, Fz, Tx, Ty, Tz]``。"""
        result: dict[str, np.ndarray | None] = {}
        world_forces: dict[str, np.ndarray] = {}
        world_moments: dict[str, np.ndarray] = {}
        ref_positions: dict[str, np.ndarray] = {}
        ref_rotations: dict[str, np.ndarray] = {}

        for side, spec in self._sides.items():
            if spec is None:
                result[side] = None
                continue
            if spec.box_pad_geom_id is not None and int(
                self._model.geom_type[spec.box_pad_geom_id]
            ) == int(_mj.mjtGeom.mjGEOM_BOX):
                _require_box_pad_nativeccd(self._model)
                assert spec.box_site_ids is not None
                self._box_caches[side] = _refresh_rigid_pad_cache(
                    self._model,
                    spec.box_pad_geom_id,
                    spec.box_site_ids,
                    self._box_caches[side],
                )
            result[side] = np.zeros(6, dtype=np.float64)
            world_forces[side] = np.zeros(3, dtype=np.float64)
            world_moments[side] = np.zeros(3, dtype=np.float64)
            ref_positions[side] = np.asarray(data.site_xpos[spec.ref_site_id], dtype=np.float64)
            ref_rotations[side] = np.asarray(
                data.site_xmat[spec.ref_site_id], dtype=np.float64
            ).reshape(3, 3)

        cfrc = np.zeros(6, dtype=np.float64)
        for cid in range(int(data.ncon)):
            con = data.contact[cid]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 not in self._geom_owners and g2 not in self._geom_owners:
                continue
            cfrc.fill(0.0)
            _mj.mj_contactForce(self._model, data, cid, cfrc)
            if not np.any(cfrc):
                continue
            contact_frame = np.asarray(con.frame, dtype=np.float64).reshape(3, 3)
            f_on_g2_world = contact_frame.T @ cfrc[:3]
            tau_on_g2_world = contact_frame.T @ cfrc[3:]
            pos = np.asarray(con.pos, dtype=np.float64)
            for sensor_gid, other_gid, sign in ((g1, g2, -1.0), (g2, g1, 1.0)):
                owners = self._geom_owners.get(sensor_gid)
                if not owners:
                    continue
                if self._target_geom_ids is not None and other_gid not in self._target_geom_ids:
                    continue
                f_w = sign * f_on_g2_world
                tau_w = sign * tau_on_g2_world
                for side in owners:
                    spec = self._sides[side]
                    assert spec is not None
                    if not self._accept_box_front_contact(
                        side,
                        spec,
                        data,
                        pos,
                        sign * contact_frame[0],
                        float(con.dist),
                    ):
                        continue
                    world_forces[side] += f_w
                    world_moments[side] += tau_w + np.cross(pos - ref_positions[side], f_w)

        for side, spec in self._sides.items():
            if spec is None:
                continue
            rot = ref_rotations[side]
            out = result[side]
            assert out is not None
            out[:3] = rot.T @ world_forces[side]
            out[3:] = rot.T @ world_moments[side]
        return result
