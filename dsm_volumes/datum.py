from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import griddata
from scipy.spatial import QhullError

from .config import PipelineConfig
from .segmentation import ObjectCandidate


def get_local_window(shape: tuple[int, int], slice_: tuple[slice, slice], pad_px: int) -> tuple[slice, slice]:
    rs, cs = slice_
    r0 = max(0, rs.start - pad_px)
    r1 = min(shape[0], rs.stop + pad_px)
    c0 = max(0, cs.start - pad_px)
    c1 = min(shape[1], cs.stop + pad_px)
    return slice(r0, r1), slice(c0, c1)


def _fit_plane(shape: tuple[int, int], ys_idx: np.ndarray, xs_idx: np.ndarray, zs: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones_like(xs_idx, dtype=float), xs_idx.astype(float), ys_idx.astype(float)])
    coeffs, *_ = np.linalg.lstsq(A, zs, rcond=None)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return coeffs[0] + coeffs[1] * xx + coeffs[2] * yy


def _fit_plane_safe(shape: tuple[int, int], ys_idx: np.ndarray, xs_idx: np.ndarray, zs: np.ndarray) -> np.ndarray:
    try:
        return _fit_plane(shape, ys_idx, xs_idx, zs)
    except np.linalg.LinAlgError:
        return np.full(shape, float(np.mean(zs)) if len(zs) else np.nan, dtype=np.float64)


def _fit_tin(shape: tuple[int, int], ys_idx: np.ndarray, xs_idx: np.ndarray, zs: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    points = np.column_stack([xs_idx, ys_idx])
    grid = griddata(points, zs, (xx, yy), method="linear")
    if np.isnan(grid).any():
        grid_nn = griddata(points, zs, (xx, yy), method="nearest")
        grid = np.where(np.isnan(grid), grid_nn, grid)
    return grid


def compute_datum_for_object(
    dsm: np.ndarray,
    all_objects_mask: np.ndarray,
    candidate: ObjectCandidate,
    config: PipelineConfig,
    method: str | None = None,
) -> tuple[np.ndarray, tuple[slice, slice]]:
    method = method or config.datum_method
    safety_px = max(0, int(round(config.datum_safety_margin_m / config.pixel_size_m)))
    collar_px = max(1, int(round(config.collar_buffer_m / config.pixel_size_m)))
    window = get_local_window(dsm.shape, candidate.slice_, safety_px + collar_px + 8)

    dsm_local = dsm[window]
    mask_local = candidate.mask[window]
    other_objects_local = all_objects_mask[window] & ~mask_local

    excluded = ndi.binary_dilation(mask_local, iterations=safety_px) if safety_px > 0 else mask_local
    dilated = ndi.binary_dilation(excluded, iterations=collar_px)
    ring = dilated & ~excluded & ~other_objects_local

    if ring.sum() < 8:
        dilated_wide = ndi.binary_dilation(excluded, iterations=collar_px * 3)
        ring = dilated_wide & ~excluded & ~other_objects_local

    ys_idx, xs_idx = np.nonzero(ring)
    zs = dsm_local[ys_idx, xs_idx]
    valid = np.isfinite(zs)
    ys_idx, xs_idx, zs = ys_idx[valid], xs_idx[valid], zs[valid]

    if len(zs) < 4 or method == "constant":
        fallback_value = float(np.mean(zs)) if len(zs) else float(np.nanmean(dsm_local))
        datum_local = np.full(dsm_local.shape, fallback_value, dtype=np.float64)
    elif method == "plane":
        datum_local = _fit_plane_safe(dsm_local.shape, ys_idx, xs_idx, zs)
    elif method == "tin":
        try:
            datum_local = _fit_tin(dsm_local.shape, ys_idx, xs_idx, zs)
        except (QhullError, ValueError) as exc:
            warnings.warn(
                f"TIN-датум не построен для объекта (id={candidate.id}): {exc}. "
                "Откат на метод 'plane'. Обычно означает, что вокруг объекта "
                "нет достаточного количества настоящих точек фонового грунта — "
                "проверьте охват ЦМП вокруг объекта.",
                RuntimeWarning,
                stacklevel=2,
            )
            datum_local = _fit_plane_safe(dsm_local.shape, ys_idx, xs_idx, zs)
    else:
        raise ValueError(f"Неизвестный метод датума: {method!r}")

    return datum_local, window


def compute_datum_for_all(
    dsm: np.ndarray,
    candidates: list[ObjectCandidate],
    config: PipelineConfig,
    method: str | None = None,
) -> dict[int, tuple[np.ndarray, tuple[slice, slice]]]:
    if not candidates:
        return {}
    all_objects_mask = np.zeros(dsm.shape, dtype=bool)
    for c in candidates:
        all_objects_mask |= c.mask

    result = {}
    for c in candidates:
        result[c.id] = compute_datum_for_object(dsm, all_objects_mask, c, config, method=method)
    return result