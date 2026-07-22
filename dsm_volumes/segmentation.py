from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage import segmentation as skseg
from skimage.feature import peak_local_max

from .config import PipelineConfig


@dataclass
class ObjectCandidate:
    id: int
    kind: str
    mask: np.ndarray
    slice_: tuple
    area_px: int

    def area_m2(self, pixel_size_m: float) -> float:
        return self.area_px * pixel_size_m**2

def _poly_design_matrix(xs: np.ndarray, ys: np.ndarray, degree: int) -> np.ndarray:
    terms = [np.ones_like(xs)]
    for total_deg in range(1, degree + 1):
        for i in range(total_deg + 1):
            terms.append((xs ** (total_deg - i)) * (ys**i))
    return np.stack(terms, axis=-1)


def fit_trend_surface(
    dsm: np.ndarray,
    degree: int = 2,
    iterations: int = 3,
    outlier_sigma: float = 1.5,
) -> np.ndarray:
    ny, nx = dsm.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    scale = max(nx, ny)
    xs = (xx - nx / 2) / scale
    ys = (yy - ny / 2) / scale

    valid = np.isfinite(dsm)
    design = _poly_design_matrix(xs, ys, degree)
    design_flat = design.reshape(-1, design.shape[-1])
    dsm_flat = dsm.ravel()

    weight_mask = valid.copy()
    coeffs = None
    for _ in range(max(1, iterations)):
        idx = weight_mask.ravel()
        if idx.sum() < design.shape[-1] + 1:
            break
        coeffs, *_ = np.linalg.lstsq(design_flat[idx], dsm_flat[idx], rcond=None)
        trend_flat = design_flat @ coeffs
        resid = dsm_flat - trend_flat
        std = np.nanstd(resid[idx])
        weight_mask = valid.ravel() & (np.abs(resid) < outlier_sigma * std)
        weight_mask = weight_mask.reshape(dsm.shape)

    trend = (design_flat @ coeffs).reshape(dsm.shape).astype(dsm.dtype)
    return trend

def detect_candidate_masks(
    dsm: np.ndarray, trend: np.ndarray, config: PipelineConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = dsm - trend
    fill_raw = residual > config.fill_threshold_m
    cut_raw = residual < -config.cut_threshold_m
    return fill_raw, cut_raw, residual


def clean_binary_mask(mask: np.ndarray, config: PipelineConfig) -> np.ndarray:
    radius = config.morphology_cleanup_radius_px
    m = mask
    if radius > 0:
        footprint = morphology.disk(radius)
        m = ndi.binary_closing(m, structure=footprint)
        m = ndi.binary_opening(m, structure=footprint)

    speck_px = max(1, int(0.3 * config.min_object_area_m2 / (config.pixel_size_m**2)))
    m = morphology.remove_small_objects(m, max_size=speck_px)
    m = morphology.remove_small_holes(m, max_size=speck_px)
    return m

def label_and_split(mask: np.ndarray, height_signal: np.ndarray, config: PipelineConfig) -> np.ndarray:
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)

    distance = ndi.distance_transform_edt(mask)
    min_distance = max(1, config.watershed_min_distance_px)
    coords = peak_local_max(distance, min_distance=min_distance, labels=mask, exclude_border=False)

    peak_mask = np.zeros(distance.shape, dtype=bool)
    if len(coords):
        peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)
    if markers.max() == 0:
        # нет выраженных локальных максимумов — весь компонент одним объектом
        markers, _ = ndi.label(mask)
        return markers
    labels = skseg.watershed(-distance, markers, mask=mask)
    return labels

def segment_objects(
    dsm: np.ndarray, config: PipelineConfig
) -> tuple[list[ObjectCandidate], np.ndarray, np.ndarray]:
    trend = fit_trend_surface(
        dsm,
        degree=config.trend_poly_degree,
        iterations=config.trend_iterations,
        outlier_sigma=config.trend_outlier_sigma,
    )
    fill_raw, cut_raw, residual = detect_candidate_masks(dsm, trend, config)
    fill_clean = clean_binary_mask(fill_raw, config)
    cut_clean = clean_binary_mask(cut_raw, config)

    candidates: list[ObjectCandidate] = []
    next_id = 1
    for kind, mask in (("fill", fill_clean), ("cut", cut_clean)):
        labels = label_and_split(mask, np.abs(residual), config)
        for region in measure.regionprops(labels):
            obj_mask = labels == region.label
            rmin, cmin, rmax, cmax = region.bbox
            candidates.append(
                ObjectCandidate(
                    id=next_id,
                    kind=kind,
                    mask=obj_mask,
                    slice_=(slice(rmin, rmax), slice(cmin, cmax)),
                    area_px=int(region.area),
                )
            )
            next_id += 1
    return candidates, trend, residual
