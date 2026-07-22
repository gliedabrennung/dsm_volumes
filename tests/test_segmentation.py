import numpy as np

from dsm_volumes.config import PipelineConfig
from dsm_volumes.segmentation import (
    clean_binary_mask,
    detect_candidate_masks,
    fit_trend_surface,
    label_and_split,
    segment_objects,
)


def _flat_plane_with_bump(n=120, pixel_size=0.15, bump_h=2.0, bump_r_px=15):
    yy, xx = np.mgrid[0:n, 0:n]
    dsm = 50.0 + 0.02 * xx
    cy, cx = n // 2, n // 2
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    dsm = dsm + np.where(r2 <= bump_r_px**2, bump_h * (1 - r2 / bump_r_px**2), 0.0)
    return dsm


def test_fit_trend_surface_recovers_plane_ignoring_bump():
    dsm = _flat_plane_with_bump()
    trend = fit_trend_surface(dsm, degree=2, iterations=3, outlier_sigma=1.5)
    background = dsm.copy()
    n = dsm.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    r2 = (yy - n // 2) ** 2 + (xx - n // 2) ** 2
    outside_bump = r2 > (20**2)
    assert np.abs((trend - background)[outside_bump]).mean() < 0.05


def test_detect_candidate_masks_finds_bump():
    dsm = _flat_plane_with_bump()
    trend = fit_trend_surface(dsm)
    config = PipelineConfig(pixel_size_m=0.15, fill_threshold_m=0.2, cut_threshold_m=0.2)
    fill_mask, cut_mask, _ = detect_candidate_masks(dsm, trend, config)
    assert fill_mask.sum() > 100
    assert cut_mask.sum() < fill_mask.sum()


def test_clean_binary_mask_removes_pixel_speckle():
    config = PipelineConfig(pixel_size_m=0.15, min_object_area_m2=15.0, morphology_cleanup_radius_px=2)
    mask = np.zeros((100, 100), dtype=bool)
    mask[10, 10] = True
    mask[40:70, 40:70] = True
    cleaned = clean_binary_mask(mask, config)
    assert not cleaned[10, 10]
    assert cleaned[55, 55]


def test_label_and_split_separates_two_touching_blobs():
    config = PipelineConfig(pixel_size_m=0.15, watershed_min_distance_px=10)
    mask = np.zeros((60, 140), dtype=bool)
    mask[10:50, 10:50] = True
    mask[10:50, 48:90] = True
    labels = label_and_split(mask, np.zeros_like(mask, dtype=float), config)
    assert labels.max() >= 2


def test_segment_objects_end_to_end_on_synthetic_bump():
    dsm = _flat_plane_with_bump(bump_h=2.5, bump_r_px=20)
    config = PipelineConfig(pixel_size_m=0.15, fill_threshold_m=0.2, cut_threshold_m=0.2, min_object_area_m2=5.0)
    candidates, trend, residual = segment_objects(dsm, config)
    assert len(candidates) >= 1
    assert candidates[0].kind in ("fill", "cut")
