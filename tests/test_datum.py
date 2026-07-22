import numpy as np

from dsm_volumes.config import PipelineConfig
from dsm_volumes.datum import compute_datum_for_object
from dsm_volumes.segmentation import ObjectCandidate


def _sloped_terrain_with_disk_object(n=100, pixel_size=0.15, slope=0.05, obj_h=3.0, obj_r_px=18):
    yy, xx = np.mgrid[0:n, 0:n]
    terrain = 100.0 + slope * xx * pixel_size
    cy, cx = n // 2, n // 2
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = r2 <= obj_r_px**2
    dsm = terrain.copy()
    dsm[mask] += obj_h
    return dsm, terrain, mask


def _make_candidate(mask):
    ys, xs = np.nonzero(mask)
    slice_ = (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))
    return ObjectCandidate(id=1, kind="fill", mask=mask, slice_=slice_, area_px=int(mask.sum()))


def test_tin_datum_recovers_sloped_terrain_better_than_constant():
    dsm, terrain, mask = _sloped_terrain_with_disk_object()
    config = PipelineConfig(pixel_size_m=0.15, collar_buffer_m=1.5, datum_safety_margin_m=0.3)
    candidate = _make_candidate(mask)

    datum_tin, window = compute_datum_for_object(dsm, mask, candidate, config, method="tin")
    datum_const, _ = compute_datum_for_object(dsm, mask, candidate, config, method="constant")

    true_terrain_local = terrain[window]
    mask_local = mask[window]

    err_tin = np.abs((datum_tin - true_terrain_local)[mask_local]).mean()
    err_const = np.abs((datum_const - true_terrain_local)[mask_local]).mean()

    assert err_tin < err_const
    assert err_tin < 0.05


def test_datum_safety_margin_reduces_bias_from_incomplete_mask():
    n = 100
    yy, xx = np.mgrid[0:n, 0:n]
    terrain = np.full((n, n), 100.0)
    cy, cx = n // 2, n // 2
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    true_r_px = 20
    detected_r_px = 17
    dsm = terrain.copy()
    obj_profile = np.where(r2 <= true_r_px**2, 2.0 * (1 - r2 / true_r_px**2), 0.0)
    dsm = dsm + obj_profile
    detected_mask = r2 <= detected_r_px**2
    candidate = _make_candidate(detected_mask)

    cfg_no_margin = PipelineConfig(pixel_size_m=0.15, datum_safety_margin_m=0.0, collar_buffer_m=1.5)
    cfg_with_margin = PipelineConfig(pixel_size_m=0.15, datum_safety_margin_m=0.6, collar_buffer_m=1.5)

    datum_no_margin, w0 = compute_datum_for_object(dsm, detected_mask, candidate, cfg_no_margin, method="tin")
    datum_with_margin, w1 = compute_datum_for_object(dsm, detected_mask, candidate, cfg_with_margin, method="tin")

    bias_no_margin = (datum_no_margin - terrain[w0])[detected_mask[w0]].mean()
    bias_with_margin = (datum_with_margin - terrain[w1])[detected_mask[w1]].mean()

    assert bias_with_margin < bias_no_margin
