import warnings

import numpy as np
from scipy.ndimage import shift as ndi_shift

from dsm_volumes import coregistration, multitemporal
from dsm_volumes.config import PipelineConfig
from dsm_volumes.pipeline import VolumeCalculationPipeline
from examples.synthetic_scene import generate_synthetic_scene

warnings.filterwarnings("ignore")


def test_estimate_shift_recovers_known_shift_when_scene_has_objects():
    scene = generate_synthetic_scene()
    dsm = scene.dsm_t1
    true_shift = (1.3, -2.1)
    shifted = ndi_shift(dsm, true_shift, order=3, mode="nearest")

    detected, error = coregistration.estimate_shift(dsm, shifted, upsample_factor=20)
    assert abs(detected[0] - (-true_shift[0])) < 0.2
    assert abs(detected[1] - (-true_shift[1])) < 0.2


def test_coregister_reduces_dod_bias_from_artificial_misalignment():
    scene = generate_synthetic_scene()
    true_shift = (1.2, -0.8)
    dsm_t0_misaligned = ndi_shift(scene.dsm_t0, [-s for s in true_shift], order=3, mode="nearest")

    baseline = multitemporal.compute_change_volume(
        multitemporal.compute_dod(scene.dsm_t0, scene.dsm_t1), scene.pixel_size_m
    )
    without_fix = multitemporal.compute_change_volume(
        multitemporal.compute_dod(dsm_t0_misaligned, scene.dsm_t1), scene.pixel_size_m
    )
    coreg = coregistration.coregister(dsm_t0_misaligned, scene.dsm_t1)
    with_fix = multitemporal.compute_change_volume(
        multitemporal.compute_dod(dsm_t0_misaligned, coreg.dsm_aligned), scene.pixel_size_m
    )

    err_without = abs(without_fix.fill_volume_m3 - baseline.fill_volume_m3) / baseline.fill_volume_m3
    err_with = abs(with_fix.fill_volume_m3 - baseline.fill_volume_m3) / baseline.fill_volume_m3

    assert err_with < err_without
    assert err_with < 0.02


def test_pipeline_run_multitemporal_with_auto_coregister_smoke():
    scene = generate_synthetic_scene()
    config = PipelineConfig(pixel_size_m=scene.pixel_size_m)
    pipeline = VolumeCalculationPipeline(config)

    true_shift = (0.6, 0.4)
    dsm_t0_misaligned = ndi_shift(scene.dsm_t0, [-s for s in true_shift], order=3, mode="nearest")

    result = pipeline.run_multitemporal(dsm_t0_misaligned, scene.dsm_t1, scene.ortho_t1, auto_coregister=True)
    assert result.coregistration is not None
    assert "shift_row_px" in result.coregistration
    assert len(result.volumes) == 3
