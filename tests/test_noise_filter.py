import numpy as np

from dsm_volumes.config import PipelineConfig
from dsm_volumes.noise_filter import classify_candidate, machinery_color_fraction
from dsm_volumes.segmentation import ObjectCandidate


def _candidate_from_bbox(shape, r0, r1, c0, c1):
    mask = np.zeros(shape, dtype=bool)
    mask[r0:r1, c0:c1] = True
    return ObjectCandidate(id=1, kind="fill", mask=mask, slice_=(slice(r0, r1), slice(c0, c1)), area_px=(r1 - r0) * (c1 - c0))


def test_small_yellow_rectangle_is_classified_as_machinery():
    config = PipelineConfig(pixel_size_m=0.15, min_object_area_m2=15.0)
    shape = (60, 60)
    candidate = _candidate_from_bbox(shape, 10, 20, 10, 30)

    ortho = np.zeros((*shape, 3), dtype=np.uint8)
    ortho[:] = (140, 120, 95)
    ortho[10:20, 10:30] = (230, 175, 35)

    ortho_local = ortho[candidate.slice_]
    result = classify_candidate(candidate, ortho_local, config)

    assert not result.is_valid_object
    assert result.machinery_color_fraction > 0.9


def test_large_natural_colored_object_passes_filter():
    config = PipelineConfig(pixel_size_m=0.15, min_object_area_m2=15.0)
    shape = (150, 150)
    candidate = _candidate_from_bbox(shape, 20, 120, 20, 120)

    ortho = np.zeros((*shape, 3), dtype=np.uint8)
    ortho[:] = (140, 120, 95)
    ortho[20:120, 20:120] = (170, 165, 150)

    ortho_local = ortho[candidate.slice_]
    result = classify_candidate(candidate, ortho_local, config)

    assert result.is_valid_object


def test_machinery_color_fraction_zero_without_ortho():
    mask = np.ones((5, 5), dtype=bool)
    config = PipelineConfig()
    assert machinery_color_fraction(None, mask, config) == 0.0
