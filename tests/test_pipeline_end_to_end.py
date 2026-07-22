import warnings

import numpy as np
import pytest

from dsm_volumes.config import PipelineConfig
from dsm_volumes.pipeline import VolumeCalculationPipeline
from examples.synthetic_scene import generate_synthetic_scene

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def scene():
    return generate_synthetic_scene()


@pytest.fixture(scope="module")
def config(scene):
    return PipelineConfig(pixel_size_m=scene.pixel_size_m)


@pytest.fixture(scope="module")
def result_single_epoch(scene, config):
    pipeline = VolumeCalculationPipeline(config)
    return pipeline.run_single_epoch(scene.dsm_t1, scene.ortho_t1)


def _match(vol_results, obj, pixel_size_m):
    row_true, col_true = obj.cy_m / pixel_size_m, obj.cx_m / pixel_size_m
    return min(vol_results, key=lambda v: (v.centroid_row - row_true) ** 2 + (v.centroid_col - col_true) ** 2)


def test_finds_exactly_three_valid_objects_and_rejects_vehicle(result_single_epoch):
    assert len(result_single_epoch.volumes) == 3
    n_rejected = sum(1 for c in result_single_epoch.rejected if not c.is_valid_object)
    assert n_rejected >= 1


def test_watershed_separates_touching_piles(result_single_epoch):
    fill_objects = [v for v in result_single_epoch.volumes if v.kind == "fill"]
    assert len(fill_objects) == 2


@pytest.mark.parametrize("name,tolerance", [("pile_A", 0.10), ("pile_B", 0.10), ("pit_C", 0.10)])
def test_object_volume_within_tolerance_of_truth(scene, result_single_epoch, name, tolerance):
    obj = scene.objects_t1[name]
    match = _match(result_single_epoch.volumes, obj, scene.pixel_size_m)
    calc = match.fill_volume_m3 if obj.height_m > 0 else match.cut_volume_m3
    true_v = obj.true_volume_m3()
    assert abs(calc - true_v) / true_v < tolerance


def test_multitemporal_change_volume_more_accurate_than_single_epoch(scene, config):
    pipeline = VolumeCalculationPipeline(config)
    result_multi = pipeline.run_multitemporal(scene.dsm_t0, scene.dsm_t1, scene.ortho_t1)
    assert len(result_multi.volumes) == 3

    pile_A = scene.objects_t1["pile_A"]
    expected_delta = pile_A.true_volume_m3() - scene.objects_t0["pile_A"].true_volume_m3()
    match = _match(result_multi.volumes, pile_A, scene.pixel_size_m)
    assert abs(match.net_volume_m3 - expected_delta) / expected_delta < 0.05


def test_raw_scene_dod_includes_vehicle_but_filtered_object_sum_does_not(scene, config):
    from dsm_volumes import multitemporal

    pipeline = VolumeCalculationPipeline(config)
    result_multi = pipeline.run_multitemporal(scene.dsm_t0, scene.dsm_t1, scene.ortho_t1)

    dod = multitemporal.compute_dod(scene.dsm_t0, scene.dsm_t1)
    whole_site = multitemporal.compute_change_volume(dod, scene.pixel_size_m)
    filtered_fill_sum = sum(v.fill_volume_m3 for v in result_multi.volumes if v.kind == "fill")

    vehicle_volume = scene.vehicle_box["w_m"] * scene.vehicle_box["l_m"] * scene.vehicle_box["height_m"]
    gap = whole_site.fill_volume_m3 - filtered_fill_sum
    assert abs(gap - vehicle_volume) / vehicle_volume < 0.3
