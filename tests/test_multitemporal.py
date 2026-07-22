import numpy as np

from dsm_volumes.multitemporal import compute_change_volume, compute_dod


def test_compute_dod_is_simple_difference():
    dsm_t0 = np.array([[10.0, 10.0], [10.0, 10.0]])
    dsm_t1 = np.array([[12.0, 8.0], [10.0, 11.0]])
    dod = compute_dod(dsm_t0, dsm_t1)
    np.testing.assert_array_almost_equal(dod, np.array([[2.0, -2.0], [0.0, 1.0]]))


def test_compute_change_volume_separates_fill_and_cut():
    dod = np.array([[2.0, -2.0], [0.0, 1.0]])
    result = compute_change_volume(dod, pixel_size_m=1.0, label="test")
    assert result.fill_volume_m3 == 3.0
    assert result.cut_volume_m3 == 2.0
    assert result.net_volume_m3 == 1.0


def test_compute_change_volume_respects_mask():
    dod = np.array([[2.0, -2.0], [0.0, 1.0]])
    mask = np.array([[True, False], [False, True]])
    result = compute_change_volume(dod, pixel_size_m=1.0, mask=mask, label="masked")
    assert result.fill_volume_m3 == 3.0
    assert result.cut_volume_m3 == 0.0
    assert result.area_m2 == 2.0
