import numpy as np

from dsm_volumes.preprocessing import align_to_grid, fill_nodata, denoise_dsm, validate_grids


def test_fill_nodata_fills_with_nearest_valid_value():
    dsm = np.array([[1.0, np.nan, 3.0], [1.0, 2.0, 3.0]])
    filled = fill_nodata(dsm)
    assert not np.isnan(filled).any()
    assert filled[0, 1] in (1.0, 2.0, 3.0)


def test_fill_nodata_no_op_when_no_gaps():
    dsm = np.array([[1.0, 2.0], [3.0, 4.0]])
    filled = fill_nodata(dsm)
    np.testing.assert_array_equal(dsm, filled)


def test_denoise_preserves_shape_and_removes_spike():
    dsm = np.ones((9, 9)) * 10.0
    dsm[4, 4] = 100.0
    denoised = denoise_dsm(dsm, size=3)
    assert denoised.shape == dsm.shape
    assert denoised[4, 4] < 20.0


def test_align_to_grid_resizes():
    small = np.random.rand(10, 10, 3)
    resized = align_to_grid(small, (20, 20))
    assert resized.shape[:2] == (20, 20)


def test_validate_grids_raises_on_mismatch():
    dsm = np.zeros((10, 10))
    ortho = np.zeros((5, 5, 3))
    try:
        validate_grids(dsm, ortho)
        assert False, "ожидалось исключение ValueError"
    except ValueError:
        pass
