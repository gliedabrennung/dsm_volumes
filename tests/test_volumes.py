import numpy as np

from dsm_volumes.volumes import compute_object_volume


def test_paraboloid_fill_volume_matches_analytic_formula():
    n, pixel_size, R, H = 200, 0.1, 8.0, 3.0
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = n / 2
    X, Y = xx * pixel_size, yy * pixel_size
    r2 = (X - cx * pixel_size) ** 2 + (Y - cy * pixel_size) ** 2
    dsm = np.where(r2 <= R**2, H * (1 - r2 / R**2), 0.0)
    datum = np.zeros_like(dsm)
    mask = r2 <= R**2

    result = compute_object_volume(dsm, datum, mask, pixel_size**2, object_id=1, kind="fill")
    true_volume = 0.5 * np.pi * H * R**2

    assert abs(result.fill_volume_m3 - true_volume) / true_volume < 0.02
    assert result.cut_volume_m3 == 0.0


def test_paraboloid_cut_volume_matches_analytic_formula():
    n, pixel_size, R, D = 200, 0.1, 8.0, 2.5
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = n / 2
    X, Y = xx * pixel_size, yy * pixel_size
    r2 = (X - cx * pixel_size) ** 2 + (Y - cy * pixel_size) ** 2
    dsm = np.where(r2 <= R**2, -D * (1 - r2 / R**2), 0.0)
    datum = np.zeros_like(dsm)
    mask = r2 <= R**2

    result = compute_object_volume(dsm, datum, mask, pixel_size**2, object_id=2, kind="cut")
    true_volume = 0.5 * np.pi * D * R**2

    assert abs(result.cut_volume_m3 - true_volume) / true_volume < 0.02
    assert result.fill_volume_m3 == 0.0


def test_net_volume_is_fill_minus_cut():
    dsm = np.array([[1.0, -1.0], [0.5, -0.5]])
    datum = np.zeros((2, 2))
    mask = np.ones((2, 2), dtype=bool)
    result = compute_object_volume(dsm, datum, mask, pixel_area_m2=1.0, object_id=3, kind="fill")
    assert result.net_volume_m3 == result.fill_volume_m3 - result.cut_volume_m3
    assert result.fill_volume_m3 == 1.5
    assert result.cut_volume_m3 == 1.5
