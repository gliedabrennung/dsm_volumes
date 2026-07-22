import numpy as np

from dsm_volumes.preprocessing import denoise_dsm
from dsm_volumes.segmentation import fit_trend_surface
from dsm_volumes.tiling import fit_trend_surface_large, iter_tiles, process_in_tiles


def test_iter_tiles_covers_whole_raster_without_gaps_or_overlap_in_core():
    shape = (250, 130)
    covered = np.zeros(shape, dtype=int)
    for tile in iter_tiles(shape, tile_size=64, overlap=8):
        covered[tile.core] += 1
    assert (covered == 1).all()


def test_process_in_tiles_matches_direct_processing():
    rng = np.random.default_rng(0)
    array = rng.normal(10, 1, size=(300, 300))
    direct = denoise_dsm(array, size=3)
    tiled = process_in_tiles(array, denoise_dsm, tile_size=100, overlap=16, n_jobs=1, size=3)
    np.testing.assert_allclose(direct, tiled, atol=1e-9)


def test_fit_trend_surface_large_matches_dense_version():
    n = 200
    yy, xx = np.mgrid[0:n, 0:n]
    dsm = 100.0 + 0.03 * xx + 0.4 * np.sin(xx / 20.0)
    dense = fit_trend_surface(dsm, degree=2, iterations=2, outlier_sigma=2.0)
    tiled = fit_trend_surface_large(dsm, degree=2, iterations=2, outlier_sigma=2.0, tile_size=64)
    np.testing.assert_allclose(dense, tiled, atol=1e-6)
