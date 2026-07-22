from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def fill_nodata(dsm: np.ndarray, nodata_mask: np.ndarray | None = None) -> np.ndarray:
    if nodata_mask is None:
        nodata_mask = np.isnan(dsm)
    if not nodata_mask.any():
        return dsm.copy()

    valid_mask = ~nodata_mask
    _, indices = ndi.distance_transform_edt(~valid_mask, return_indices=True)
    filled = dsm[tuple(indices)]
    out = dsm.copy()
    out[nodata_mask] = filled[nodata_mask]
    return out


def denoise_dsm(dsm: np.ndarray, size: int = 3) -> np.ndarray:
    return ndi.median_filter(dsm, size=size)


def align_to_grid(array: np.ndarray, target_shape: tuple[int, int], order: int = 1) -> np.ndarray:
    if array.shape[:2] == target_shape:
        return array
    zoom_factors = [target_shape[0] / array.shape[0], target_shape[1] / array.shape[1]]
    if array.ndim == 3:
        zoom_factors.append(1.0)
    return ndi.zoom(array, zoom_factors, order=order)


def validate_grids(dsm: np.ndarray, ortho: np.ndarray | None) -> None:
    if ortho is not None and ortho.shape[:2] != dsm.shape:
        raise ValueError(
            f"Сетки ЦМП {dsm.shape} и ортофото {ortho.shape[:2]} не совпадают. "
            "Используйте align_to_grid() или переприведите растры на этапе "
            "фотограмметрической обработки (единый экспорт-экстент)."
        )
