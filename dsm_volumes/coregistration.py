from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation


@dataclass
class CoregistrationResult:
    shift_row_px: float
    shift_col_px: float
    error: float
    dsm_aligned: np.ndarray


def estimate_shift(
    dsm_reference: np.ndarray, dsm_moving: np.ndarray, upsample_factor: int = 20
) -> tuple[tuple[float, float], float]:
    a = np.nan_to_num(dsm_reference, nan=float(np.nanmean(dsm_reference)))
    b = np.nan_to_num(dsm_moving, nan=float(np.nanmean(dsm_moving)))
    shift, error, _phase_diff = phase_cross_correlation(a, b, upsample_factor=upsample_factor)
    return (float(shift[0]), float(shift[1])), float(error)


def apply_shift(dsm: np.ndarray, shift: tuple[float, float], order: int = 3) -> np.ndarray:
    return ndi_shift(dsm, shift, order=order, mode="nearest")


def coregister(
    dsm_reference: np.ndarray, dsm_moving: np.ndarray, upsample_factor: int = 20
) -> CoregistrationResult:
    shift, error = estimate_shift(dsm_reference, dsm_moving, upsample_factor=upsample_factor)
    aligned = apply_shift(dsm_moving, shift)
    return CoregistrationResult(shift_row_px=shift[0], shift_col_px=shift[1], error=error, dsm_aligned=aligned)
