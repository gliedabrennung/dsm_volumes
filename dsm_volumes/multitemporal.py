from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import align_to_grid


@dataclass
class ChangeVolumeResult:
    label: str
    area_m2: float
    cut_volume_m3: float
    fill_volume_m3: float
    net_volume_m3: float


def compute_dod(dsm_t0: np.ndarray, dsm_t1: np.ndarray) -> np.ndarray:
    if dsm_t0.shape != dsm_t1.shape:
        dsm_t0 = align_to_grid(dsm_t0, dsm_t1.shape)
    return dsm_t1 - dsm_t0


def compute_change_volume(
    dod: np.ndarray,
    pixel_size_m: float,
    mask: np.ndarray | None = None,
    label: str = "участок",
) -> ChangeVolumeResult:
    region = dod if mask is None else dod[mask]
    region = region[np.isfinite(region)]
    pixel_area = pixel_size_m**2

    fill_volume = float(region[region > 0].sum() * pixel_area)
    cut_volume = float(-region[region < 0].sum() * pixel_area)
    area_m2 = float(region.size * pixel_area)

    return ChangeVolumeResult(
        label=label,
        area_m2=area_m2,
        cut_volume_m3=cut_volume,
        fill_volume_m3=fill_volume,
        net_volume_m3=fill_volume - cut_volume,
    )


def compute_change_volumes_by_object(
    dod: np.ndarray,
    pixel_size_m: float,
    object_masks: dict[str, np.ndarray],
) -> list[ChangeVolumeResult]:
    return [compute_change_volume(dod, pixel_size_m, mask, label) for label, mask in object_masks.items()]
