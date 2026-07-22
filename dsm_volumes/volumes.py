from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segmentation import ObjectCandidate


@dataclass
class VolumeResult:
    object_id: int
    kind: str
    area_m2: float
    cut_volume_m3: float
    fill_volume_m3: float
    net_volume_m3: float
    mean_height_m: float
    max_height_m: float
    min_height_m: float
    centroid_row: float
    centroid_col: float


def compute_object_volume(
    dsm_local: np.ndarray,
    datum_local: np.ndarray,
    mask_local: np.ndarray,
    pixel_area_m2: float,
    object_id: int,
    kind: str,
    window_offset: tuple[int, int] = (0, 0),
) -> VolumeResult:
    diff = (dsm_local - datum_local)[mask_local]
    fill_part = diff[diff > 0]
    cut_part = diff[diff < 0]

    fill_volume = float(fill_part.sum() * pixel_area_m2) + 0.0
    cut_volume = float(-cut_part.sum() * pixel_area_m2) + 0.0

    ys, xs = np.nonzero(mask_local)
    centroid_row = float(ys.mean()) + window_offset[0] if len(ys) else 0.0
    centroid_col = float(xs.mean()) + window_offset[1] if len(xs) else 0.0

    return VolumeResult(
        object_id=object_id,
        kind=kind,
        area_m2=float(mask_local.sum() * pixel_area_m2),
        cut_volume_m3=cut_volume,
        fill_volume_m3=fill_volume,
        net_volume_m3=fill_volume - cut_volume,
        mean_height_m=float(diff.mean()) if len(diff) else 0.0,
        max_height_m=float(diff.max()) if len(diff) else 0.0,
        min_height_m=float(diff.min()) if len(diff) else 0.0,
        centroid_row=centroid_row,
        centroid_col=centroid_col,
    )


def compute_all_volumes(
    dsm: np.ndarray,
    candidates: list[ObjectCandidate],
    datum_by_id: dict[int, tuple[np.ndarray, tuple[slice, slice]]],
    pixel_size_m: float,
) -> list[VolumeResult]:
    pixel_area = pixel_size_m**2
    results = []
    for c in candidates:
        if c.id not in datum_by_id:
            continue
        datum_local, window = datum_by_id[c.id]
        dsm_local = dsm[window]
        mask_local = c.mask[window]
        offset = (window[0].start, window[1].start)
        results.append(
            compute_object_volume(dsm_local, datum_local, mask_local, pixel_area, c.id, c.kind, offset)
        )
    return results
