from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import rgb_to_hsv
from skimage import measure

from .config import PipelineConfig
from .segmentation import ObjectCandidate


@dataclass
class NoiseClassification:
    object_id: int
    is_valid_object: bool
    reason: str
    area_m2: float
    extent: float
    machinery_color_fraction: float


def machinery_color_fraction(ortho_rgb: np.ndarray, mask: np.ndarray, config: PipelineConfig) -> float:
    if ortho_rgb is None or mask.sum() == 0:
        return 0.0
    pixels = ortho_rgb[mask].astype(np.float64) / 255.0
    hsv = rgb_to_hsv(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    hue_lo, hue_hi = config.machinery_hue_range
    hit = (
        (h >= hue_lo)
        & (h <= hue_hi)
        & (s >= config.machinery_saturation_min)
        & (v >= config.machinery_value_min)
    )
    return float(hit.mean())


def classify_candidate(
    candidate: ObjectCandidate,
    ortho_local: np.ndarray | None,
    config: PipelineConfig,
) -> NoiseClassification:
    area_m2 = candidate.area_m2(config.pixel_size_m)

    local_mask = candidate.mask[candidate.slice_]
    props = measure.regionprops(local_mask.astype(np.uint8))
    extent = float(props[0].extent) if props else 1.0

    color_fraction = 0.0
    if ortho_local is not None:
        color_fraction = machinery_color_fraction(ortho_local, local_mask, config)

    reasons = []
    is_valid = True

    if area_m2 < config.min_object_area_m2:
        is_valid = False
        reasons.append(f"площадь {area_m2:.1f} м² < мин. порога {config.min_object_area_m2:.1f} м²")

    if config.max_object_area_m2 is not None and area_m2 > config.max_object_area_m2:
        is_valid = False
        reasons.append(f"площадь {area_m2:.1f} м² > макс. порога {config.max_object_area_m2:.1f} м²")

    if color_fraction >= config.machinery_color_fraction_threshold:
        is_valid = False
        reasons.append(f"{color_fraction:.0%} площади в сигнальном цвете техники")

    if extent > config.max_extent_for_natural_pile and area_m2 < 4 * config.min_object_area_m2:
        is_valid = False
        reasons.append(f"компактный прямоугольный контур (extent={extent:.2f}) при малой площади")

    reason = "; ".join(reasons) if reasons else "прошёл все проверки"
    return NoiseClassification(
        object_id=candidate.id,
        is_valid_object=is_valid,
        reason=reason,
        area_m2=area_m2,
        extent=extent,
        machinery_color_fraction=color_fraction,
    )


def filter_candidates(
    candidates: list[ObjectCandidate],
    ortho: np.ndarray | None,
    config: PipelineConfig,
) -> tuple[list[ObjectCandidate], list[NoiseClassification]]:
    classifications = []
    valid = []
    for c in candidates:
        ortho_local = ortho[c.slice_] if ortho is not None else None
        cls = classify_candidate(c, ortho_local, config)
        classifications.append(cls)
        if cls.is_valid_object:
            valid.append(c)
    return valid, classifications
