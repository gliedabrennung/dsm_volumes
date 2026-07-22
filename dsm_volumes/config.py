from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class PipelineConfig:
    pixel_size_m: float = 0.15
    trend_poly_degree: int = 2
    trend_iterations: int = 3
    trend_outlier_sigma: float = 1.5
    fill_threshold_m: float = 0.18
    cut_threshold_m: float = 0.18
    min_object_area_m2: float = 15.0
    morphology_cleanup_radius_px: int = 2
    watershed_min_distance_px: int = 15
    machinery_hue_range: Tuple[float, float] = (0.06, 0.17)
    machinery_saturation_min: float = 0.45
    machinery_value_min: float = 0.45
    machinery_color_fraction_threshold: float = 0.3
    max_object_area_m2: Optional[float] = None
    max_extent_for_natural_pile: float = 0.88
    datum_safety_margin_m: float = 0.5
    collar_buffer_m: float = 1.5
    datum_method: str = "tin"
    tile_size_px: int = 512
    tile_overlap_px: int = 64
    n_jobs: int = 4

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = self.to_dict()
        if path.suffix in (".yml", ".yaml") and _HAS_YAML:
            path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
        else:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        text = path.read_text()
        if path.suffix in (".yml", ".yaml") and _HAS_YAML:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls(**data)
