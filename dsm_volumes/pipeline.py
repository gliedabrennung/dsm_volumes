from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import multitemporal, reporting
from .config import PipelineConfig
from .coregistration import coregister
from .datum import compute_datum_for_all
from .noise_filter import filter_candidates
from .preprocessing import denoise_dsm, fill_nodata, validate_grids
from .reporting import RasterTransform
from .segmentation import ObjectCandidate, segment_objects
from .volumes import VolumeResult, compute_all_volumes


@dataclass
class PipelineResult:
    candidates: list[ObjectCandidate]
    volumes: list[VolumeResult]
    rejected: list
    trend: np.ndarray
    residual: np.ndarray
    output_files: dict[str, str] = field(default_factory=dict)
    coregistration: dict | None = None

    def summary_dict(self) -> dict:
        return {
            "n_objects": len(self.volumes),
            "n_rejected_as_noise": len(self.rejected) - len(self.volumes),
            "objects": [asdict(v) for v in self.volumes],
            "coregistration": self.coregistration,
            "output_files": self.output_files,
        }


class VolumeCalculationPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def _preprocess(self, dsm: np.ndarray, ortho: np.ndarray | None) -> np.ndarray:
        validate_grids(dsm, ortho)
        dsm = fill_nodata(dsm)
        dsm = denoise_dsm(dsm, size=3)
        return dsm

    def _detect_and_filter(
        self, dsm: np.ndarray, ortho: np.ndarray | None
    ) -> tuple[list[ObjectCandidate], list, np.ndarray, np.ndarray]:
        candidates, trend, residual = segment_objects(dsm, self.config)
        valid_candidates, classifications = filter_candidates(candidates, ortho, self.config)
        return valid_candidates, classifications, trend, residual

    def run_single_epoch(
        self,
        dsm: np.ndarray,
        ortho: np.ndarray | None = None,
        out_dir: str | Path | None = None,
        transform: RasterTransform | None = None,
        report_prefix: str = "report",
    ) -> PipelineResult:
        dsm_clean = self._preprocess(dsm, ortho)
        valid_candidates, classifications, trend, residual = self._detect_and_filter(dsm_clean, ortho)

        datum_by_id = compute_datum_for_all(dsm_clean, valid_candidates, self.config)
        vol_results = compute_all_volumes(dsm_clean, valid_candidates, datum_by_id, self.config.pixel_size_m)

        result = PipelineResult(
            candidates=valid_candidates,
            volumes=vol_results,
            rejected=classifications,
            trend=trend,
            residual=residual,
        )

        if out_dir is not None:
            result.output_files = self._write_reports(
                dsm_clean, valid_candidates, vol_results, classifications, transform, out_dir, report_prefix
            )
        return result

    def run_multitemporal(
        self,
        dsm_t0: np.ndarray,
        dsm_t1: np.ndarray,
        ortho_t1: np.ndarray | None = None,
        out_dir: str | Path | None = None,
        transform: RasterTransform | None = None,
        report_prefix: str = "report_multitemporal",
        auto_coregister: bool = False,
    ) -> PipelineResult:
        dsm_t0_clean = self._preprocess(dsm_t0, None)
        dsm_t1_clean = self._preprocess(dsm_t1, ortho_t1)

        coreg_info: dict | None = None
        if auto_coregister:
            coreg_result = coregister(dsm_t0_clean, dsm_t1_clean)
            coreg_info = {
                "shift_row_px": coreg_result.shift_row_px,
                "shift_col_px": coreg_result.shift_col_px,
                "error": coreg_result.error,
            }
            dsm_t1_clean = coreg_result.dsm_aligned

        valid_candidates, classifications, trend, residual = self._detect_and_filter(dsm_t1_clean, ortho_t1)

        dod = multitemporal.compute_dod(dsm_t0_clean, dsm_t1_clean)

        object_masks = {str(c.id): c.mask for c in valid_candidates}
        change_by_object = multitemporal.compute_change_volumes_by_object(dod, self.config.pixel_size_m, object_masks)
        change_map = {int(r.label): r for r in change_by_object}

        vol_results = []
        for c in valid_candidates:
            ch = change_map.get(c.id)
            if ch is None:
                continue
            ys, xs = np.nonzero(c.mask)
            vol_results.append(
                VolumeResult(
                    object_id=c.id,
                    kind=c.kind,
                    area_m2=ch.area_m2,
                    cut_volume_m3=ch.cut_volume_m3,
                    fill_volume_m3=ch.fill_volume_m3,
                    net_volume_m3=ch.net_volume_m3,
                    mean_height_m=float(dod[c.mask].mean()) if c.mask.any() else 0.0,
                    max_height_m=float(dod[c.mask].max()) if c.mask.any() else 0.0,
                    min_height_m=float(dod[c.mask].min()) if c.mask.any() else 0.0,
                    centroid_row=float(ys.mean()) if len(ys) else 0.0,
                    centroid_col=float(xs.mean()) if len(xs) else 0.0,
                )
            )

        whole_site_change = multitemporal.compute_change_volume(dod, self.config.pixel_size_m, label="вся площадка")

        result = PipelineResult(
            candidates=valid_candidates,
            volumes=vol_results,
            rejected=classifications,
            trend=trend,
            residual=residual,
            coregistration=coreg_info,
        )

        if out_dir is not None:
            out_files = self._write_reports(
                dsm_t1_clean, valid_candidates, vol_results, classifications, transform, out_dir, report_prefix
            )
            out_dir = Path(out_dir)
            cartogram_path = out_dir / f"{report_prefix}_dod_cartogram.png"
            reporting.save_cartogram(
                dod, cartogram_path, self.config.pixel_size_m,
                title="Картограмма изменений между эпохами (DoD = ЦМП(t1) − ЦМП(t0))",
                object_masks=[(c.id, c.mask) for c in valid_candidates],
            )
            out_files["dod_cartogram_png"] = str(cartogram_path)
            out_files["whole_site_summary"] = asdict(whole_site_change)
            if coreg_info is not None:
                out_files["coregistration"] = coreg_info
            result.output_files = out_files

        return result

    def _write_reports(
        self,
        dsm: np.ndarray,
        candidates: list[ObjectCandidate],
        vol_results: list[VolumeResult],
        classifications: list,
        transform: RasterTransform | None,
        out_dir: str | Path,
        prefix: str,
    ) -> dict[str, str]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        statement_df = reporting.build_volume_statement(vol_results, transform)
        noise_df = reporting.build_noise_report(classifications)
        xlsx_path = out_dir / f"{prefix}_vedomost.xlsx"
        reporting.save_statement_xlsx(statement_df, xlsx_path, extra_sheets={"Отсев (аудит)": noise_df})

        diff_full = np.zeros(dsm.shape, dtype=np.float64)
        datum_by_id = compute_datum_for_all(dsm, candidates, self.config)
        for c in candidates:
            datum_local, window = datum_by_id[c.id]
            mask_local = c.mask[window]
            diff_full[window][mask_local] = (dsm[window] - datum_local)[mask_local]

        cartogram_path = out_dir / f"{prefix}_cartogram.png"
        reporting.save_cartogram(
            diff_full, cartogram_path, self.config.pixel_size_m,
            title="Картограмма выемки/насыпи по объектам учёта",
            object_masks=[(c.id, c.mask) for c in candidates],
        )

        geojson_path = out_dir / f"{prefix}_objects.geojson"
        results_by_id = {v.object_id: v for v in vol_results}
        transform_ = transform or RasterTransform(0.0, 0.0, self.config.pixel_size_m)
        reporting.export_geojson(candidates, results_by_id, transform_, geojson_path)

        summary_path = out_dir / f"{prefix}_summary.json"
        summary_path.write_text(
            json.dumps([asdict(v) for v in vol_results], ensure_ascii=False, indent=2)
        )

        return {
            "vedomost_xlsx": str(xlsx_path),
            "cartogram_png": str(cartogram_path),
            "objects_geojson": str(geojson_path),
            "summary_json": str(summary_path),
        }
