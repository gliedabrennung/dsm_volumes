from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .pipeline import VolumeCalculationPipeline
from .reporting import RasterTransform


@dataclass
class _RasterData:
    array: np.ndarray
    transform: Any | None
    crs: Any | None
    nodata: float | None


def _read_raster_raw(path: str) -> _RasterData:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return _RasterData(array=np.load(p), transform=None, crs=None, nodata=None)

    try:
        import rasterio  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"Для чтения {p.suffix} требуется rasterio (pip install rasterio). "
            "Либо экспортируйте растр в .npy."
        ) from exc

    with rasterio.open(p) as src:
        if src.count == 1:
            array = src.read(1)
        else:
            n_bands = min(3, src.count)
            array = np.moveaxis(src.read(list(range(1, n_bands + 1))), 0, -1)
        return _RasterData(array=array, transform=src.transform, crs=src.crs, nodata=src.nodata)


def _transforms_close(t1, t2, tol: float = 1e-6) -> bool:
    if t1 is None or t2 is None:
        return t1 is t2
    return all(abs(a - b) < tol for a, b in zip(tuple(t1)[:6], tuple(t2)[:6]))


def _align_to_reference(data: _RasterData, reference: _RasterData, label: str) -> np.ndarray:
    same_grid = data.array.shape[:2] == reference.array.shape[:2] and _transforms_close(
        data.transform, reference.transform
    )

    if same_grid:
        array = data.array
    elif data.transform is not None and reference.transform is not None:
        from rasterio.warp import Resampling, reproject

        print(
            f"[dsm_volumes] {label}: сетка ({data.array.shape[:2]}) отличается от ЦМП "
            f"({reference.array.shape[:2]}) — выполняю reproject на сетку ЦМП.",
            file=sys.stderr,
        )
        is_multiband = data.array.ndim == 3
        src = np.moveaxis(data.array, -1, 0) if is_multiband else data.array[np.newaxis, :, :]
        dst = np.zeros((src.shape[0], *reference.array.shape[:2]), dtype=src.dtype)
        reproject(
            source=src,
            destination=dst,
            src_transform=data.transform,
            src_crs=data.crs,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            src_nodata=data.nodata,
            dst_nodata=data.nodata,
            resampling=Resampling.bilinear,
        )
        array = np.moveaxis(dst, 0, -1) if is_multiband else dst[0]
    else:
        from .preprocessing import align_to_grid

        print(
            f"[dsm_volumes] {label}: нет geo-метаданных для точного совмещения — "
            "использую пиксельный ресайз (align_to_grid), без учёта geo-привязки.",
            file=sys.stderr,
        )
        array = align_to_grid(data.array, reference.array.shape[:2])

    if data.nodata is not None:
        array = np.where(array == data.nodata, np.nan, array)
    return array


def _crop_to_valid_extent(data: _RasterData, margin_m: float = 15.0) -> _RasterData:
    if data.nodata is not None:
        valid = data.array != data.nodata
    else:
        valid = np.isfinite(data.array)
    if valid.ndim == 3:
        valid = valid.any(axis=-1)

    if not valid.any() or valid.all():
        return data

    ys, xs = np.nonzero(valid)
    pixel_size_m = abs(data.transform.a) if data.transform is not None else 1.0
    margin_px = max(0, int(round(margin_m / pixel_size_m)))

    r0, r1 = max(0, ys.min() - margin_px), min(data.array.shape[0], ys.max() + margin_px + 1)
    c0, c1 = max(0, xs.min() - margin_px), min(data.array.shape[1], xs.max() + margin_px + 1)

    cropped_array = data.array[r0:r1, c0:c1]
    cropped_transform = data.transform
    if data.transform is not None:
        from rasterio.windows import Window, transform as window_transform

        cropped_transform = window_transform(Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0), data.transform)

    print(
        f"[dsm_volumes] валидная область ЦМП — {valid.sum()} из {data.array.size} px "
        f"({valid.sum() / data.array.size * 100:.1f}%). Обрезаю растр {data.array.shape} -> "
        f"{cropped_array.shape} (bbox валидных данных + запас {margin_m} м) перед обработкой.",
        file=sys.stderr,
    )
    return _RasterData(array=cropped_array, transform=cropped_transform, crs=data.crs, nodata=data.nodata)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsm_volumes",
        description="Автоматизированный расчёт объёмов по ЦМП фотограмметрической обработки.",
    )
    parser.add_argument("--dsm", required=True, help="Путь к ЦМП текущей эпохи (GeoTIFF или .npy) — эталонная сетка")
    parser.add_argument("--ortho", help="Путь к ортофотоплану текущей эпохи (любое разрешение — выравнивается автоматически)")
    parser.add_argument("--dsm-prev", help="ЦМП предыдущей эпохи — включает мультивременной режим (DoD); тоже выравнивается автоматически")
    parser.add_argument(
        "--auto-coregister", action="store_true",
        help="Автоматически скорректировать субпиксельный сдвиг между эпохами перед расчётом DoD",
    )
    parser.add_argument("--config", help="JSON/YAML файл конфигурации PipelineConfig")
    parser.add_argument("--pixel-size", type=float, help="Размер пикселя, м (если не берётся из геопривязки)")
    parser.add_argument("--out-dir", required=True, help="Каталог для отчётов")
    parser.add_argument("--prefix", default="report", help="Префикс имён файлов отчёта")
    parser.add_argument(
        "--crop-margin-m", type=float, default=15.0,
        help="Запас (м) при автообрезке ЦМП до фактической валидной области данных (по умолчанию 15 м, 0 — не обрезать)",
    )
    parser.add_argument(
        "--no-auto-crop", action="store_true",
        help="Отключить автообрезку ЦМП до валидной области (обрабатывать весь растр как есть)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    config = PipelineConfig.load(args.config) if args.config else PipelineConfig()

    dsm_data = _read_raster_raw(args.dsm)
    if not args.no_auto_crop:
        dsm_data = _crop_to_valid_extent(dsm_data, margin_m=args.crop_margin_m)
    dsm = dsm_data.array
    if dsm_data.nodata is not None:
        dsm = np.where(dsm == dsm_data.nodata, np.nan, dsm)

    transform = None
    if dsm_data.transform is not None:
        transform = RasterTransform(
            origin_x=dsm_data.transform.c, origin_y=dsm_data.transform.f, pixel_size_m=abs(dsm_data.transform.a)
        )
        if args.pixel_size is None:
            config.pixel_size_m = transform.pixel_size_m
    if args.pixel_size:
        config.pixel_size_m = args.pixel_size

    ortho = None
    if args.ortho:
        ortho_data = _read_raster_raw(args.ortho)
        ortho = _align_to_reference(ortho_data, dsm_data, label="ортофото")
        if ortho.ndim == 2:
            ortho = np.repeat(ortho[:, :, None], 3, axis=2)
        ortho = np.nan_to_num(ortho, nan=0).astype(np.uint8)

    pipeline = VolumeCalculationPipeline(config)

    if args.dsm_prev:
        dsm_prev_data = _read_raster_raw(args.dsm_prev)
        dsm_prev = _align_to_reference(dsm_prev_data, dsm_data, label="ЦМП предыдущей эпохи")
        result = pipeline.run_multitemporal(
            dsm_prev, dsm, ortho, out_dir=args.out_dir, transform=transform,
            report_prefix=args.prefix, auto_coregister=args.auto_coregister,
        )
    else:
        result = pipeline.run_single_epoch(
            dsm, ortho, out_dir=args.out_dir, transform=transform, report_prefix=args.prefix
        )

    print(json.dumps(result.summary_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())