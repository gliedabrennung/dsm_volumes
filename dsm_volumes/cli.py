from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .pipeline import VolumeCalculationPipeline
from .reporting import RasterTransform


def _load_raster(path: str) -> tuple[np.ndarray, RasterTransform | None]:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.load(p), None
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError(
            f"Для чтения {p.suffix} требуется rasterio (pip install rasterio). "
            "Либо экспортируйте ЦМП/ортофото в .npy."
        ) from exc
    with rasterio.open(p) as src:
        array = src.read(1) if src.count == 1 else np.moveaxis(src.read(), 0, -1)
        aff = src.transform
        rt = RasterTransform(origin_x=aff.c, origin_y=aff.f, pixel_size_m=abs(aff.a))
    return array, rt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsm_volumes",
        description="Автоматизированный расчёт объёмов по ЦМП фотограмметрической обработки.",
    )
    parser.add_argument("--dsm", required=True, help="Путь к ЦМП текущей эпохи (GeoTIFF или .npy)")
    parser.add_argument("--ortho", help="Путь к ортофотоплану текущей эпохи")
    parser.add_argument("--dsm-prev", help="ЦМП предыдущей эпохи — включает мультивременной режим (DoD)")
    parser.add_argument(
        "--auto-coregister", action="store_true",
        help="Автоматически скорректировать субпиксельный сдвиг между эпохами перед расчётом DoD",
    )
    parser.add_argument("--config", help="JSON/YAML файл конфигурации PipelineConfig")
    parser.add_argument("--pixel-size", type=float, help="Размер пикселя, м (если не берётся из геопривязки)")
    parser.add_argument("--out-dir", required=True, help="Каталог для отчётов")
    parser.add_argument("--prefix", default="report", help="Префикс имён файлов отчёта")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    config = PipelineConfig.load(args.config) if args.config else PipelineConfig()
    if args.pixel_size:
        config.pixel_size_m = args.pixel_size

    dsm, transform = _load_raster(args.dsm)
    if args.pixel_size is None and transform is not None:
        config.pixel_size_m = transform.pixel_size_m

    ortho = None
    if args.ortho:
        ortho, _ = _load_raster(args.ortho)

    pipeline = VolumeCalculationPipeline(config)

    if args.dsm_prev:
        dsm_prev, _ = _load_raster(args.dsm_prev)
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
