from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np


@dataclass
class TileSpec:
    core: tuple[slice, slice]
    padded: tuple[slice, slice]
    core_in_padded: tuple[slice, slice]


def iter_tiles(shape: tuple[int, int], tile_size: int, overlap: int) -> Iterator[TileSpec]:
    ny, nx = shape
    for r0 in range(0, ny, tile_size):
        r1 = min(ny, r0 + tile_size)
        for c0 in range(0, nx, tile_size):
            c1 = min(nx, c0 + tile_size)
            core = (slice(r0, r1), slice(c0, c1))

            pr0, pc0 = max(0, r0 - overlap), max(0, c0 - overlap)
            pr1, pc1 = min(ny, r1 + overlap), min(nx, c1 + overlap)
            padded = (slice(pr0, pr1), slice(pc0, pc1))

            core_in_padded = (
                slice(r0 - pr0, r0 - pr0 + (r1 - r0)),
                slice(c0 - pc0, c0 - pc0 + (c1 - c0)),
            )
            yield TileSpec(core=core, padded=padded, core_in_padded=core_in_padded)


def _apply_func_to_chunk(args):
    chunk, func, func_kwargs = args
    return func(chunk, **func_kwargs)


def process_in_tiles(
    array: np.ndarray,
    func: Callable[..., np.ndarray],
    tile_size: int = 512,
    overlap: int = 64,
    n_jobs: int = 1,
    **func_kwargs,
) -> np.ndarray:
    shape = array.shape
    out = np.empty(shape, dtype=array.dtype)
    tiles = list(iter_tiles(shape, tile_size, overlap))

    if n_jobs and n_jobs > 1 and len(tiles) > 1:
        tasks = [(array[t.padded], func, func_kwargs) for t in tiles]
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            processed_chunks = list(ex.map(_apply_func_to_chunk, tasks))
    else:
        processed_chunks = [func(array[t.padded], **func_kwargs) for t in tiles]

    for tile, processed in zip(tiles, processed_chunks):
        out[tile.core] = processed[tile.core_in_padded]
    return out

def _design_matrix_1d(xs: np.ndarray, ys: np.ndarray, degree: int) -> np.ndarray:
    terms = [np.ones_like(xs)]
    for total_deg in range(1, degree + 1):
        for i in range(total_deg + 1):
            terms.append((xs ** (total_deg - i)) * (ys**i))
    return np.stack(terms, axis=-1)  # (N, k)


def fit_trend_surface_large(
    dsm: np.ndarray,
    degree: int = 2,
    iterations: int = 3,
    outlier_sigma: float = 1.5,
    tile_size: int = 1024,
) -> np.ndarray:
    ny, nx = dsm.shape
    scale = max(nx, ny)
    k = ((degree + 1) * (degree + 2)) // 2

    valid = np.isfinite(dsm)
    weight_mask = valid.copy()
    coeffs = np.zeros(k)
    trend_full = np.zeros(dsm.shape, dtype=np.float64)

    for _ in range(max(1, iterations)):
        ATA = np.zeros((k, k))
        ATb = np.zeros(k)
        for tile in iter_tiles((ny, nx), tile_size, overlap=0):
            rs, cs = tile.core
            m = weight_mask[rs, cs]
            if not m.any():
                continue
            rows = np.arange(rs.start, rs.stop)
            cols = np.arange(cs.start, cs.stop)
            yy, xx = np.meshgrid(rows, cols, indexing="ij")
            xs_n = ((xx - nx / 2) / scale)[m]
            ys_n = ((yy - ny / 2) / scale)[m]
            zs = dsm[rs, cs][m]
            A = _design_matrix_1d(xs_n, ys_n, degree)
            ATA += A.T @ A
            ATb += A.T @ zs

        coeffs = np.linalg.solve(ATA, ATb)

        sq_sum, count = 0.0, 0
        for tile in iter_tiles((ny, nx), tile_size, overlap=0):
            rs, cs = tile.core
            rows = np.arange(rs.start, rs.stop)
            cols = np.arange(cs.start, cs.stop)
            yy, xx = np.meshgrid(rows, cols, indexing="ij")
            xs_n = (xx - nx / 2) / scale
            ys_n = (yy - ny / 2) / scale
            trend_tile = (_design_matrix_1d(xs_n.ravel(), ys_n.ravel(), degree) @ coeffs).reshape(xs_n.shape)
            trend_full[rs, cs] = trend_tile
            m = weight_mask[rs, cs]
            resid_tile = dsm[rs, cs] - trend_tile
            sq_sum += float(np.sum(resid_tile[m] ** 2))
            count += int(m.sum())

        std = np.sqrt(sq_sum / max(1, count))
        weight_mask = valid & (np.abs(dsm - trend_full) < outlier_sigma * std)

    return trend_full.astype(dsm.dtype)
