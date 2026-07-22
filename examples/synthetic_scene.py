from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class SceneObject:
    name: str
    cx_m: float
    cy_m: float
    radius_m: float
    height_m: float

    def true_volume_m3(self) -> float:
        return 0.5 * np.pi * abs(self.height_m) * self.radius_m**2


@dataclass
class SyntheticScene:
    pixel_size_m: float
    nx: int
    ny: int
    terrain: np.ndarray
    dsm_t0: np.ndarray
    dsm_t1: np.ndarray
    ortho_t1: np.ndarray
    objects_t1: dict[str, SceneObject]
    objects_t0: dict[str, SceneObject]
    vehicle_box: dict

    @property
    def extent_m(self) -> tuple[float, float]:
        return self.nx * self.pixel_size_m, self.ny * self.pixel_size_m


def _paraboloid(X: np.ndarray, Y: np.ndarray, obj: SceneObject) -> np.ndarray:
    r2 = (X - obj.cx_m) ** 2 + (Y - obj.cy_m) ** 2
    h = obj.height_m * (1.0 - r2 / obj.radius_m**2)
    return np.where(r2 <= obj.radius_m**2, h, 0.0)


def generate_terrain(nx: int, ny: int, pixel_size_m: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(np.arange(nx) * pixel_size_m, np.arange(ny) * pixel_size_m)

    base = 100.0
    slope = 0.03 * X - 0.02 * Y
    undulation = 0.12 * np.sin(2 * np.pi * X / 35.0) * np.cos(2 * np.pi * Y / 50.0)
    noise = gaussian_filter(rng.normal(0, 1, size=(ny, nx)), sigma=10)
    noise = 0.08 * noise / (np.abs(noise).max() + 1e-9)

    return base + slope + undulation + noise


def generate_extreme_terrain_patch(
    pixel_size_m: float = 0.15, n: int = 160, seed: int = 99
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(np.arange(n) * pixel_size_m, np.arange(n) * pixel_size_m)
    slope = 0.09 * X - 0.06 * Y
    undulation = 0.7 * np.sin(2 * np.pi * X / 18.0) * np.cos(2 * np.pi * Y / 22.0)
    terrain = 100.0 + slope + undulation
    return terrain, X, Y


def generate_synthetic_scene(pixel_size_m: float = 0.15, nx: int = 400, ny: int = 400, seed: int = 42) -> SyntheticScene:
    terrain = generate_terrain(nx, ny, pixel_size_m, seed=seed)
    X, Y = np.meshgrid(np.arange(nx) * pixel_size_m, np.arange(ny) * pixel_size_m)

    pile_A = SceneObject("pile_A", cx_m=19.5, cy_m=21.0, radius_m=9.0, height_m=4.2)
    gap_m = 0.6
    pile_B = SceneObject(
        "pile_B", cx_m=pile_A.cx_m + pile_A.radius_m + 5.0 + gap_m, cy_m=pile_A.cy_m, radius_m=5.0, height_m=2.3
    )
    pit_C = SceneObject("pit_C", cx_m=42.0, cy_m=42.0, radius_m=7.0, height_m=-2.8)

    pile_A_t0 = SceneObject("pile_A", pile_A.cx_m, pile_A.cy_m, pile_A.radius_m, height_m=2.5)
    pit_C_t0 = SceneObject("pit_C", pit_C.cx_m, pit_C.cy_m, pit_C.radius_m, height_m=-1.0)

    def build_dsm(terrain_arr, objects):
        dsm = terrain_arr.copy()
        for obj in objects:
            if abs(obj.height_m) < 1e-9:
                continue
            dsm = dsm + _paraboloid(X, Y, obj)
        return dsm

    dsm_t0 = build_dsm(terrain, [pile_A_t0, pit_C_t0])
    dsm_t1 = build_dsm(terrain, [pile_A, pile_B, pit_C])

    vehicle = dict(x0_m=6.0, y0_m=27.5, w_m=2.6, l_m=7.0, height_m=1.6)
    vx0, vx1 = int(vehicle["x0_m"] / pixel_size_m), int((vehicle["x0_m"] + vehicle["w_m"]) / pixel_size_m)
    vy0, vy1 = int(vehicle["y0_m"] / pixel_size_m), int((vehicle["y0_m"] + vehicle["l_m"]) / pixel_size_m)
    dsm_t1[vy0:vy1, vx0:vx1] += vehicle["height_m"]

    rng = np.random.default_rng(seed + 1)
    ortho = np.empty((ny, nx, 3), dtype=np.uint8)
    ground_color = np.array([140, 120, 95])
    ortho[:] = ground_color
    ortho = ortho.astype(np.float64) + rng.normal(0, 4, size=(ny, nx, 3))

    mask_A = _paraboloid(X, Y, pile_A) > 0.05
    mask_B = _paraboloid(X, Y, pile_B) > 0.05
    mask_C = _paraboloid(X, Y, pit_C) < -0.05
    pile_color = np.array([176, 172, 160])
    pit_color = np.array([95, 74, 56])
    ortho[mask_A | mask_B] = pile_color + rng.normal(0, 5, size=3)
    ortho[mask_C] = pit_color + rng.normal(0, 4, size=3)

    vehicle_color = np.array([232, 178, 35])
    ortho[vy0:vy1, vx0:vx1] = vehicle_color

    ortho = np.clip(ortho, 0, 255).astype(np.uint8)

    return SyntheticScene(
        pixel_size_m=pixel_size_m,
        nx=nx,
        ny=ny,
        terrain=terrain,
        dsm_t0=dsm_t0,
        dsm_t1=dsm_t1,
        ortho_t1=ortho,
        objects_t1={"pile_A": pile_A, "pile_B": pile_B, "pit_C": pit_C},
        objects_t0={"pile_A": pile_A_t0, "pit_C": pit_C_t0},
        vehicle_box=vehicle,
    )
