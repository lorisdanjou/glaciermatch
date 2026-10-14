"""Put two GeoTIFFs on a common pixel grid (image 1's lattice, cropped to the overlap)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject

from .checks import Check, Status, check_inputs, has_errors, lattice_offset, overlap_bounds
from .io import read_info

_RESAMPLING = {"bilinear": Resampling.bilinear, "cubic": Resampling.cubic}


class CoregistrationError(Exception):
    def __init__(self, message: str, checks: list[Check] | None = None):
        super().__init__(message)
        self.checks = checks or []


@dataclass
class CoregResult:
    img1: np.ndarray            # float32, NaN where invalid
    img2: np.ndarray            # float32, NaN where invalid
    valid: np.ndarray           # bool, True where both images have data
    transform: Affine           # of the common grid
    crs: CRS
    resampling: str             # how image 2 was brought onto the grid
    frac_offset: tuple[float, float] | None  # lattice offset of image 2 (px), if applicable
    checks: list[Check]

    @property
    def pixel_size(self) -> float:
        return abs(self.transform.a)

    @property
    def shape(self) -> tuple[int, int]:
        return self.img1.shape

    def pix_to_map(self, col: float, row: float) -> tuple[float, float]:
        return self.transform * (col + 0.5, row + 0.5)

    def map_to_pix(self, x: float, y: float) -> tuple[float, float]:
        col, row = ~self.transform * (x, y)
        return col - 0.5, row - 0.5

    def matching_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """NaN-free float32 copies for correlation: invalid pixels get the mean of the valid ones."""
        out = []
        for img in (self.img1, self.img2):
            a = img.copy()
            bad = ~np.isfinite(a)
            fill = float(a[~bad].mean()) if (~bad).any() else 0.0
            a[bad] = fill
            out.append(a)
        return out[0], out[1]


def _reference_transform(info) -> tuple[Affine, int, int]:
    if info.north_up:
        return info.transform, info.width, info.height
    t, w, h = calculate_default_transform(
        info.crs, info.crs, info.width, info.height, *info.bounds
    )
    return t, w, h


def coregister(
    path1: str,
    path2: str,
    band1: int = 1,
    band2: int = 1,
    resampling: str = "bilinear",
) -> CoregResult:
    """Resample both images onto image 1's pixel lattice, cropped to their overlap.

    Image 1 is never interpolated when it is north-up. Image 2 is copied without
    interpolation when its lattice is aligned, and resampled otherwise.
    """
    info1, info2 = read_info(path1), read_info(path2)
    checks = check_inputs(info1, info2, band1, band2)
    if has_errors(checks):
        msgs = "; ".join(c.message for c in checks if c.status is Status.ERROR)
        raise CoregistrationError(msgs, checks)

    ref_t, ref_w, ref_h = _reference_transform(info1)
    ov = overlap_bounds(info1, info2)
    rx, ry = ref_t.a, -ref_t.e
    x0, y0 = ref_t.c, ref_t.f
    eps = 1e-6
    c0 = max(0, math.ceil((ov[0] - x0) / rx - eps))
    c1 = min(ref_w, math.floor((ov[2] - x0) / rx + eps))
    r0 = max(0, math.ceil((y0 - ov[3]) / ry - eps))
    r1 = min(ref_h, math.floor((y0 - ov[1]) / ry + eps))
    if c1 <= c0 or r1 <= r0:
        raise CoregistrationError("Overlap is smaller than one pixel.", checks)
    width, height = c1 - c0, r1 - r0
    grid_t = Affine(rx, 0, x0 + c0 * rx, 0, -ry, y0 - r0 * ry)

    # Does image 2's lattice coincide with the grid? Then a copy, not an interpolation.
    same_grid = (
        info2.crs == info1.crs and info1.north_up and info2.north_up
        and abs(info2.pixel_size[0] - rx) / rx < 1e-3 and abs(info2.pixel_size[1] - ry) / ry < 1e-3
    )
    frac = lattice_offset(info1, info2) if same_grid else None
    aligned = same_grid and abs(frac[0]) <= 1e-3 and abs(frac[1]) <= 1e-3
    method2 = "nearest" if aligned else resampling
    if method2 not in _RESAMPLING and method2 != "nearest":
        raise ValueError(f"resampling must be one of {sorted(_RESAMPLING)}")
    rs2 = Resampling.nearest if aligned else _RESAMPLING[resampling]
    rs1 = Resampling.nearest if info1.north_up else _RESAMPLING[resampling]

    def warp(path, band, rs):
        dst = np.full((height, width), np.nan, dtype=np.float32)
        with rasterio.open(path) as ds:
            reproject(
                source=rasterio.band(ds, band),
                destination=dst,
                src_nodata=ds.nodata,
                dst_transform=grid_t,
                dst_crs=info1.crs,
                dst_nodata=np.nan,
                resampling=rs,
            )
        return dst

    img1 = warp(path1, band1, rs1)
    img2 = warp(path2, band2, rs2)
    valid = np.isfinite(img1) & np.isfinite(img2)
    if not valid.any():
        raise CoregistrationError("No valid pixels in the overlap.", checks)

    frac_valid = float(valid.mean())
    checks.append(Check(
        "valid_pixels",
        Status.OK if frac_valid >= 0.5 else Status.WARN,
        f"{100 * frac_valid:.0f}% of the {width}x{height} common grid has data in both images.",
    ))
    return CoregResult(img1, img2, valid, grid_t, info1.crs, method2, frac, checks)
