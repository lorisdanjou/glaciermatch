"""GeoTIFF metadata reading."""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import rasterio
from affine import Affine
from rasterio.crs import CRS


@dataclass(frozen=True)
class RasterInfo:
    path: str
    crs: CRS | None
    transform: Affine
    width: int
    height: int
    count: int
    dtype: str
    nodata: float | None
    bounds: tuple[float, float, float, float]  # left, bottom, right, top

    @property
    def pixel_size(self) -> tuple[float, float]:
        """(x, y) pixel size in CRS units."""
        t = self.transform
        return hypot(t.a, t.d), hypot(t.b, t.e)

    @property
    def north_up(self) -> bool:
        t = self.transform
        return t.b == 0 and t.d == 0 and t.a > 0 and t.e < 0

    @property
    def georeferenced(self) -> bool:
        return self.crs is not None and self.transform != Affine.identity()


def read_info(path: str) -> RasterInfo:
    with rasterio.open(path) as ds:
        t = ds.transform
        corners = [t * (0, 0), t * (ds.width, 0), t * (0, ds.height), t * (ds.width, ds.height)]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return RasterInfo(
            path=str(path),
            crs=ds.crs,
            transform=t,
            width=ds.width,
            height=ds.height,
            count=ds.count,
            dtype=ds.dtypes[0],
            nodata=ds.nodata,
            bounds=(min(xs), min(ys), max(xs), max(ys)),
        )
