"""Input validation for a pair of GeoTIFFs. All checks run at load time."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rasterio.warp import transform_bounds

from .io import RasterInfo


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    message: str


def has_errors(checks: list[Check]) -> bool:
    return any(c.status is Status.ERROR for c in checks)


def overlap_bounds(info1: RasterInfo, info2: RasterInfo):
    """Intersection of both extents, in image 1's CRS. None if they do not overlap."""
    b2 = info2.bounds
    if info2.crs != info1.crs:
        b2 = transform_bounds(info2.crs, info1.crs, *b2, densify_pts=21)
    b1 = info1.bounds
    left, bottom = max(b1[0], b2[0]), max(b1[1], b2[1])
    right, top = min(b1[2], b2[2]), min(b1[3], b2[3])
    if right <= left or top <= bottom:
        return None
    return left, bottom, right, top


def lattice_offset(info1: RasterInfo, info2: RasterInfo) -> tuple[float, float]:
    """Fractional pixel offset (in [-0.5, 0.5)) of image 2's lattice relative to image 1's.

    Only meaningful when both are north-up with the same CRS and pixel size.
    """
    rx, ry = info1.pixel_size
    fx = ((info2.transform.c - info1.transform.c) / rx + 0.5) % 1.0 - 0.5
    fy = ((info1.transform.f - info2.transform.f) / ry + 0.5) % 1.0 - 0.5
    return fx, fy


def check_inputs(
    info1: RasterInfo,
    info2: RasterInfo,
    band1: int = 1,
    band2: int = 1,
    res_tol: float = 0.01,
    min_overlap: float = 0.10,
    lattice_tol: float = 1e-3,
) -> list[Check]:
    out: list[Check] = []

    def add(name, status, msg):
        out.append(Check(name, status, msg))

    # georeferencing
    for k, info in enumerate((info1, info2), 1):
        if not info.georeferenced:
            add("georeferenced", Status.ERROR, f"Image {k} has no CRS/geotransform.")
    if has_errors(out):
        return out
    add("georeferenced", Status.OK, "Both images are georeferenced.")

    # bands
    for k, (info, band) in enumerate(((info1, band1), (info2, band2)), 1):
        if not 1 <= band <= info.count:
            add("bands", Status.ERROR, f"Image {k}: band {band} requested, file has {info.count}.")
        elif info.count > 1:
            add("bands", Status.WARN, f"Image {k} has {info.count} bands; using band {band}.")
    if not any(c.name == "bands" for c in out):
        add("bands", Status.OK, "Single-band inputs.")
    if has_errors(out):
        return out

    # nodata
    for k, info in enumerate((info1, info2), 1):
        if info.nodata is None:
            add("nodata", Status.WARN, f"Image {k} defines no nodata value; all pixels treated as valid.")
    if not any(c.name == "nodata" for c in out):
        add("nodata", Status.OK, "Nodata defined for both images.")

    # CRS
    same_crs = info1.crs == info2.crs
    if same_crs:
        add("crs", Status.OK, f"Same CRS ({info1.crs.to_string()}).")
    else:
        add("crs", Status.WARN,
            f"CRS differ ({info1.crs.to_string()} vs {info2.crs.to_string()}); "
            "image 2 will be reprojected to image 1's CRS.")
    if not info1.crs.is_projected:
        add("units", Status.WARN, "Image 1 CRS is geographic: displacements in metres are not available.")

    # pixel size
    (rx1, ry1), (rx2, ry2) = info1.pixel_size, info2.pixel_size
    dx, dy = abs(rx2 - rx1) / rx1, abs(ry2 - ry1) / ry1
    same_res = same_crs and dx <= res_tol and dy <= res_tol
    if not same_crs:
        # Pixel sizes are in different units; the comparison happens after reprojection.
        add("pixel_size", Status.WARN, "Pixel sizes not comparable across CRS; image 2 will be resampled.")
    elif same_res:
        add("pixel_size", Status.OK, f"Same pixel size ({rx1:g} x {ry1:g}).")
    else:
        add("pixel_size", Status.WARN,
            f"Pixel sizes differ ({rx1:g}x{ry1:g} vs {rx2:g}x{ry2:g}); image 2 will be resampled.")

    # orientation
    for k, info in enumerate((info1, info2), 1):
        if not info.north_up:
            add("north_up", Status.WARN, f"Image {k} is rotated/sheared; it will be warped to a north-up grid.")
    if not any(c.name == "north_up" for c in out):
        add("north_up", Status.OK, "Both images are north-up.")

    # overlap
    ov = overlap_bounds(info1, info2)
    if ov is None:
        add("overlap", Status.ERROR, "The images do not overlap.")
        return out
    area = (ov[2] - ov[0]) * (ov[3] - ov[1])
    a1 = (info1.bounds[2] - info1.bounds[0]) * (info1.bounds[3] - info1.bounds[1])
    f1 = area / a1
    if same_crs:
        a2 = (info2.bounds[2] - info2.bounds[0]) * (info2.bounds[3] - info2.bounds[1])
        f2 = area / a2
    else:
        f2 = float("nan")
    msg = f"Overlap covers {100 * f1:.0f}% of image 1" + ("" if f2 != f2 else f" and {100 * f2:.0f}% of image 2") + "."
    add("overlap", Status.WARN if min(f1, f2 if f2 == f2 else 1) < min_overlap else Status.OK, msg)

    # sub-pixel lattice offset
    if same_res and info1.north_up and info2.north_up:
        fx, fy = lattice_offset(info1, info2)
        if abs(fx) <= lattice_tol and abs(fy) <= lattice_tol:
            add("lattice_offset", Status.OK, "Pixel grids are aligned: image 2 is cropped without interpolation.")
        else:
            add("lattice_offset", Status.WARN,
                f"Pixel grids offset by ({fx:+.3f}, {fy:+.3f}) px; image 2 will be interpolated.")
    return out
