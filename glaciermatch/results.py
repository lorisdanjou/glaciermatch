"""Turn match results into a table with map coordinates, metres and velocities; export it."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .coregistration import CoregResult
from .matching import MatchResult

COLUMNS = [
    "id", "col", "row", "x", "y", "dx_px", "dy_px", "dx_m", "dy_m", "disp_px", "disp_m",
    "vel_m_day", "vel_m_yr", "corr", "peak_ratio", "ok", "flags",
]


def build_table(
    coreg: CoregResult,
    points: list[tuple[int, int]],
    results: list[MatchResult],
    dt_days: float | None = None,
) -> pd.DataFrame:
    """One row per point. dx/dy_px follow the image convention (+y down); dx/dy_m are
    east/north. Metre and velocity columns are NaN when the CRS is not projected or
    ``dt_days`` is unknown."""
    metres = float("nan")
    if coreg.crs is not None and coreg.crs.is_projected:
        metres = coreg.pixel_size * coreg.crs.linear_units_factor[1]
    rows = []
    for i, ((c, r), m) in enumerate(zip(points, results), 1):
        x, y = coreg.pix_to_map(c, r)
        disp_px = math.hypot(m.dx, m.dy)
        disp_m = disp_px * metres
        vel = disp_m / dt_days if dt_days and dt_days > 0 else float("nan")
        rows.append({
            "id": i, "col": c, "row": r, "x": x, "y": y,
            "dx_px": m.dx, "dy_px": m.dy,
            "dx_m": m.dx * metres, "dy_m": -m.dy * metres,
            "disp_px": disp_px, "disp_m": disp_m,
            "vel_m_day": vel, "vel_m_yr": vel * 365.25,
            "corr": m.corr, "peak_ratio": m.peak_ratio,
            "ok": bool(m.ok), "flags": ",".join(m.flags),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def export_csv(table: pd.DataFrame, path: str) -> None:
    table.to_csv(path, index=False, float_format="%.6g")


def export_geojson(table: pd.DataFrame, path: str, crs) -> None:
    """Point features at the sample locations (CRS of the common grid)."""
    def clean(v):
        if isinstance(v, (np.floating, float)):
            return None if not math.isfinite(v) else float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v

    feats = []
    for rec in table.to_dict("records"):
        props = {k: clean(v) for k, v in rec.items() if k not in ("x", "y")}
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [rec["x"], rec["y"]]},
            "properties": props,
        })
    fc = {"type": "FeatureCollection", "features": feats}
    if crs is not None:
        fc["crs"] = {"type": "name", "properties": {"name": crs.to_string()}}
    with open(path, "w") as f:
        json.dump(fc, f, indent=1)
