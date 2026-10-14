"""Coarse-to-fine matching over a Gaussian pyramid."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from .matching import MatchResult, match_point, odd
from .pyramid import build_pyramid, to_level


def match_points(
    img1: np.ndarray,
    img2: np.ndarray,
    points: list[tuple[float, float]],
    sample_size: int = 31,
    search_size: int = 71,
    n_levels: int = 3,
    refine_radius: int = 4,
    min_corr: float = 0.5,
    min_peak_ratio: float = 1.05,
    subpixel: bool = True,
    valid: np.ndarray | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[MatchResult]:
    """Match each (x, y) point (full-resolution pixel coords) of img1 in img2.

    The coarsest level uses the full ``search_size`` window; finer levels search only
    ``refine_radius`` pixels around the upsampled coarse estimate. The sample and search
    windows keep the same size in pixels at every level, so they cover 2**level times more
    ground at coarse levels. The pyramid is shortened if the coarsest image would be
    smaller than the search window. ``progress(done, total)`` is called after each point;
    it may raise to abort.
    """
    T, S = odd(sample_size), odd(search_size)
    pyr1 = build_pyramid(img1, n_levels, min_size=S)
    pyr2 = build_pyramid(img2, n_levels, min_size=S)
    n = min(len(pyr1), len(pyr2))
    vpyr = None
    if valid is not None:
        vf = build_pyramid(valid.astype(np.float32), n)
        vpyr = [v > 0.999 for v in vf]  # a coarse pixel is valid only if all its parents were

    refine_size = min(S, T + 2 * refine_radius)
    results = []
    for i, (px, py) in enumerate(points):
        x0, y0 = int(round(px)), int(round(py))
        gx = gy = 0.0   # displacement estimate, in pixels of the current level
        res = MatchResult()
        level_flags: set[str] = set()
        for level in range(n - 1, -1, -1):
            x, y = int(round(to_level(x0, level))), int(round(to_level(y0, level)))
            size = S if level == n - 1 else refine_size
            res = match_point(
                pyr1[level], pyr2[level], x, y, T, size,
                guess=(int(round(gx)), int(round(gy))),
                subpixel=subpixel and level == 0,
                valid=vpyr[level] if vpyr else None,
            )
            if not np.isfinite(res.dx):
                level_flags.update(res.flags)
                break
            if level < n - 1:
                # The refined shift should stay close to the upsampled coarse estimate.
                if max(abs(res.dx - gx), abs(res.dy - gy)) > 0.75 * refine_radius:
                    level_flags.add("level_inconsistent")
            level_flags.update(res.flags)
            if level > 0:
                gx, gy = 2 * res.dx, 2 * res.dy
        res = replace(res, flags=sorted(level_flags))
        if np.isfinite(res.dx):
            if res.corr < min_corr:
                res.flags.append("low_corr")
            if res.peak_ratio < min_peak_ratio:
                res.flags.append("ambiguous")
        results.append(res)
        if progress:
            progress(i + 1, len(points))
    return results
