"""Normalised cross-correlation of a template sliding over a search window."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.signal import fftconvolve

FLAT_STD = 1e-6  # template std below this (after float32) is considered textureless


def odd(n: int) -> int:
    n = int(n)
    return n if n % 2 else n + 1


@dataclass
class MatchResult:
    dx: float = float("nan")      # displacement image 1 -> image 2, pixels (+x = right)
    dy: float = float("nan")      # (+y = down, image convention)
    corr: float = float("nan")    # peak NCC value
    peak_ratio: float = float("nan")  # peak / best value outside a 3 px exclusion zone
    flags: list[str] = field(default_factory=list)
    surface: np.ndarray | None = None  # NCC surface at the final level
    surface_origin: tuple[int, int] | None = None  # displacement (dx, dy) of surface[0, 0]

    @property
    def ok(self) -> bool:
        return not self.flags and np.isfinite(self.dx)


def ncc_surface(template: np.ndarray, search: np.ndarray) -> np.ndarray:
    """Zero-mean normalised cross-correlation (cv2.matchTemplate, TM_CCOEFF_NORMED)."""
    s = cv2.matchTemplate(search.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    s = np.nan_to_num(s, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return np.clip(s, -1.0, 1.0)


def ncc_surface_fft(template: np.ndarray, search: np.ndarray) -> np.ndarray:
    """Pure numpy/scipy reference implementation of :func:`ncc_surface` (used in tests)."""
    t = template.astype(np.float64)
    f = search.astype(np.float64)
    th, tw = t.shape
    n = th * tw
    t0 = t - t.mean()
    t_norm = np.sqrt((t0 ** 2).sum())
    num = fftconvolve(f, t0[::-1, ::-1], mode="valid")
    ones = np.ones_like(t)
    s1 = fftconvolve(f, ones, mode="valid")
    s2 = fftconvolve(f ** 2, ones, mode="valid")
    var = np.maximum(s2 - s1 ** 2 / n, 0.0)
    den = t_norm * np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 1e-9, num / den, -1.0)
    return np.clip(out, -1.0, 1.0)


def _parabolic(m1: float, c: float, p1: float) -> float:
    d = m1 - 2 * c + p1
    return 0.0 if abs(d) < 1e-12 else float(np.clip(0.5 * (m1 - p1) / d, -0.5, 0.5))


def match_point(
    img1: np.ndarray,
    img2: np.ndarray,
    x: int,
    y: int,
    sample_size: int,
    search_size: int,
    guess: tuple[int, int] = (0, 0),
    subpixel: bool = True,
    valid: np.ndarray | None = None,
    exclusion: int = 1,
) -> MatchResult:
    """Match the ``sample_size`` square around (x, y) of img1 inside img2.

    The search window (``search_size`` square, clipped to the image) is centred on
    (x, y) + ``guess``. Returned dx, dy are the total displacement of the sample centre.
    """
    T, S = odd(sample_size), odd(search_size)
    if S < T:
        raise ValueError("search_size must be >= sample_size")
    ht = T // 2
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    res = MatchResult()

    tx0, ty0 = x - ht, y - ht
    if tx0 < 0 or ty0 < 0 or tx0 + T > w1 or ty0 + T > h1:
        res.flags.append("outside")
        return res
    template = img1[ty0:ty0 + T, tx0:tx0 + T]
    if valid is not None and not valid[ty0:ty0 + T, tx0:tx0 + T].all():
        res.flags.append("masked")
    if float(template.std()) < FLAT_STD:
        res.flags.append("low_texture")
        return res

    cx, cy = x + int(guess[0]), y + int(guess[1])
    hs = S // 2
    sx0, sy0 = max(0, cx - hs), max(0, cy - hs)
    sx1, sy1 = min(w2, cx + hs + 1), min(h2, cy + hs + 1)
    if sx1 - sx0 < T or sy1 - sy0 < T:
        res.flags.append("outside")
        return res
    if valid is not None and not valid[sy0:sy1, sx0:sx1].all() and "masked" not in res.flags:
        res.flags.append("masked")

    surf = ncc_surface(template, img2[sy0:sy1, sx0:sx1])
    iy, ix = np.unravel_index(int(np.argmax(surf)), surf.shape)
    peak = float(surf[iy, ix])

    # second-best peak outside the exclusion zone around the main peak
    r = exclusion + 1
    masked = surf.copy()
    masked[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1] = -np.inf
    second = float(masked.max())
    if np.isfinite(second) and second > 0:
        res.peak_ratio = peak / second
    else:
        res.peak_ratio = float("inf")

    sub_x = sub_y = 0.0
    on_border = ix in (0, surf.shape[1] - 1) or iy in (0, surf.shape[0] - 1)
    if on_border:
        res.flags.append("border")
    elif subpixel:
        sub_x = _parabolic(surf[iy, ix - 1], peak, surf[iy, ix + 1])
        sub_y = _parabolic(surf[iy - 1, ix], peak, surf[iy + 1, ix])

    res.dx = sx0 + ix + ht + sub_x - x
    res.dy = sy0 + iy + ht + sub_y - y
    res.corr = peak
    res.surface = surf
    res.surface_origin = (sx0 + ht - x, sy0 + ht - y)
    return res
