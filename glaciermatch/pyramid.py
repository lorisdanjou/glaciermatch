"""Gaussian image pyramids and coordinate scaling between levels."""
from __future__ import annotations

import cv2
import numpy as np


def build_pyramid(img: np.ndarray, n_levels: int, min_size: int = 1) -> list[np.ndarray]:
    """Level 0 is the input; each next level is blurred and halved (cv2.pyrDown).

    Stops early if the next level would be smaller than ``min_size`` in either axis.
    """
    levels = [np.ascontiguousarray(img, dtype=np.float32)]
    while len(levels) < n_levels:
        h, w = levels[-1].shape
        if (h + 1) // 2 < min_size or (w + 1) // 2 < min_size:
            break
        levels.append(cv2.pyrDown(levels[-1]))
    return levels


def to_level(coord: float, level: int) -> float:
    """Full-resolution pixel coordinate -> pixel coordinate at ``level``.

    pyrDown samples every second pixel, so pixel i of level L sits at i * 2**L.
    """
    return coord / (1 << level)


def from_level(coord: float, level: int) -> float:
    return coord * (1 << level)
