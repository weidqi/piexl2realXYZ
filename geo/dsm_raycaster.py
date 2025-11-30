from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .camera_model import CameraModel
from .dsm_io import DsmGrid


@dataclass
class RaycastConfig:
    step: float = 0.05
    s_near: float = 0.1
    s_far: float = 500.0
    binary_search_iters: int = 8


def _binary_search(
    s0: float,
    s1: float,
    ray_dir_c: np.ndarray,
    cam: CameraModel,
    dsm: DsmGrid,
    diff_fn: Callable[[float], float],
    iters: int,
) -> float:
    lo, hi = s0, s1
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        diff = diff_fn(mid)
        if diff > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def raycast_pixel(u: int, v: int, cam: CameraModel, dsm: DsmGrid, cfg: RaycastConfig) -> float:
    """Cast a single pixel ray against the DSM and return z_c depth or NaN."""

    ray_dir_c = cam.pixel_to_ray(u, v)

    def diff_at_s(s: float) -> float:
        X_c = s * ray_dir_c
        X_w = cam.cam_to_world(X_c)
        x_w, y_w, z_w = X_w
        z_dsm = dsm.bilinear_sample(x_w, y_w)
        return z_w - z_dsm

    prev_diff: Optional[float] = None
    prev_s: Optional[float] = None
    s = cfg.s_near
    while s < cfg.s_far:
        diff = diff_at_s(s)
        if np.isnan(diff):
            s += cfg.step
            continue

        if prev_diff is not None and np.sign(diff) != np.sign(prev_diff):
            s_hit = _binary_search(prev_s, s, ray_dir_c, cam, dsm, diff_at_s, cfg.binary_search_iters)
            depth = s_hit * ray_dir_c[2]
            return float(depth)

        prev_diff = diff
        prev_s = s
        s += cfg.step

    return float("nan")


def compute_dsm_depth_map(cam: CameraModel, dsm: DsmGrid, cfg: RaycastConfig, image_size: Tuple[int, int]) -> np.ndarray:
    """Compute a DSM-backed depth map for the entire image.

    ``image_size`` is given as ``(height, width)``.
    """

    height, width = image_size
    depth = np.full((height, width), np.nan, dtype=float)
    for v in range(height):
        for u in range(width):
            depth[v, u] = raycast_pixel(u, v, cam, dsm, cfg)
    return depth
