from typing import Dict

import numpy as np

from .error_model import gaussian_blur


def fuse_ground(D_dsm: np.ndarray, D_corr: np.ndarray, ground_mask: np.ndarray, weight: float = 0.8) -> np.ndarray:
    blended = np.where(np.isfinite(D_dsm), D_dsm, D_corr)
    if weight < 1.0:
        blended = weight * blended + (1 - weight) * D_corr
    return np.where(ground_mask, blended, np.nan)


def fuse_roof(D_dsm: np.ndarray, D_corr: np.ndarray, roof_mask: np.ndarray, detail_sigma: float = 5.0, alpha: float = 0.4) -> np.ndarray:
    base = D_dsm
    detail = D_corr - gaussian_blur(D_corr, sigma=detail_sigma)
    fused = base + alpha * detail
    return np.where(roof_mask, fused, np.nan)


def fuse_facade(D_plane: np.ndarray, D_corr: np.ndarray, facade_mask: np.ndarray, detail_sigma: float = 5.0, beta: float = 0.4) -> np.ndarray:
    detail = D_corr - gaussian_blur(D_corr, sigma=detail_sigma)
    fused = D_plane + beta * detail
    return np.where(facade_mask, fused, np.nan)


def fuse_vegetation(D_dsm: np.ndarray, D_corr: np.ndarray, vegetation_mask: np.ndarray, weight: float = 0.5) -> np.ndarray:
    blended = weight * D_dsm + (1 - weight) * D_corr
    return np.where(vegetation_mask, blended, np.nan)


def combine_regions(maps: Dict[str, np.ndarray], fallback: np.ndarray) -> np.ndarray:
    """Combine per-region depth maps by prioritized overwrite order."""

    out = np.full_like(fallback, np.nan)
    for name in ["ground", "roof", "facade", "vegetation"]:
        if name in maps:
            region = maps[name]
            mask = np.isfinite(region)
            out[mask] = region[mask]
    remaining = ~np.isfinite(out) & np.isfinite(fallback)
    out[remaining] = fallback[remaining]
    return out
