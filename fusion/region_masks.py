from typing import Dict

import numpy as np


def _compute_gradient_magnitude(depth_map: np.ndarray) -> np.ndarray:
    """Return gradient magnitude while ignoring NaNs with edge padding."""

    if np.all(np.isnan(depth_map)):
        return np.zeros_like(depth_map)

    finite = np.nan_to_num(depth_map, nan=np.nanmedian(depth_map))
    gy, gx = np.gradient(finite)
    return np.sqrt(gx * gx + gy * gy)


def build_ground_mask(dsm_depth: np.ndarray, slope_threshold: float = 0.05) -> np.ndarray:
    """Identify near-flat regions in the image domain as ground candidates."""

    grad = _compute_gradient_magnitude(dsm_depth)
    return np.isfinite(dsm_depth) & (grad < slope_threshold)


def build_roof_mask(dsm_depth: np.ndarray, percentile: float = 40.0) -> np.ndarray:
    """Heuristic roof mask derived from the image-space DSM depth."""

    finite = np.isfinite(dsm_depth)
    if not np.any(finite):
        return np.zeros_like(dsm_depth, dtype=bool)
    cutoff = np.nanpercentile(dsm_depth, percentile)
    return finite & (dsm_depth <= cutoff)


def build_sky_mask(depth_map: np.ndarray, max_depth: float = 1e6) -> np.ndarray:
    """Mark pixels with invalid or extremely large depth as sky."""

    return ~np.isfinite(depth_map) | (depth_map > max_depth)


def normalize_masks(masks: Dict[str, np.ndarray], shape) -> Dict[str, np.ndarray]:
    """Ensure masks share a common shape and dtype."""

    normalized = {}
    for name, mask in masks.items():
        normalized[name] = mask.astype(bool)
        if normalized[name].shape != shape:
            raise ValueError(f"Mask {name} has shape {mask.shape}, expected {shape}")
    return normalized
