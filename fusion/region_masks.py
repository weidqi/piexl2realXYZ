from typing import Dict, Optional

import numpy as np

from geo.dsm_io import DsmGrid


def _compute_gradient_magnitude(dsm: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(dsm)
    return np.sqrt(gx * gx + gy * gy)


def build_ground_mask(dsm: DsmGrid, slope_threshold: float = 0.05) -> np.ndarray:
    """Identify near-flat regions as ground candidates."""

    grad = _compute_gradient_magnitude(dsm.data)
    mask = grad < slope_threshold
    return mask


def build_roof_mask(dsm: DsmGrid, min_height: Optional[float] = None) -> np.ndarray:
    """Simple heuristic to identify elevated roof regions."""

    height = dsm.data
    threshold = np.percentile(height[~np.isnan(height)], 70) if min_height is None else min_height
    mask = height > threshold
    return mask


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
