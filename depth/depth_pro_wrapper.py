from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class DepthProWrapper:
    """Thin wrapper around a Depth Pro inference callable.

    The callable should accept an image array (H, W, 3) and focal length in
    pixels, returning a metric depth map aligned with the input resolution.
    """

    infer_fn: Callable[[np.ndarray, float], np.ndarray]

    def infer_depth(self, image_rgb: np.ndarray, f_px: float) -> np.ndarray:
        depth = self.infer_fn(image_rgb, f_px)
        if depth.shape[:2] != image_rgb.shape[:2]:
            raise ValueError("Depth map resolution must match the input image")
        return depth.astype(float)
