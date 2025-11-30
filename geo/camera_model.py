from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def make_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Create a 3x3 intrinsic matrix."""

    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


@dataclass
class CameraModel:
    """Encapsulates camera intrinsics and extrinsics.

    Extrinsics use the convention ``X_c = R (X_w - C_w)``.
    """

    K: np.ndarray
    R: np.ndarray
    C_w: np.ndarray

    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """Return a normalized ray direction in camera coordinates."""

        pixel_h = np.array([u, v, 1.0], dtype=float)
        ray_dir = np.linalg.inv(self.K) @ pixel_h
        return _normalize(ray_dir)

    def world_to_cam(self, X_w: np.ndarray) -> np.ndarray:
        """Transform world coordinates into the camera frame."""

        return self.R @ (X_w - self.C_w)

    def cam_to_world(self, X_c: np.ndarray) -> np.ndarray:
        """Transform camera coordinates into the world frame."""

        return self.C_w + self.R.T @ X_c

    def project(self, X_w: np.ndarray) -> Tuple[float, float, float]:
        """Project a world point to pixel coordinates.

        Returns ``(u, v, z_c)`` where ``z_c`` is depth along the optical axis.
        """

        X_c = self.world_to_cam(X_w)
        if X_c[2] <= 0:
            return float("nan"), float("nan"), float(X_c[2])

        x = X_c[0] / X_c[2]
        y = X_c[1] / X_c[2]
        pixel = self.K @ np.array([x, y, 1.0])
        return float(pixel[0]), float(pixel[1]), float(X_c[2])
