import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class DsmGrid:
    """A lightweight DSM container with affine transforms.

    The grid stores elevation in meters and a 3x3 affine transform that maps
    integer pixel indices ``(row, col)`` into world coordinates ``(X_w, Y_w)``.
    The bottom row of the matrix should be ``[0, 0, 1]``.
    """

    data: np.ndarray
    transform: np.ndarray

    def world_from_pixel(self, row: float, col: float) -> Tuple[float, float]:
        """Convert DSM indices into world coordinates using the affine transform."""

        pixel = np.array([col, row, 1.0], dtype=float)
        world = self.transform @ pixel
        return float(world[0]), float(world[1])

    def pixel_from_world(self, x_w: float, y_w: float) -> Tuple[float, float]:
        """Convert world coordinates into DSM indices.

        The inverse transform is computed on-demand to avoid storing both.
        """

        inv = np.linalg.inv(self.transform)
        world = np.array([x_w, y_w, 1.0], dtype=float)
        pixel = inv @ world
        return float(pixel[1]), float(pixel[0])  # (row, col)

    def bilinear_sample(self, x_w: float, y_w: float) -> float:
        """Sample the DSM at arbitrary world coordinates using bilinear filtering.

        Returns ``np.nan`` if the query lies outside the DSM bounds.
        """

        row, col = self.pixel_from_world(x_w, y_w)
        rows, cols = self.data.shape
        if row < 0 or row >= rows - 1 or col < 0 or col >= cols - 1:
            return float("nan")

        r0 = int(np.floor(row))
        c0 = int(np.floor(col))
        dr = row - r0
        dc = col - c0

        v00 = self.data[r0, c0]
        v01 = self.data[r0, c0 + 1]
        v10 = self.data[r0 + 1, c0]
        v11 = self.data[r0 + 1, c0 + 1]

        top = v00 * (1 - dc) + v01 * dc
        bottom = v10 * (1 - dc) + v11 * dc
        return float(top * (1 - dr) + bottom * dr)

    def inside_bounds(self, x_w: float, y_w: float) -> bool:
        """Check if a world coordinate lies inside the DSM extent."""

        row, col = self.pixel_from_world(x_w, y_w)
        rows, cols = self.data.shape
        return 0 <= row < rows and 0 <= col < cols


def load_dsm_from_json(path: Path) -> DsmGrid:
    """Load a DSM from a small JSON sidecar for lightweight demos/tests.

    The JSON schema is expected to be ``{"transform": [[...],[...],[...]],
    "data": [[...], ...]}``.
    """

    payload = json.loads(Path(path).read_text())
    transform = np.array(payload["transform"], dtype=float)
    data = np.array(payload["data"], dtype=float)
    return DsmGrid(data=data, transform=transform)


def save_dsm_to_json(dsm: DsmGrid, path: Path) -> None:
    """Persist a small DSM grid into a JSON file."""

    payload = {
        "transform": dsm.transform.tolist(),
        "data": dsm.data.tolist(),
    }
    Path(path).write_text(json.dumps(payload, indent=2))
