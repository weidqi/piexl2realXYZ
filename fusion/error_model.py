from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    A = np.stack([x, np.ones_like(x)], axis=1)
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


def fit_global_linear(D_dp: np.ndarray, D_dsm: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    valid = mask & np.isfinite(D_dp) & np.isfinite(D_dsm)
    if valid.sum() < 10:
        return 1.0, 0.0
    x = D_dp[valid].reshape(-1)
    y = D_dsm[valid].reshape(-1)
    return _fit_linear(x, y)


def fit_blockwise_linear(
    D_dp: np.ndarray, D_dsm: np.ndarray, mask: np.ndarray, grid: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = D_dp.shape
    a_grid = np.full((grid, grid), 1.0, dtype=float)
    b_grid = np.zeros((grid, grid), dtype=float)

    for gi in range(grid):
        for gj in range(grid):
            r0 = int(gi * h / grid)
            r1 = int((gi + 1) * h / grid)
            c0 = int(gj * w / grid)
            c1 = int((gj + 1) * w / grid)
            block_mask = mask[r0:r1, c0:c1]
            valid = block_mask & np.isfinite(D_dp[r0:r1, c0:c1]) & np.isfinite(D_dsm[r0:r1, c0:c1])
            if valid.sum() < 20:
                continue
            x = D_dp[r0:r1, c0:c1][valid]
            y = D_dsm[r0:r1, c0:c1][valid]
            a, b = _fit_linear(x, y)
            a_grid[gi, gj] = a
            b_grid[gi, gj] = b

    A = _bilinear_upsample(a_grid, h, w)
    B = _bilinear_upsample(b_grid, h, w)
    return A, B


def apply_correction(D_dp: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A * D_dp + B


def apply_residual_smoothing(D_corr: np.ndarray, D_dsm: np.ndarray, mask: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    R = np.zeros_like(D_corr)
    valid = mask & np.isfinite(D_corr) & np.isfinite(D_dsm)
    R[valid] = D_corr[valid] - D_dsm[valid]
    R_smooth = gaussian_blur(R, sigma=sigma)
    return D_corr - R_smooth


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(3 * sigma))
    coords = np.arange(-radius, radius + 1)
    kernel = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    def convolve_1d(arr: np.ndarray, axis: int) -> np.ndarray:
        padded = np.pad(arr, [(radius, radius)] * arr.ndim, mode="edge")
        for _ in range(axis):
            padded = np.swapaxes(padded, 0, 1)
        out = np.empty_like(arr)
        for idx in range(arr.shape[axis]):
            slices = [slice(None)] * arr.ndim
            slices[axis] = slice(idx, idx + 2 * radius + 1)
            window = padded[tuple(slices)]
            out_idx = [slice(None)] * arr.ndim
            out_idx[axis] = idx
            out[tuple(out_idx)] = np.tensordot(window, kernel, axes=(axis, 0))
        for _ in range(axis):
            out = np.swapaxes(out, 0, 1)
        return out

    tmp = convolve_1d(image, axis=1)
    blurred = convolve_1d(tmp, axis=0)
    return blurred


def _bilinear_upsample(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    gh, gw = grid.shape
    xs = np.linspace(0, gw - 1, w)
    ys = np.linspace(0, gh - 1, h)
    xi, yi = np.meshgrid(xs, ys)
    x0 = np.floor(xi).astype(int)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    y0 = np.floor(yi).astype(int)
    y1 = np.clip(y0 + 1, 0, gh - 1)

    wa = (x1 - xi) * (y1 - yi)
    wb = (xi - x0) * (y1 - yi)
    wc = (x1 - xi) * (yi - y0)
    wd = (xi - x0) * (yi - y0)

    return (
        wa * grid[y0, x0]
        + wb * grid[y0, x1]
        + wc * grid[y1, x0]
        + wd * grid[y1, x1]
    )
