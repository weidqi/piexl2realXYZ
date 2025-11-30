from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from depth.depth_pro_wrapper import DepthProWrapper
from fusion.depth_fusion import combine_regions, fuse_ground, fuse_roof, fuse_vegetation
from fusion.error_model import (
    apply_correction,
    apply_residual_smoothing,
    fit_blockwise_linear,
    fit_global_linear,
)
from fusion.multiframe import merge_frames_median
from fusion.region_masks import build_ground_mask, build_roof_mask, build_sky_mask, normalize_masks
from geo.camera_model import CameraModel
from geo.dsm_io import DsmGrid
from geo.dsm_raycaster import RaycastConfig, compute_dsm_depth_map


@dataclass
class FusionConfig:
    grid: int = 16
    sigma_residual: float = 12.0
    detail_sigma: float = 5.0
    roof_alpha: float = 0.4
    facade_beta: float = 0.4
    ground_weight: float = 0.8
    vegetation_weight: float = 0.5


def correct_single_frame(
    image_rgb: np.ndarray,
    cam: CameraModel,
    dsm: DsmGrid,
    depth_pro: DepthProWrapper,
    reliable_mask: Optional[np.ndarray] = None,
    grid: int = 16,
    sigma_residual: float = 12.0,
    ground_weight: float = 0.8,
    roof_alpha: float = 0.4,
    roof_detail_sigma: float = 5.0,
    vegetation_weight: float = 0.5,
    ray_cfg: Optional[RaycastConfig] = None,
) -> Dict[str, np.ndarray]:
    """Run the DSM-grounded correction for a single frame."""

    ray_cfg = ray_cfg or RaycastConfig()
    H, W = image_rgb.shape[:2]
    D_dsm = compute_dsm_depth_map(cam, dsm, ray_cfg, (H, W))
    D_dp = depth_pro.infer_depth(image_rgb, cam.K[0, 0])

    ground_mask = build_ground_mask(D_dsm)
    roof_mask = build_roof_mask(D_dsm)
    sky_mask = build_sky_mask(D_dsm)

    base_masks = {
        "ground": ground_mask,
        "roof": roof_mask,
        "sky": sky_mask,
        "reliable": np.isfinite(D_dsm) if reliable_mask is None else reliable_mask,
    }

    vegetation_mask = ~(ground_mask | roof_mask | sky_mask)
    base_masks["vegetation"] = vegetation_mask
    masks = normalize_masks(base_masks, D_dsm.shape)

    a, b = fit_global_linear(D_dp, D_dsm, masks["reliable"])
    D_lin = apply_correction(D_dp, a, b)
    A, B = fit_blockwise_linear(D_dp, D_dsm, masks["reliable"], grid=grid)
    D_corr = apply_correction(D_dp, A, B)
    D_corr = apply_residual_smoothing(D_corr, D_dsm, masks["reliable"], sigma=sigma_residual)

    fused = {
        "ground": fuse_ground(D_dsm, D_corr, masks["ground"], weight=ground_weight),
        "roof": fuse_roof(D_dsm, D_corr, masks["roof"], detail_sigma=roof_detail_sigma, alpha=roof_alpha),
        "vegetation": fuse_vegetation(D_dsm, D_corr, masks["vegetation"], weight=vegetation_weight),
    }

    fallback = np.where(~masks["sky"], D_corr, np.nan)
    D_final = combine_regions(fused, fallback)
    return {
        "D_dsm": D_dsm,
        "D_dp": D_dp,
        "D_linear": D_lin,
        "D_corr": D_corr,
        "D_final": D_final,
        "masks": masks,
    }


def fuse_background_frames(results: Iterable[np.ndarray]) -> np.ndarray:
    return merge_frames_median(list(results))


def run_pipeline(
    images: List[np.ndarray],
    cam: CameraModel,
    dsm: DsmGrid,
    depth_pro: DepthProWrapper,
    fusion_cfg: FusionConfig,
    ray_cfg: Optional[RaycastConfig] = None,
) -> np.ndarray:
    """Run the end-to-end pipeline and return the fused static background depth."""

    ray_cfg = ray_cfg or RaycastConfig()
    single_results = []
    for image in images:
        res = correct_single_frame(
            image,
            cam,
            dsm,
            depth_pro,
            grid=fusion_cfg.grid,
            sigma_residual=fusion_cfg.sigma_residual,
            ground_weight=fusion_cfg.ground_weight,
            roof_alpha=fusion_cfg.roof_alpha,
            roof_detail_sigma=fusion_cfg.detail_sigma,
            vegetation_weight=fusion_cfg.vegetation_weight,
            ray_cfg=ray_cfg,
        )
        single_results.append(res["D_final"])
    return fuse_background_frames(single_results)


if __name__ == "__main__":
    raise SystemExit(
        "This module provides reusable functions; integrate it in your application or notebook to run the pipeline."
    )
