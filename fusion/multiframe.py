import numpy as np


def merge_frames_median(depth_maps):
    stack = np.stack(depth_maps, axis=0)
    return np.nanmedian(stack, axis=0)


def merge_frames_mean(depth_maps):
    stack = np.stack(depth_maps, axis=0)
    return np.nanmean(stack, axis=0)
