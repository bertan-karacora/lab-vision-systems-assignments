import numpy as np


def create_epochs_continuous(epochs, nums_samples):
    """Convert array of epoch number per iteration to a continuous epoch scale."""
    nums_samples_cumulative = np.cumsum(nums_samples)
    nums_samples_per_epoch = np.sum(np.split(nums_samples, np.unique(epochs, return_index=True)[1][1:]), axis=-1)
    epochs_continuous = nums_samples_cumulative / nums_samples_per_epoch[epochs - 1]
    return epochs_continuous


def smooth(f, k=5):
    """Smoothing a function using a low-pass filter (mean) of size K"""
    kernel = np.ones(k) / k
    # Account for boundaries
    f = np.concatenate([f[: int(k // 2)], f, f[int(-k // 2) :]])
    smooth_f = np.convolve(f, kernel, mode="same")
    # Remove boundary-fixes
    smooth_f = smooth_f[k // 2 : -k // 2]
    return smooth_f
