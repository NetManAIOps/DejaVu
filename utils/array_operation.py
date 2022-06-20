import numpy as np
import torch as th


def forward_fill_na(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    if axis < 0:
        axis = len(arr.shape) + axis
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [np.arange(k)[tuple([slice(None) if dim == i else np.newaxis
                               for dim in range(len(arr.shape))])]
           for i, k in enumerate(arr.shape)]
    slc[axis] = idx
    return arr[tuple(slc)]


def intersection1d(t1, t2):
    combined = th.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection


def difference1d(t1, t2):
    combined = th.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    return difference
