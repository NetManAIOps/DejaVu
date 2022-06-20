from collections import defaultdict
from functools import partial
from typing import Set, List, Any, Optional

import numpy as np


def top_k_accuracy(y_true: List[Set[Any]], y_pred: List[List[Any]], k=1, printer=None):
    assert len(y_true) == len(y_pred)
    cnt = 0
    for a, b in zip(y_true, y_pred):
        left = a
        right = set(b[:k])
        if left <= right:
            cnt += 1
        else:
            if printer:
                printer(f"expected: {left}, actual: {right}")
    return cnt / max(len(y_true), 1)


top_1_accuracy = partial(top_k_accuracy, k=1)
top_2_accuracy = partial(top_k_accuracy, k=2)
top_3_accuracy = partial(top_k_accuracy, k=3)


def get_rank(y_true: Set[Any], y_pred: List[Any], max_rank: Optional[int] = None) -> List[float]:
    rank_dict = defaultdict(lambda: len(y_pred) + 1 if max_rank is None else (max_rank + len(y_pred)) / 2)
    for idx, item in enumerate(y_pred, start=1):
        if item in y_true:
            rank_dict[item] = idx
    return [rank_dict[_] for _ in y_true]


# noinspection PyPep8Naming
def MFR(y_true: List[Set[Any]], y_pred: List[List[Any]], max_rank: Optional[int] = None):
    return np.mean([
        np.min(get_rank(a, b, max_rank))
        for a, b in zip(y_true, y_pred)
    ])


# noinspection PyPep8Naming
def MAR(y_true: List[Set[Any]], y_pred: List[List[Any]], max_rank: Optional[int] = None):
    return np.mean([
        np.mean(get_rank(a, b, max_rank))
        for a, b in zip(y_true, y_pred)
    ])


def get_evaluation_metrics_dict(y_true: List[Set[Any]], y_pred: List[List[Any]], max_rank: Optional[int] = None):
    metrics = {
        "A@1": top_1_accuracy(y_true, y_pred),
        "A@2": top_2_accuracy(y_true, y_pred),
        "A@3": top_3_accuracy(y_true, y_pred),
        "A@5": top_k_accuracy(y_true, y_pred, k=5),
        "MAR": MAR(y_true, y_pred, max_rank=max_rank),
    }
    return metrics


rca_evaluation_metrics = {
    "A@1": top_1_accuracy,
    "A@2": top_2_accuracy,
    "A@3": top_3_accuracy,
    "MAR": MAR,
    "MFR": MFR,
}

__all__ = [
    'rca_evaluation_metrics',
    "top_1_accuracy",
    "top_2_accuracy",
    "top_3_accuracy",
    "top_k_accuracy",
    "MAR", "MFR",
    "get_evaluation_metrics_dict",
]
