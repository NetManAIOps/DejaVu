import re

import regex
from pathlib import Path
from typing import Union, Optional, Callable, Dict

from loguru import logger
from pytorch_lightning.utilities.cloud_io import load
import torch as th
from torch.nn import Module

__all__ = ['best_checkpoint', 'load_weights_from_checkpoint']


def best_checkpoint(
        path: Union[str, Path], metric: str = 'val_loss', mode: str = 'min', debug: bool = False
) -> Path:
    if debug:
        logger.debug(f"{path=} {metric=} {mode=}")
    if mode == 'min':
        cmp = lambda a, b: a < b
    elif mode == 'max':
        cmp = lambda a, b: a > b
    else:
        raise RuntimeError(f"Unknown {mode=}")
    path = Path(path)
    best_model_path = None
    best_metric = float('inf') if mode == 'min' else float('-inf')

    metric_pattern = r'([A-Za-z0-9_\-@]+=[0-9.]+)'
    name_pattern = re.compile(rf'{metric_pattern}{"".join([rf"(?:-{metric_pattern})?" for _ in range(100)])}\.ckpt')
    for ckpt_path in path.glob("**/**/*.ckpt"):
        match = name_pattern.match(ckpt_path.name)
        if not match:
            logger.warning(f"{ckpt_path=} not match")
            continue
        # noinspection PyTypeChecker
        metrics = dict(tuple(p.split('=')) for p in match.groups() if p is not None)
        if debug:
            logger.debug(f"{ckpt_path=} has metrics: {metrics}")
        if metric not in metrics:
            continue
        if cmp(_m := float(metrics[metric]), best_metric):
            best_metric = _m
            best_model_path = ckpt_path
    if debug:
        logger.debug(f"best ckpt: {best_model_path=}")
    return best_model_path


def load_weights_from_checkpoint(
        model: Module, checkpoint_path: Union[str, Path],
        map_location: Optional[
            Union[str, Callable, th.device, Dict[Union[str, th.device], Union[str, th.device]]]
        ] = None,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    return model
