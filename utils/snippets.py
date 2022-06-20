import io
from contextlib import contextmanager
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch as th
from loguru import logger
from pyprof import Profiler
from tap import Tap
from torch import nn


def merge_dataloader_results(
        train_ret, valid_ret, test_ret,
        train_ids: List[int], valid_ids: List[int], test_ids: List[int],
):
    ret = np.empty((len(train_ret) + len(valid_ret) + len(test_ret),), dtype=object)
    ret[:] = list(train_ret) + list(valid_ret) + list(test_ret)
    failure_ids = np.concatenate([train_ids, valid_ids, test_ids])
    ret = ret[np.argsort(failure_ids)]
    if isinstance(train_ret, list):
        ret = list(ret)
    elif isinstance(train_ret, th.Tensor):
        ret = th.vstack(list(ret)).to(dtype=train_ret.dtype, device=train_ret.device)
    else:
        ret = np.asarray(ret, dtype=train_ret.dtype)
    return ret


def count_parameters(model: th.nn.Module) -> int:
    total_params = 0
    ret = io.StringIO()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        print(f"|{name:<80} |{param:<10.0f}|", file=ret)
        total_params += param
    logger.info(
        "\n========================================================================\n"
        f"{ret.getvalue()}\n"
        f"Total Trainable Params: {total_params}\n"
        "\n========================================================================\n"
    )
    logger.info(f"Total Trainable Params: {total_params}")
    return total_params


def plot_module_with_dot(model: nn.Module, init_data, output_dir: Path = Path('.'), name: str = "model"):
    logger.info(f"Plot model with dot in {output_dir=}")
    from torchviz import make_dot
    make_dot(model(init_data), params=dict(model.named_parameters())).render(
        output_dir / f"{name}_architecture", cleanup=True, format='pdf'
    )


def tanh_estimator(x):
    return (0.5 * np.tanh(0.1 * x) + 0.5) * 20. - 10.


def command_output_one_line_summary(metrics: List[float], profiler: Profiler, config: 'Tap') -> str:
    assert hasattr(config, 'output_dir'), f"{type(config)=} is not a FDGBaseConfig"
    return (
        f"command output one-line summary: "
        f"{','.join([f'{_:.2f}' for _ in metrics])},"
        f"{profiler.total},"
        f"{getattr(config, 'output_dir')},"
        f"{config.get_reproducibility_info()['command_line']},"
        f"{config.get_reproducibility_info().get('git_url', '')}"
    )


@contextmanager
def disable_logger_context(name: str):
    logger.disable(name)
    yield
    logger.enable(name)


def read_dataframe(path: Union[str, Path]):
    path = str(path)
    if path.endswith("csv"):
        return pd.read_csv(path)
    elif path.endswith("pkl"):
        return pd.read_pickle(path)
    else:
        return pd.read_csv(path)
