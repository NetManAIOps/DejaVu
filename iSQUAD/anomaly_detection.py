import numpy as np
import torch as th
from scipy.stats import t


def robust_threshold(
        kpi: th.Tensor, dim: int = -1, upper_limit: float = 0.97, lower_limit: float = 0.03, m: float = 1,
        window_size: int = 60,
) -> th.Tensor:
    pi = 3.1415926
    if dim < 0:
        dim = len(kpi.size()) + dim
    x_windows = kpi.unfold(dim, window_size, 1).float()  # (..., window_size)
    x = x_windows[..., -1]  # (..., window_size)
    median = th.median(x_windows, dim=-1).values  # (...)
    mad = th.median(th.abs(x_windows - th.unsqueeze(median, -1)), dim=-1).values  # (...)
    t_x = x.clone()
    # We do not need this, because our input data is already normalized
    # t_x[x < m] = 2 * m / pi * th.tan(pi * (x[x < m] - m) / (2 * m)) + m
    cdf = th.arctan((t_x - median) / th.clip(mad, min=th.full_like(mad, 1e-4))) / pi + 0.5
    ret = th.zeros_like(x)
    ret[cdf > upper_limit] = 1
    ret[cdf < lower_limit] = -1
    ret = th.cat([
        th.zeros(
            kpi.size()[:dim] + (window_size - 1,) + (
                kpi.size()[dim + 1:] if dim < len(kpi.size()) else th.Size(())
            ),
            device=ret.device,
            dtype=ret.dtype,
        ),
        ret,
    ], dim=dim)
    assert ret.size() == kpi.size()
    # move back to the original device
    return ret


def t_test(
        kpi: th.Tensor, window_size: int = 10,
        dim: int = -1, significance_level: float = 0.05,
) -> th.Tensor:
    if dim < 0:
        dim = len(kpi.size()) + dim
    data1 = th.index_select(kpi, dim, th.arange(0, kpi.size()[dim] - window_size))
    data2 = th.index_select(kpi, dim, th.arange(kpi.size()[dim] - window_size, kpi.size()[dim]))
    l1 = data1.size()[dim]
    l2 = data2.size()[dim]
    df = l1 + l2 - 2
    # robust
    # mean1 = th.median(data1, dim=dim).values
    # std1 = th.median(th.abs(data1 - mean1[:, None]), dim=dim).values
    # mean2 = th.median(data2, dim=dim).values
    # std2 = th.median(th.abs(data2 - mean2[:, None]), dim=dim).values
    # original
    (std1, mean1), (std2, mean2) = th.std_mean(data1, dim=dim), th.std_mean(data2, dim=dim)
    sed = th.clip(
        th.sqrt(
            ((l1 - 1) * th.square(std1) + (l2 - 1) * th.square(std2)) / df
        ),
        min=th.tensor(1e-4)
    ) * th.sqrt(th.tensor(1 / l1 + 1 / l2))
    t_stat = (mean1 - mean2) / sed
    p = th.from_numpy(np.array((1.0 - t.cdf(th.abs(t_stat).detach().numpy(), df)) * 2.0)).to(kpi.device)
    return (p < significance_level).float() * th.sign(t_stat)
