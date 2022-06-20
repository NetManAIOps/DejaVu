from typing import Optional

import torch as th


@th.jit.script
def add_missing_to_tensor(t: th.Tensor, number: int = 1) -> th.Tensor:
    t = t.detach().clone()
    ori_size = t.size()
    t = t.view(-1, ori_size[-1])
    pos = th.randint(1, ori_size[-1], size=(t.size()[0], number), dtype=th.long)
    t[th.arange(t.size()[0]).unsqueeze(-1), pos] = t[th.arange(t.size()[0], dtype=th.long).unsqueeze(-1), pos - 1]
    if len(ori_size) == 3:
        return t.view(ori_size[0], ori_size[1], ori_size[2])
    elif len(ori_size) == 1:
        return t.view(ori_size[0])
    elif len(ori_size) == 2:
        return t.view(ori_size[0], ori_size[1])
    elif len(ori_size) == 4:
        return t.view(ori_size[0], ori_size[1], ori_size[2], ori_size[3])
    else:
        raise RuntimeError(f"Unexpected ori_size=({','.join([str(_) for _ in ori_size])})")


@th.jit.script
def add_spike_to_tensor(
        t: th.Tensor, number: int = 1, median: Optional[th.Tensor] = None, mad: Optional[th.Tensor] = None
) -> th.Tensor:
    t = t.detach().clone()
    ori_size = t.size()
    t = t.view(-1, ori_size[-1])
    pos = th.randint(1, ori_size[-1], size=(t.size()[0], number), dtype=th.long)
    if median is None:
        median = th.median(t, dim=1, keepdim=True).values
    if mad is None:
        mad = th.median(th.abs(t - median), dim=1, keepdim=True).values
    t[
        th.arange(t.size()[0], dtype=th.long).unsqueeze(-1), pos
    ] = ((th.randn_like(pos.float()) - 0.5) * 20 + 3) * mad + median
    if len(ori_size) == 3:
        return t.view(ori_size[0], ori_size[1], ori_size[2])
    elif len(ori_size) == 1:
        return t.view(ori_size[0])
    elif len(ori_size) == 2:
        return t.view(ori_size[0], ori_size[1])
    elif len(ori_size) == 4:
        return t.view(ori_size[0], ori_size[1], ori_size[2], ori_size[3])
    else:
        raise RuntimeError(f"Unexpected ori_size=({','.join([str(_) for _ in ori_size])})")