from typing import List, Tuple
import torch as th
from einops import rearrange

from utils.sequential_model_builder import SequentialModelBuilder
from itertools import accumulate


__all__ = ['FeatureExtractorDecoder']


# class SingleDecoder(th.nn.Module):
#     def __init__(self, input_size: th.Size, embedding_size: int, window_size: Tuple[int, int] = (10, 10)):
#         super().__init__()
#         mapper_builder = SequentialModelBuilder(input_shape=input_size[:-2] + (embedding_size,))
#         kernel_size = 3
#         mapper_builder.add_reshape(
#             -1, 10, sum(window_size) + 1 - kernel_size
#         ).add_conv_transpose_1d(
#             out_channels=embedding_size, kernel_size=kernel_size,
#         ).add_activation(
#         ).add_reshape(-1, input_size[-3], embedding_size, input_size[-1])
#         self.unify_mapper = mapper_builder.build()
#         self.decoder = th.nn.GRU(
#             input_size=embedding_size, hidden_size=input_size[-2], num_layers=1,
#         )
#
#     def forward(self, z):
#         rec_x = self.decode(z)
#         return rec_x
#
#     def decode(self, input_x: th.Tensor) -> th.Tensor:
#         """
#         :param input_x: (batch_size, n_nodes, n_metrics, n_ts)
#         :return: (batch_size, n_nodes, self.z_dim, n_ts)
#         """
#         input_x = self.unify_mapper(input_x)
#         x = rearrange(input_x, "b n m t -> t (b n) m")
#         z, _ = self.decoder(x)
#         return rearrange(z, "t (b n) z -> b n z t", b=input_x.shape[0])


class SingleDecoder(th.nn.Module):
    def __init__(self, input_size: th.Size, embedding_size: int, window_size: Tuple[int, int] = (10, 10)):
        super().__init__()
        self.n_ts = sum(window_size)
        self.decoder = th.nn.GRU(
            input_size=embedding_size, hidden_size=input_size[-2], num_layers=1,
        )

    def forward(self, z):
        batch = z.shape[0]
        instance = z.shape[1]
        rec_x = self.decoder(rearrange(z, "batch instance (z ts) -> ts (batch instance) z", ts=self.n_ts))[0]
        # print(f"{rec_x.shape=}")
        rec_x = rearrange(rec_x, "ts (batch instance) x ->batch instance x ts", batch=batch, instance=instance)
        return rec_x


class FeatureExtractorDecoder(th.nn.Module):
    def __init__(self, input_size_list: List[th.Size], embedding_size: int):
        super().__init__()
        self.module_list = th.nn.ModuleList(
            [SingleDecoder(_, embedding_size) for _ in input_size_list]
        )
        self.node_bins = list(accumulate([0] + [_[-3] for _ in input_size_list]))
        self.input_size_list = input_size_list
        assert len(self.node_bins) == len(self.module_list) + 1

    def forward(self, z: th.Tensor) -> List[th.Tensor]:
        rets = []
        for left, right, m, size in zip(
                self.node_bins[:-1], self.node_bins[1:], self.module_list, self.input_size_list
        ):
            rec_x = m(z[..., left:right, :])
            assert rec_x.size()[-len(size):] == size, f"{rec_x.size()=} {size=}"
            rets.append(rec_x)
        return rets
