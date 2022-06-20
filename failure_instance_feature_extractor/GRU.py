import torch as th
from einops import rearrange

from utils.sequential_model_builder import SequentialModelBuilder


class GRUFeatureModule(th.nn.Module):
    def __init__(self, input_size: th.Size, embedding_size: int, num_layers: int = 1):
        super().__init__()
        self.x_dim = input_size[-2]
        self.n_ts = input_size[-1]
        self.n_instances = input_size[-3]
        self.z_dim = embedding_size
        self.num_layers = num_layers

        self.encoder = th.nn.GRU(
            input_size=self.x_dim, hidden_size=self.z_dim, num_layers=num_layers,
        )

        # self.unify_mapper = SequentialModelBuilder(
        #     (-1, self.z_dim, self.n_ts)
        # ).add_flatten(-2).add_linear(128).add_activation().add_linear(embedding_size).build()
        self.unify_mapper = SequentialModelBuilder(
            (-1, self.n_instances, self.z_dim, self.n_ts), debug=False,
        ).add_reshape(
            -1, self.z_dim, self.n_ts,
        ).add_conv_1d(
            out_channels=10, kernel_size=(3,)
        ).add_activation().add_flatten(start_dim=-2).add_linear(embedding_size).add_reshape(
            -1, self.n_instances, embedding_size,
        ).build()

    def forward(self, x):
        z = self.encode(x)
        embedding = th.cat([self.unify_mapper(z)], dim=-1)
        return embedding

    def encode(self, input_x: th.Tensor) -> th.Tensor:
        """
        :param input_x: (batch_size, n_nodes, n_metrics, n_ts)
        :return: (batch_size, n_nodes, self.z_dim, n_ts)
        """
        batch_size, n_nodes, _, n_ts = input_x.size()
        x = rearrange(input_x, "b n m t -> t (b n) m", b=batch_size, n=n_nodes, m=self.x_dim, t=n_ts)
        assert x.size() == (n_ts, batch_size * n_nodes, self.x_dim)
        z, _ = self.encoder(x)
        return rearrange(z, "t (b n) z -> b n z t", b=batch_size, n=n_nodes, z=self.z_dim, t=n_ts)
