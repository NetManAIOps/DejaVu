import torch as th
from torch import nn
from typing import Tuple

from utils.sequential_model_builder import SequentialModelBuilder


class GRUVAEFeatureModule(th.nn.Module):
    def __init__(self, input_size: th.Size, embedding_size: int, num_layers: int = 1):
        super().__init__()
        self.x_dim = input_size[-2]
        self.n_ts = input_size[-1]
        self.z_dim = embedding_size
        self.num_layers = num_layers

        hidden_size = 128
        self.encoder = th.nn.GRU(
            input_size=self.x_dim, hidden_size=hidden_size, num_layers=num_layers,
        )
        z_h_builder = SequentialModelBuilder(input_shape=(-1, hidden_size)).add_linear(
            hidden_size,
        ).add_activation()
        self.z_h = z_h_builder.build()
        self.z_mean = nn.Linear(z_h_builder.last_shape[-1], embedding_size)
        self.z_std = nn.Sequential(nn.Linear(z_h_builder.last_shape[-1], embedding_size), nn.Softplus())

        self.decoder = th.nn.GRU(
            input_size=self.z_dim, hidden_size=hidden_size, num_layers=num_layers,
        )
        x_h_builder = SequentialModelBuilder(input_shape=(-1, hidden_size)).add_linear(
            hidden_size,
        ).add_activation()
        self.x_h = x_h_builder.build()
        self.x_mean = nn.Linear(x_h_builder.last_shape[-1], self.x_dim)
        self.x_std = nn.Sequential(nn.Linear(x_h_builder.last_shape[-1], self.x_dim), nn.Softplus())

        self.unify_mapper = SequentialModelBuilder(
            (-1, self.z_dim, self.n_ts)
        ).add_flatten(-2).add_linear(hidden_size).add_activation().add_linear(embedding_size).build()

        self.rec_loss = th.zeros(1, 1, 1)

    def forward(self, x):
        z_mean, z_std = self.encode(x)
        if self.training:
            sample_size = 1
        else:
            sample_size = 64
        z = th.randn(
            (sample_size,) + (1,) * len(z_mean.size()), device=x.device
        ) * z_std.unsqueeze(0) + z_mean.unsqueeze(0)
        x_mean, x_std = self.decode(z)
        log_p_x_z = th.sum(-0.5 * th.square((x.unsqueeze(0) - x_mean) / x_std) - th.log(x_std), dim=[-1, -2])
        log_p_z = th.sum(-0.5 * th.square(z), dim=[-1, -2])
        log_q_z_x = th.sum(
            -0.5 * th.square((z - z_mean.unsqueeze(0)) / z_std.unsqueeze(0)) - th.log(z_std),
            dim=[-1, -2]
        )
        rec_loss = th.mean(
            th.sqrt(th.sum(th.square(x.unsqueeze(0) - x_mean), dim=[-1])), dim=[0, -1]
        )  # (batch_size, n_nodes, n_metrics)
        # self.rec_loss = - th.mean(log_p_x_z + log_p_z - log_q_z_x, dim=0)
        self.rec_loss = rec_loss
        embedding = th.cat([self.unify_mapper(z_mean), rec_loss.detach().unsqueeze(-1)], dim=-1)
        return embedding

    @th.jit.export
    def encode(self, input_x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :param input_x: (batch_size, n_nodes, n_metrics, n_ts)
        :return:
        """
        batch_size, n_nodes, _, n_ts = input_x.size()
        x = input_x.view(-1, self.x_dim, n_ts)  # (batch, feat, seq)
        x = th.swapdims(x, 1, 2)  # (batch, seq, feat)
        x = th.swapdims(x, 0, 1)  # (seq, batch, feat)
        assert x.size() == (n_ts, batch_size * n_nodes, self.x_dim)
        z_h, _ = self.encoder(x)
        z_h = self.z_h(z_h)  # (seq, batch, feat)
        z_mean = self.z_mean(z_h)
        z_std = self.z_std(z_h) + 1e-3
        assert z_mean.size() == (n_ts, batch_size * n_nodes, self.z_dim)
        assert z_std.size() == (n_ts, batch_size * n_nodes, self.z_dim)
        z_mean = th.swapdims(z_mean, -3, -2)  # (batch, seq, feat)
        z_mean = th.swapdims(z_mean, -2, -1)  # (batch, feat, seq)
        z_mean = z_mean.view(batch_size, n_nodes, self.z_dim, n_ts)
        z_std = th.swapdims(z_std, -3, -2)  # (batch, seq, feat)
        z_std = th.swapdims(z_std, -2, -1)  # (batch, feat, seq)
        z_std = z_std.view(batch_size, n_nodes, self.z_dim, n_ts)
        return z_mean, z_std

    @th.jit.export
    def decode(self, z: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :param z: (sample_size, batch_size, n_nodes, self.z_dim, n_ts)
        :return:
        """
        sample_size, batch_size, n_nodes, _, n_ts = z.size()
        z = z.reshape(-1, self.z_dim, n_ts)  # (batch, feat, seq)
        z = th.swapdims(z, 1, 2)  # (batch, seq, feat)
        z = th.swapdims(z, 0, 1)  # (seq, batch, feat)
        assert z.size() == (n_ts, sample_size * batch_size * n_nodes, self.z_dim)
        x_h, _ = self.decoder(z)
        x_h = self.x_h(x_h)
        x_mean = self.x_mean(x_h)
        x_std = self.x_std(x_h) + 1e-3
        assert x_mean.size() == (n_ts, sample_size * batch_size * n_nodes, self.x_dim)
        assert x_std.size() == (n_ts, sample_size * batch_size * n_nodes, self.x_dim)
        x_mean = th.swapdims(x_mean, -3, -2)  # (batch, seq, feat)
        x_mean = th.swapdims(x_mean, -2, -1)  # (batch, feat, seq)
        x_mean = x_mean.view(sample_size, batch_size, n_nodes, self.x_dim, n_ts)
        x_std = th.swapdims(x_std, -3, -2)  # (batch, seq, feat)
        x_std = th.swapdims(x_std, -2, -1)  # (batch, feat, seq)
        x_std = x_std.view(sample_size, batch_size, n_nodes, self.x_dim, n_ts)
        return x_mean, x_std
