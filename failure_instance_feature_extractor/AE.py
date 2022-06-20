from collections import namedtuple

import torch as th
from torch.distributions import Normal

from utils.sequential_model_builder import SequentialModelBuilder


class AEFeatureModule(th.nn.Module):
    def __init__(self, input_size: th.Size, embedding_size: int):
        super().__init__()
        self.n_metrics = input_size[-2]
        self.n_ts = input_size[-1]
        self.z_dim = 8
        self.module_list = th.nn.ModuleList(
            [MetricVAE(x_dim=self.n_ts, z_dim=self.z_dim) for _ in range(self.n_metrics)])

        self.unify_mapper = SequentialModelBuilder(
            (-1, self.n_metrics, self.z_dim)
        ).add_flatten(-2).add_linear(128).add_activation().add_linear(embedding_size).build()

        self.rec_loss = th.zeros(1, 1, 1, 1)

    def forward(self, input_x):
        z = th.zeros(input_x.size()[:-1] + (self.z_dim,), dtype=input_x.dtype, device=input_x.device)
        rec_loss = th.zeros(input_x.size()[:-1], dtype=input_x.dtype, device=input_x.device)
        for i in range(self.n_metrics):
            m_ret = self.module_list[i](input_x[..., i, :])
            z[..., i, :] = th.mean(m_ret.z_mean, dim=0)
            rec_loss[..., i] = th.mean(m_ret.ELBO, dim=0)
        self.rec_loss = th.mean(rec_loss, dim=-1, keepdim=True)
        return th.cat([self.unify_mapper(z), self.rec_loss.detach()], dim=-1)


class MetricVAE(th.nn.Module):
    ret_type = namedtuple('MetricAERet', ['ELBO', 'x_mean', 'x_std', 'z_mean', 'z_std', 'z_samples'])

    def __init__(self, x_dim: int, z_dim: int):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim

        encoder_builder = SequentialModelBuilder(input_shape=(-1, x_dim))
        encoder_builder.add_linear(16)
        encoder_builder.add_activation()
        # encoder_builder.add_linear(16)
        # encoder_builder.add_activation()
        self.encoder = encoder_builder.build()
        self.z_mean = SequentialModelBuilder(encoder_builder.last_shape).add_linear(z_dim).build()
        self.z_std = SequentialModelBuilder(encoder_builder.last_shape).add_linear(
            z_dim
        ).add_activation('softplus').build()

        decoder_builder = SequentialModelBuilder(input_shape=(-1, z_dim))
        decoder_builder.add_linear(16)
        decoder_builder.add_activation()
        # decoder_builder.add_linear(x_dim)
        # decoder_builder.add_activation()
        self.decoder = decoder_builder.build()
        self.x_mean = SequentialModelBuilder(decoder_builder.last_shape).add_linear(x_dim).build()
        self.x_std = SequentialModelBuilder(decoder_builder.last_shape).add_linear(
            x_dim
        ).add_activation('softplus').build()

    def ELBO(self, x: th.Tensor, z_samples: int = 1) -> 'MetricVAE.ret_type':
        z_mean, z_std = self.encode(x)
        z = th.randn(
            (z_samples,) + z_mean.size(), device=x.device, dtype=x.dtype
        ) * z_std.unsqueeze(dim=0) + z_mean.unsqueeze(dim=0)  # (z_samples, ..., z_dim)
        x_mean, x_std = self.decode(z)
        log_px_z = th.sum(Normal(loc=x_mean, scale=x_std).log_prob(x), dim=-1)
        log_p_z = th.sum(Normal(0, 1).log_prob(z), dim=-1)
        log_qz_x = th.sum(Normal(z_mean, z_std).log_prob(z), dim=-1)
        elbo = - (log_px_z + log_p_z - log_qz_x)
        return self.ret_type(ELBO=elbo, z_mean=z_mean, z_std=z_std, z_samples=z, x_mean=x_mean, x_std=x_std)

    def forward(self, x):
        return self.ELBO(x)

    def encode(self, x):
        z_state = self.encoder(x)
        z_mean = self.z_mean(z_state)
        z_std = self.z_std(z_state) + 1e-4
        return z_mean, z_std

    def decode(self, z):
        x_state = self.decoder(z)
        x_mean = self.x_mean(x_state)
        x_std = self.x_std(x_state) + 1e-4
        return x_mean, x_std