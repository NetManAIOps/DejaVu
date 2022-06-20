import torch as th

from utils.sequential_model_builder import SequentialModelBuilder


class GRUAEFeatureModule(th.nn.Module):
    def __init__(self, input_size: th.Size, embedding_size: int, num_layers: int = 1):
        super().__init__()
        self.x_dim = input_size[-2]
        self.n_ts = input_size[-1]
        self.z_dim = embedding_size
        self.num_layers = num_layers

        self.encoder = th.nn.GRU(
            input_size=self.x_dim, hidden_size=self.z_dim, num_layers=num_layers,
        )

        self.decoder = th.nn.GRU(
            input_size=self.z_dim, hidden_size=self.x_dim, num_layers=num_layers,
        )

        self.unify_mapper = SequentialModelBuilder(
            (-1, self.z_dim, self.n_ts)
        ).add_flatten(-2).add_linear(128).add_activation().add_linear(embedding_size).build()

        self.rec_loss = th.zeros(1, 1, 1)

    def forward(self, x):
        z = self.encode(x)
        rec_x = self.decode(z)
        rec_loss = th.mean(
            th.sqrt(th.sum(th.square(x - rec_x), dim=[-1])), dim=-1, keepdim=True
        )  # (batch_size, n_nodes, n_metrics)
        self.rec_loss = rec_loss
        embedding = th.cat([self.unify_mapper(z), rec_loss.detach()], dim=-1)
        return embedding

    @th.jit.export
    def encode(self, input_x: th.Tensor) -> th.Tensor:
        """
        :param input_x: (batch_size, n_nodes, n_metrics, n_ts)
        :return:
        """
        batch_size, n_nodes, _, n_ts = input_x.size()
        x = input_x.view(-1, self.x_dim, n_ts)  # (batch, feat, seq)
        x = th.swapdims(x, 1, 2)  # (batch, seq, feat)
        x = th.swapdims(x, 0, 1)  # (seq, batch, feat)
        assert x.size() == (n_ts, batch_size * n_nodes, self.x_dim)
        z, _ = self.encoder(x)
        assert z.size() == (n_ts, batch_size * n_nodes, self.z_dim)
        z = th.swapdims(z, 0, 1)  # (batch, seq, feat)
        z = th.swapdims(z, 1, 2)  # (batch, feat, seq)
        return z.view(batch_size, n_nodes, self.z_dim, n_ts)

    @th.jit.export
    def decode(self, z: th.Tensor) -> th.Tensor:
        """
        :param z: (batch_size, n_nodes, self.z_dim, n_ts)
        :return:
        """
        batch_size, n_nodes, _, n_ts = z.size()
        z = z.view(-1, self.z_dim, n_ts)  # (batch, feat, seq)
        z = th.swapdims(z, 1, 2)  # (batch, seq, feat)
        z = th.swapdims(z, 0, 1)  # (seq, batch, feat)
        assert z.size() == (n_ts, batch_size * n_nodes, self.z_dim)
        x, _ = self.decoder(z)
        assert x.size() == (n_ts, batch_size * n_nodes, self.x_dim)
        x = th.swapdims(x, 0, 1)  # (batch, seq, feat)
        x = th.swapdims(x, 1, 2)  # (batch, feat, seq)
        return x.view(batch_size, n_nodes, self.x_dim, n_ts)