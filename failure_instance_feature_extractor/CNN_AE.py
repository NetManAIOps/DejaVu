import torch as th

from utils.sequential_model_builder import SequentialModelBuilder


class CNNAEFeatureModule(th.nn.Module):
    def __init__(self, input_size: th.Size, embedding_size: int):
        super().__init__()
        encoder_builder = SequentialModelBuilder(input_shape=input_size)
        encoder_builder.add_reshape(
            -1, input_size[-2], input_size[-1],
        ).add_conv_1d(out_channels=10, kernel_size=(3,))
        encoder_builder.add_activation()
        encoder_builder.add_reshape(
            -1, input_size[0], encoder_builder.last_shape[-2] * encoder_builder.last_shape[-1]
        ).add_linear(
            out_features=embedding_size,
        )
        self.encoder = encoder_builder.build()

        decoder_builder = SequentialModelBuilder(input_shape=input_size[:-2] + (embedding_size,))
        decoder_builder.add_linear(
            out_features=encoder_builder.output_shapes[-3][-2] * encoder_builder.output_shapes[-3][-1]
        ).add_activation().add_reshape(
            -1, encoder_builder.output_shapes[-3][-2], encoder_builder.output_shapes[-3][-1]
        ).add_conv_transpose_1d(
            out_channels=input_size[-2], kernel_size=3
        ).add_reshape(-1, input_size[0], input_size[-2], input_size[-1])
        self.decoder = decoder_builder.build()

        self.rec_loss = th.zeros(1, 1, 1)  # (batch_size, n_nodes, n_metrics, n_ts)

    def forward(self, input_x):
        z = self.encoder(input_x)
        rec_x = self.decoder(z)
        self.rec_loss = th.mean(
            th.sqrt(th.sum(th.square(input_x - rec_x), dim=[-1])), dim=-1, keepdim=True
        )
        return th.cat([z, self.rec_loss.detach()], dim=-1)
