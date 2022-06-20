import torch as th

from utils.sequential_model_builder import SequentialModelBuilder


class CNNFeatureModule(th.nn.Module):
    def __init__(self, input_size: th.Size, embedding_size: int, has_dropout: bool = False):
        """
        :param input_size:  (batch_size, channel, x_dim)
        :param embedding_size:
        :param has_dropout:
        """
        super().__init__()
        self.input_size = input_size
        builder = SequentialModelBuilder(input_shape=input_size)
        if has_dropout:
            builder.add_dropout(0.2)
        builder.add_reshape(
            -1, input_size[-2], input_size[-1],
        ).add_conv_1d(out_channels=10, kernel_size=(3,))
        builder.add_activation()

        # builder.add_dropout(0.2)
        # builder.add_reshape(
        #                 -1, 1, input_size[-2], input_size[-1],
        #             ).add_conv_2d(
        #     out_channels=64, kernel_size=(input_size[-2], 3), padding=0,
        # ).add_activation()
        # builder.add_dropout()
        # builder.add_reshape(-1, builder.last_shape[-3], builder.last_shape[-1])
        # builder.add_max_pool_1d(kernel_size=builder.last_shape[-1])
        builder.add_reshape(
            -1, input_size[0], builder.last_shape[-2] * builder.last_shape[-1]
        )
        if has_dropout:
            builder.add_dropout(0.5)
        builder.add_linear(
            out_features=embedding_size,
        )
        # builder.add_activation().add_linear(feature_size)
        # builder.add_activation().add_linear(out_features=feature_size)
        self.builder = builder
        self.module = builder.build()

    def forward(self, input_x):
        return self.module(input_x)