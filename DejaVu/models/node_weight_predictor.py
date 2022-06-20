import torch as th

from utils.sequential_model_builder import SequentialModelBuilder


class NodeWeightPredictor(th.nn.Module):
    def __init__(
            self,
            feature_size: int,
            has_dropout: bool = False,
    ):
        super(NodeWeightPredictor, self).__init__()
        predictor_builder = SequentialModelBuilder(input_shape=(-1, feature_size,))
        if has_dropout:
            predictor_builder.add_dropout(0.5)
        predictor_builder.add_linear(
            128, bias=True
        ).add_activation()
        if has_dropout:
            predictor_builder.add_dropout(0.5)
        predictor_builder.add_linear(
            1, bias=False
        )
        self.predictor = predictor_builder.build()

    def forward(self, x: th.Tensor):
        return th.squeeze(self.predictor(x), dim=-1)
