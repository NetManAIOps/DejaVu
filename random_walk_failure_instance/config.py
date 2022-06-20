from typing import Literal

from failure_dependency_graph import FDGBaseConfig


class RandomWalkFailureInstanceConfig(FDGBaseConfig):
    backward_weight: float = 0.1

    corr_aggregation_method: Literal['sum', 'max', 'min', 'mean'] = 'mean'
    anomaly_score_aggregation_method: Literal['sum', 'max', 'min', 'mean'] = 'mean'
