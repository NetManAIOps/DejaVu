from typing import Literal

from failure_dependency_graph import FDGBaseConfig


class RandomWalkSingleMetricConfig(FDGBaseConfig):
    CI_method: Literal['GES', 'LiNGAM', 'fGES'] = 'fGES'
    corr_window_size: int = 10

    score_aggregation_method: Literal['sum', 'max', 'min', 'mean'] = 'sum'

    backward_weight: float = 0.1

    unchanged_metric_threshold: float = 1e-2

    flush_causal_graph_cache: bool = True
