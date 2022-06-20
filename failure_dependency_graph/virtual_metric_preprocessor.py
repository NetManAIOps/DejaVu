from typing import Optional
import torch as th
from failure_dependency_graph import FDG
from metric_preprocess import MetricPreprocessor


class FakeMetricPreprocessor(MetricPreprocessor):
    @staticmethod
    def extract_features(
            fdg: FDG,
            start_ts: int, length: int, granularity: int = 60,
            fill_na: bool = True, clip_value: Optional[float] = 10.
    ):
        timestamp_list = [start_ts + i * granularity for i in range(length)]
        timestamp_2_idx = {ts: idx for idx, ts in enumerate(timestamp_list)}
        features_list = [
            th.zeros((len(fdg.failure_instances[fc]), len(fdg.FC_metrics_dict[fc]), length), dtype=th.float32)
            for fc in fdg.failure_classes
        ]
        return features_list, timestamp_2_idx