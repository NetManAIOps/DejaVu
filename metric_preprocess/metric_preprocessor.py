from functools import lru_cache, cached_property
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import torch as th
from pyprof import profile
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm

__all__ = [
    "MetricPreprocessor",
    "get_input_tensor_size_dict",
    "get_input_tensor_size_list",
]

from failure_dependency_graph import FDG
from utils.array_operation import forward_fill_na


class MetricPreprocessor:
    def __init__(self, fdg: FDG, granularity=60, fill_na: bool = True, clip_value: Optional[float] = 10.):
        self._granularity = granularity
        self._fill_na = fill_na
        self._clip_value = clip_value

        # 3600是考虑到了window size。window size不能太大
        self._max_window_size = 3600 * 2
        start_ts = fdg.timestamp_range[0] - self._max_window_size
        end_ts = fdg.timestamp_range[1] + self._max_window_size
        self._features_list, self._timestamp_2_idx = self.extract_features(
            fdg,
            start_ts=start_ts,
            length=(end_ts - start_ts) // granularity + 1,
            granularity=granularity,
            fill_na=fill_na,
            clip_value=clip_value,
        )

        from einops import rearrange
        self._flatten_features = th.cat(
            [
                rearrange(_, "n m t -> (n m) t")
                for _ in self.features_list
            ]
        )
        self._flatten_metric_instance_gids = th.cat([
            th.tensor([gid for _ in fdg.FI_metrics_dict[fdg.gid_to_instance(gid)]])
            for gid in range(fdg.n_failure_instances)
        ])
        self._flatten_metric_names = np.concatenate([
            np.array([_ for _ in fdg.FI_metrics_dict[fdg.gid_to_instance(gid)]])
            for gid in range(fdg.n_failure_instances)
        ])
        self._flatten_anomaly_direction_constraint = th.tensor([
            {'u': 1, 'd': -1, 'b': 0}[fdg.anomaly_direction_constraint[_.split('##')[-1]]]
            for _ in self._flatten_metric_names
        ])

    def to(self, device: Union[str, th.device]) -> 'MetricPreprocessor':
        self._features_list = move_data_to_device(self._features_list, device)
        return self

    @property
    def features_list(self) -> List[th.Tensor]:
        """
        :return: A list of features extracted from raw metrics by sliding window,
        each element of which is a tensor of the shape (n_failure_instance, n_metrics, n_timestamps)
        """
        return self._features_list

    @property
    def flatten_features(self) -> th.Tensor:
        """
        :return: A tensor of the shape ((n_failure_classes, n_failure_instances, n_metrics), n_timestamps)
        """
        return self._flatten_features

    @property
    def flatten_metric_instance_gids(self) -> th.Tensor:
        """
        :return: The corresponding instance gids of the flattened features tensor
        """
        return self._flatten_metric_instance_gids

    @property
    def flatten_anomaly_direction_constraint(self):
        """
        :return: +1 for up, -1 for down, 0 for both
        """
        return self._flatten_anomaly_direction_constraint

    @property
    def flatten_metric_names(self) -> np.ndarray:
        """
        :return: The corresponding metric names of the flattened features tensor
        """
        return self._flatten_metric_names

    @cached_property
    def timestamps(self) -> np.ndarray:
        return np.array(sorted(self._timestamp_2_idx.keys()))

    @lru_cache
    def get_timestamp_indices(self, fault_ts: int, window_size: Tuple[int, int] = 5):
        """
        :param fault_ts:
        :param window_size:
        :return: The indices of the corresponding timestamps in the features tensor
        """
        assert window_size[0] <= self._max_window_size and window_size[1] <= self._max_window_size, \
            f"{self._max_window_size=} {window_size=}"

        start_ts = fault_ts - window_size[0] * self._granularity
        length = sum(window_size)
        timestamp_list = [start_ts + i * self._granularity for i in range(length)]
        ts_idx = np.asarray([self._timestamp_2_idx[_] for _ in timestamp_list])
        return ts_idx

    @lru_cache(maxsize=None)
    @profile
    def __call__(
            self, fault_ts: int, window_size: Tuple[int, int] = (10, 10), batch_normalization: bool = True
    ) -> List[th.Tensor]:
        if batch_normalization:
            def batch_rescale(feat):
                return feat - th.nanmean(feat[..., :-window_size[1]], dim=-1, keepdim=True)
        else:
            def batch_rescale(feat):
                return feat

        ts_idx = self.get_timestamp_indices(fault_ts, window_size)

        features_list = [batch_rescale(_[..., ts_idx]) for _ in self._features_list]
        return features_list

    @staticmethod
    @profile
    def extract_features(
            fdg: FDG,
            start_ts: int, length: int, granularity: int = 60,
            fill_na: bool = True, clip_value: Optional[float] = 10.
    ):
        with profile('ts select'):
            the_df = fdg.metrics_df[
                (fdg.metrics_df.timestamp >= start_ts) &
                (fdg.metrics_df.timestamp < start_ts + length * granularity)
                ]

        # look back much more data points to ensure there is beginning missing points
        timestamp_list = [start_ts + i * granularity for i in range(length)]
        timestamp_2_idx = {ts: idx for idx, ts in enumerate(timestamp_list)}
        features_list = []
        for failure_class in tqdm(fdg.failure_classes, desc='preprocess metrics for each instance type'):
            with profile('instance type iter'):
                with profile('get idx dict'):
                    metric_2_node_idx = {}
                    metric_2_metric_idx = {}
                    node_type_metrics = set()
                    for i, instance in enumerate(fdg.failure_instances[failure_class]):
                        for j, metric in enumerate(fdg.FI_metrics_dict[instance]):
                            metric_2_node_idx[metric] = i
                            metric_2_metric_idx[metric] = j
                            node_type_metrics.add(metric)
                with profile("get values from df"):
                    _feat = np.full(
                        (len(fdg.failure_instances[failure_class]), fdg.metric_number_dict[failure_class], length),
                        float('nan'))

                    with profile("index"):
                        _df = the_df.loc[
                            the_df.name.isin(node_type_metrics) & the_df.timestamp.isin(timestamp_list),
                            ['timestamp', 'value', 'name']
                        ]
                    with profile("fill into feat"):
                        if len(_df):
                            _df['idx0'] = _df['name'].map(lambda _: metric_2_node_idx[_])
                            _df['idx1'] = _df['name'].map(lambda _: metric_2_metric_idx[_])
                            _df['idx2'] = _df.timestamp.map(lambda _: timestamp_2_idx[_])
                            _feat[_df.idx0.values, _df.idx1.values, _df.idx2.values] = _df['value'].values
                        else:
                            pass
                if fill_na:
                    with profile("fill na"):
                        # 通过之前的最近一个点填充
                        _feat = forward_fill_na(_feat, axis=-1)

                        # 如果还没填充上的，通过全局的均值填充
                        for i, j in zip(*np.where(np.any(np.isnan(_feat), axis=-1))):
                            with profile("metric iter"):
                                metric_name = fdg.FI_metrics_dict[fdg.failure_instances[failure_class][i]][j]
                                np.nan_to_num(
                                    _feat[i, j, :],
                                    copy=False,
                                    nan=fdg.metric_mean_dict.get(metric_name, -10)
                                )
                    assert np.all(np.isfinite(_feat))

                if clip_value is not None:
                    # clip extreme values
                    _feat = np.clip(_feat, -clip_value, clip_value)
                    # _feat = tanh_estimator(_feat)

                features_list.append(th.from_numpy(_feat))
                assert features_list[-1].size() == (
                    len(fdg.failure_instances[failure_class]),
                    fdg.metric_number_dict[failure_class],
                    length
                ) == get_input_tensor_size_dict(fdg, (0, length))[failure_class], f"{features_list[-1].size()}"
        return features_list, timestamp_2_idx


@lru_cache
def get_input_tensor_size_dict(fdg: FDG, window_size: Tuple[int, int]) -> Dict[str, th.Size]:
    return {
        k: (
            len(fdg.failure_instances[k]), v, window_size[0] + window_size[1]
        ) for k, v in fdg.metric_number_dict.items()
    }


@lru_cache
def get_input_tensor_size_list(fdg: FDG, window_size: Tuple[int, int]) -> List[th.Size]:
    return [get_input_tensor_size_dict(fdg, window_size=window_size)[_] for _ in fdg.failure_classes]
