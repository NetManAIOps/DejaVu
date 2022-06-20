import random
import re
from functools import lru_cache
from itertools import groupby
from typing import Optional, Sequence, Tuple, List, Dict, Union

import dgl
import numpy as np
import torch as th
from diskcache import Cache
from pyprof import profile
from pytorch_lightning.utilities import move_data_to_device
from torch.utils.data import DataLoader

from DejaVu.config import DejaVuConfig

__all__ = [
    "DejaVuDataset",
    "prepare_sklearn_dataset", "SKLEARN_DATASET_TYPE",
]

from failure_dependency_graph.incomplete_FDG_factory import IncompleteFDGFactory

from utils.metric_augmentation import add_missing_to_tensor, add_spike_to_tensor
from failure_dependency_graph import FDG
from failure_dependency_graph.model_interface import split_failures_by_type, FDGModelInterface
from metric_preprocess import MetricPreprocessor


class DejaVuDataset(th.utils.data.Dataset):
    @profile
    def __init__(
            self, cdp: FDG, feature_extractor: MetricPreprocessor, *,
            fault_ids: Optional[Sequence[int]] = None,
            window_size: Tuple[int, int] = (0, 5),
            granularity: int = 60,
            augmentation: bool = False,
            normal_data_weight: float = 0.,
            drop_edges_fraction: float = 0.,
            device: Optional[Union[th.device, str]] = None,
    ):
        self.cdp = cdp
        self.feature_extractor = feature_extractor
        self.extra_window_size = 5
        self._window_size = (window_size[0] + self.extra_window_size, window_size[1] + self.extra_window_size)
        self._granularity = granularity
        if fault_ids is not None:
            self.fault_ids = np.array(fault_ids)
        else:
            self.fault_ids = np.array(self.cdp.failure_ids)

        self.augmentation = augmentation

        self.normal_data_weight = normal_data_weight

        self.incomplete_FDG_factory = IncompleteFDGFactory(fdg=self.cdp, drop_edges_fraction=drop_edges_fraction)

        self.__cached_median_and_mad = {}

        if not self.augmentation:
            self.__getitem__ = lru_cache()(self.__getitem__)

        self._device = device

    def to(self, device: Union[th.device, str]) -> 'DejaVuDataset':
        self._device = device
        if hasattr(self.__getitem__, 'cache_clear'):
            self.__getitem__.cache_clear()
        return self

    @lru_cache
    def __len__(self):
        actual_len = len(self.fault_ids)
        normal_len = int(actual_len * self.normal_data_weight)
        return actual_len + normal_len

    @profile
    def __getitem__(self, index) -> Tuple[List[th.Tensor], th.Tensor, int, dgl.DGLGraph]:
        fault_idx = index % len(self.fault_ids)
        fault_id = self.fault_ids[fault_idx]
        fault = self.cdp.failure_at(fault_id)
        group_id = index // len(self.fault_ids)
        if group_id > 0:
            bias = random.choice(
                self.cdp.normal_timestamps(duration=self._window_size[1] - self.extra_window_size)
            ) - fault['timestamp']
        else:
            bias = 0
        ori_features = self.feature_extractor(
            fault_ts=fault['timestamp'] + bias,
            window_size=self._window_size,
        )
        labels = th.zeros((self.cdp.n_failure_instances,), dtype=th.float32)
        if bias == 0:
            for rc_node in self.cdp.root_cause_instances_of(fault_id):
                target_id = th.tensor(self.cdp.instance_to_gid(rc_node))
                labels[target_id] = 1
        else:
            pass  # 正常数据

        if self.augmentation:
            lag = 0
            features = _add_lag(self.extra_window_size, sum(self._window_size), ori_features, lag=lag)
            # 获取缓存的median和mad
            if bias == 0:
                _mm_cache_key = (fault_idx, lag)
                if _mm_cache_key not in self.__cached_median_and_mad:
                    __median = [th.median(_.view(-1, _.size()[-1]), dim=1, keepdim=True).values for _ in features]
                    __mad = [
                        th.median(th.abs(_.view(-1, _.size()[-1]) - __m), dim=1, keepdim=True).values
                        for _, __m in zip(features, __median)
                    ]
                    self.__cached_median_and_mad[_mm_cache_key] = (__median, __mad)
                median_and_mad = self.__cached_median_and_mad[_mm_cache_key]
            else:
                median_and_mad = None
            features = _add_missing_or_spike(features, size=random.randint(1, 5), median_and_mad=median_and_mad)
        else:
            features = _add_lag(self.extra_window_size, sum(self._window_size), ori_features, lag=0)

        g: dgl.DGLGraph = self.incomplete_FDG_factory(failure_id=fault_id)
        result = features, labels, fault_id, g
        if self._device is not None:
            result = move_data_to_device(result, self._device)
        return result


@th.jit.script
def _add_lag(extra_window_size: int, window_length: int, features: List[th.Tensor], lag: int = 0) -> List[th.Tensor]:
    assert abs(lag) <= extra_window_size
    left = extra_window_size + lag
    right = window_length - extra_window_size + lag
    return [
        _[..., left:right]
        for _ in features
    ]


@th.jit.script
def _add_missing_or_spike(
        features: List[th.Tensor], size: int = 1,
        median_and_mad: Optional[Tuple[List[th.Tensor], List[th.Tensor]]] = None,
) -> List[th.Tensor]:
    feature_length = features[0].size()[-1]
    assert 1 <= size < feature_length
    if size < 1:
        return features
    choice = th.randn(1)
    if choice > 0.5:
        return [
            add_missing_to_tensor(_, size) for _ in features
        ]
    else:
        if median_and_mad is None:
            return [
                add_spike_to_tensor(_, size) for _ in features
            ]
        else:
            return [
                add_spike_to_tensor(feat, size, median, mad)
                for feat, median, mad in zip(features, median_and_mad[0], median_and_mad[1])
            ]


SKLEARN_DATASET_TYPE = Tuple[
    Dict[str, Tuple[List[str], Tuple[Tuple[np.ndarray, np.ndarray, List[int], List[str]], ...]]],
    Tuple[List[int], ...]
]


def prepare_sklearn_dataset(
        cdp: FDG, config: DejaVuConfig, cache: Cache, mode: str
) -> SKLEARN_DATASET_TYPE:
    from DejaVu.explanability import prepare_ts_feature_dataset
    feature_extractor = FDGModelInterface.get_metric_preprocessor(fdg=cdp, config=config, cache=cache)
    ts_feature_key = f'TS-Feature-{mode=}'
    if ts_feature_key not in cache or config.flush_dataset_cache:
        cache.set(ts_feature_key, prepare_ts_feature_dataset(
            fdg=cdp, fe=feature_extractor, window_size=config.window_size, mode=mode,
        ))
    ts_features_dict = cache.get(ts_feature_key)
    del feature_extractor
    train_fault_ids, validation_fault_ids, test_fault_ids = split_failures_by_type(
        cdp.failures_df, fdg=cdp, split=config.dataset_split_ratio, train_set_sampling_ratio=config.train_set_sampling,
    )
    train_fault_ids = set(train_fault_ids)
    validation_fault_ids = set(validation_fault_ids)
    # test_fault_ids = set(test_fault_ids)
    rst = {}
    for node_type in cdp.failure_classes:
        assert node_type in ts_features_dict, f"{ts_features_dict.keys()=} {cdp.failure_classes=}"
        ts_feats = ts_features_dict[node_type].replace([np.nan, np.inf, -np.inf], 0)
        _data = []
        for i, (index, row) in enumerate(ts_feats.iterrows()):
            match = re.match(r'fault_id=(?P<id>\d+)\.node=\'(?P<node>.*)\'', index)
            fault_id = int(match.group('id'))
            node = match.group('node')
            rc_node = cdp.failure_at(fault_id)['root_cause_node']
            _data.append((
                0 if fault_id in train_fault_ids else (1 if fault_id in validation_fault_ids else 2),
                row.values,
                int(rc_node == node),
                fault_id,
                node,
            ))
        key_func = lambda _: _[0]
        _data = sorted(_data, key=key_func)
        _x_list, _y_list, _fault_ids, _node_names = [], [], [], []
        for _, _parted_data in groupby(_data, key=key_func):
            _parted_data = list(_parted_data)
            _x_list.append(np.vstack([_[1] for _ in _parted_data]))
            _y_list.append(np.array([_[2] for _ in _parted_data]))
            _fault_ids.append(list([_[3] for _ in _parted_data]))
            _node_names.append(list([_[4] for _ in _parted_data]))
        rst[node_type] = (
            list(ts_feats.columns),
            (
                (_x_list[0], _y_list[0], _fault_ids[0], _node_names[0]),
                (_x_list[1], _y_list[1], _fault_ids[1], _node_names[1]),
                (_x_list[2], _y_list[2], _fault_ids[2], _node_names[2]),
            )
        )

    return rst, (list(train_fault_ids), list(validation_fault_ids), list(test_fault_ids))
