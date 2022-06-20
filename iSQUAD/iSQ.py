import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

import torch as th
from loguru import logger
from pyprof import profile
from sklearn.neighbors import KDTree
from tqdm import tqdm

from failure_dependency_graph import FDG, split_failures_by_type
from iSQUAD.anomaly_detection import robust_threshold, t_test
from iSQUAD.config import ISQUADConfig
from metric_preprocess import MetricPreprocessor


class ISQUARD:
    def __init__(self, fdg: FDG, config: ISQUADConfig, mp: MetricPreprocessor):
        self.fdg = fdg
        self.faults_df = self.fdg.failures_df
        self.train_fault_ids, self.validation_fault_ids, self.test_fault_ids = split_failures_by_type(
            self.fdg.failures_df, split=config.dataset_split_ratio, train_set_sampling_ratio=config.train_set_sampling,
            fdg=fdg,
        )
        self._train_len = len(self.train_fault_ids)
        self.config = config
        self.mp = mp

        self._fault_ids = self.train_fault_ids + self.validation_fault_ids + self.test_fault_ids
        self._fault_id_2_idx: Dict[int, int] = {fault_id: idx for idx, fault_id in enumerate(self._fault_ids)}
        ##############################
        self._metrics = []
        self._metrics_2_type: Dict[str, int] = {}
        self._metric_patterns: th.Tensor = th.Tensor()  # each tensor in shape (n_metrics, n_faults, )
        self._idx_2_cluster: Dict[int, int] = {}  #
        self._cluster_2_root_cause_gid: Dict[int, int] = {}
        self._cluster_2_indices: Dict[int, List[int]] = {}  #
        self._kd_tree: Optional[KDTree] = None

        self._y_preds: List[List[int]] = []  # List of prediction list (list of gid)

    def __call__(self):
        with profile("training"):
            self._kpi_anomaly_detection_()
            if self.config.enable_dependency_cleansing:
                self._dependency_cleansing_()
            self._init_cluster_()
            # self._topic_()
            self._post_init_cluster()
            self._label_clusters()
        self._detection_()
        return self._y_preds

    @profile
    def _kpi_anomaly_detection_(self):
        window_size = (60, 10)
        for node_type in self.fdg.failure_classes:
            for node in self.fdg.failure_instances[node_type]:
                for metric in self.fdg.FI_metrics_dict[node]:
                    self._metrics.append(metric)
                    self._metrics_2_type[metric] = self.fdg.failure_classes.index(
                        self.fdg.instance_to_class(node)
                    )
        self._metric_patterns = th.zeros(
            (len(self._metrics), len(self._fault_ids)),
            dtype=th.int32,
        )
        logger.debug(f"Original metric pattern size: {self._metric_patterns.size()}")
        for fault_idx, fault_id in tqdm(
                enumerate(self._fault_ids), total=len(self._fault_ids), desc='kpi anomaly detection'
        ):
            fault_kpis = self.mp(fault_ts=self.faults_df.iloc[fault_id]['timestamp'], window_size=window_size)
            fault_kpis = th.cat([th.flatten(_, 0, 1) for _ in fault_kpis], dim=0)
            assert fault_kpis.size()[-1] == sum(window_size) and len(fault_kpis.size()) == 2
            rt_ret = robust_threshold(fault_kpis, window_size=window_size[0])[..., -window_size[1]:]
            # "For level-shift detection, the window is set to 30 minutes by default
            # and the t-value threshold is set empirically."
            t_ret = t_test(fault_kpis[:, -30 - window_size[1]:], window_size=window_size[1], significance_level=0.05)
            self._metric_patterns[th.sum(rt_ret, dim=-1) >= window_size[1] / 2, fault_idx] = 1
            self._metric_patterns[th.sum(rt_ret, dim=-1) <= - window_size[1] / 2, fault_idx] = -1
            self._metric_patterns[t_ret >= 1, fault_idx] = 2
            self._metric_patterns[t_ret <= -1, fault_idx] = -2

    @profile
    def _dependency_cleansing_(self):
        remove_indices = set()
        threshold = 0.95
        combinations = list(itertools.combinations(range(len(self._metrics)), 2))
        for i, j in tqdm(
                combinations, desc='dependency_cleansing'
        ):
            both = th.count_nonzero(th.logical_and(
                self._metric_patterns[i, :self._train_len] != 0, self._metric_patterns[j, :self._train_len] != 0
            ))
            given_i = th.count_nonzero(self._metric_patterns[i, :self._train_len] != 0)
            given_j = th.count_nonzero(self._metric_patterns[j, :self._train_len] != 0)
            if both == 0:
                continue
            elif both / given_i > threshold:
                remove_indices.add(j)
            elif both / given_j > threshold:
                remove_indices.add(i)
            else:
                continue
        indices = sorted(list(set(range(len(self._metrics))) - remove_indices))
        self._metrics = [_ for idx, _ in enumerate(self._metrics) if idx not in remove_indices]
        self._metric_patterns = self._metric_patterns[indices, :]
        logger.debug(f"Cleansed metric pattern size: {self._metric_patterns.size()}")

    @profile
    def _init_cluster_(self):
        self._cluster_2_indices = {self._fault_id_2_idx[_]: [self._fault_id_2_idx[_]] for _ in
                                   self._fault_ids[:self._train_len]}
        self._kd_tree = KDTree(self._metric_patterns[:, :self._train_len].T)

    @profile
    def _post_init_cluster(self):
        for cluster, indices in self._cluster_2_indices.items():
            for idx in indices:
                self._idx_2_cluster[idx] = cluster

    @profile
    def _topic_(self):
        changed = False
        while changed:
            changed = False
            for cluster, fault_indices in list(self._cluster_2_indices.items()):
                if len(fault_indices) > 1:
                    continue
                i = fault_indices[0]
                j = self._kd_tree.query(self._metric_patterns[:, i].unsqueeze(0), 2)[1][0]
                if j[0] != i:
                    j = j[0]
                else:
                    j = j[1]
                assert i != j
                if i not in self._cluster_2_indices or j not in self._cluster_2_indices:
                    continue
                if self.similarity(self._metric_patterns[:, i], self._metric_patterns[:, j]) > 0.67:
                    changed = True
                    if len(self._cluster_2_indices[i]) > len(self._cluster_2_indices[j]):
                        self._cluster_2_indices[i] += self._cluster_2_indices[j]
                        del self._cluster_2_indices[j]
                    else:
                        self._cluster_2_indices[j] += self._cluster_2_indices[i]
                        del self._cluster_2_indices[i]
        for cluster, indices in self._cluster_2_indices.items():
            for idx in indices:
                self._idx_2_cluster[idx] = cluster

    def similarity(self, pattern1: th.Tensor, pattern2: th.Tensor, concerned_metrics: Set[str] = None):
        if concerned_metrics is None:
            concerned_metrics = set(self._metrics)
        type_sum = defaultdict(float)
        for i, metric in enumerate(self._metrics):
            if metric in concerned_metrics:
                type_sum[self._metrics_2_type[metric]] += pattern1[i] == pattern2[i]
        v = th.tensor(list(type_sum.values()))
        return th.sqrt(th.sum(th.square(v)) / th.prod(th.tensor(v.size())))

    def _detection_(self):
        def fault_metrics(fid):
            _ret = set()
            for _node_type in {self.fdg.instance_to_class(_) for _ in self.fdg.root_cause_instances_of(fid)}:
                for _node in self.fdg.failure_instances[_node_type]:
                    _ret |= set(self.fdg.FI_metrics_dict[_node])
            return _ret

        assert self._kd_tree is not None
        for fault_id in tqdm(self.test_fault_ids, desc='detecting'):
            with profile("Inference for each failure"):
                isq = self._metric_patterns[:, self._fault_id_2_idx[fault_id]]
                similarities = th.tensor(list(map(
                    lambda _: self.similarity(
                        isq, self._metric_patterns[:, _],
                        concerned_metrics=fault_metrics(fault_id)
                    ),
                    range(self._train_len),
                )))
                nearest = th.arange(len(similarities))[th.argsort(similarities, descending=True)].tolist()
                # nearest = self._kd_tree.query(isq.unsqueeze(0), k=max(len(self.train_fault_ids), 16))[1][0]
                ret = []
                for i in nearest:
                    __rc = self._cluster_2_root_cause_gid[self._idx_2_cluster[i]]
                    if __rc not in ret:
                        ret.append(__rc)
                self._y_preds.append(ret)
        assert len(self._y_preds) == len(self.test_fault_ids)

    def _label_clusters(self):
        for cluster_idx, fault_indices in self._cluster_2_indices.items():
            fault_ids = [self._fault_ids[_] for _ in fault_indices]
            rc_counter = Counter(sum(
                [
                    list(map(
                        self.fdg.instance_to_gid,
                        self.faults_df.iloc[fault_id]['root_cause_node'].split(";")
                    )) for fault_id in fault_ids
                ],
                []
            ))
            self._cluster_2_root_cause_gid[cluster_idx] = rc_counter.most_common(1)[0][0]
