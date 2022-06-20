from collections import defaultdict
from itertools import combinations_with_replacement
from typing import List

import numpy as np
import torch as th
from loguru import logger
from pyprof import profile
from tqdm import tqdm

from failure_dependency_graph import FDG
from metric_preprocess import MetricPreprocessor


class SystemGraph:
    def __init__(self, cdp: FDG, fault_id: int, feature_extractor: MetricPreprocessor):
        self.cdp = cdp
        self.fault_id = fault_id
        self.granularity = 60
        self.window_size = (30, 5)
        fault_ts = cdp.failures_df.iloc[fault_id]['timestamp']
        self.features: List[th.Tensor] = feature_extractor(
            fault_ts=fault_ts,
            window_size=self.window_size
        )
        self.is_node_abnormal = defaultdict(lambda: False)
        for node_type, type_features in zip(self.cdp.failure_classes, self.features):
            max_node, max_score = None, 0
            for node, node_feature in zip(self.cdp.failure_instances[node_type], type_features):
                # if cdp.faults_df.iloc[fault_id]['root_cause_node'] == node:
                #     self.is_node_abnormal[node] = True
                # else:
                #     self.is_node_abnormal[node] = False
                his = node_feature[:, :self.window_size[0]]
                # median = th.median(his, dim=-1).values
                # mad: th.Tensor = th.median(th.abs(his - median.view(-1, 1)), dim=-1).values
                mean = th.mean(his, dim=-1)
                std = th.std(his, dim=-1) + th.tensor(1e-2)
                cur = node_feature[:, self.window_size[0]:]
                score = th.max(
                    th.mean(th.abs(cur - mean.view(-1, 1)) / th.maximum(mean.view(-1, 1) * 0.1, std.view(-1, 1)),
                            dim=-1)
                )
                if score > max_score:
                    max_score = score
                    max_node = node
            if max_score > 5:
                # print(max_node, max_score)
                self.is_node_abnormal[max_node] = True

    @staticmethod
    def similarity(a: 'SystemGraph', b: 'SystemGraph') -> float:
        cnt = 0.
        sim = 0.
        for i, node_type in enumerate(a.cdp.failure_instances):
            type_sim = 0.
            # 使用任意匹配会导致无法精确定位到同类型的节点
            for j1, j2 in combinations_with_replacement(range(len(a.cdp.failure_instances[node_type])), 2):
                # for j1 in range(len(a.cdp.nodes[node_type])):
                #     j2 = j1
                if not a.is_node_abnormal[a.cdp.failure_instances[node_type][j1]] \
                        or not b.is_node_abnormal[b.cdp.failure_instances[node_type][j2]]:
                    continue
                feat_a = a.features[i][j1]
                feat_b = b.features[i][j2]
                change_a = th.mean(feat_a[:, a.window_size[0]:]) - th.mean(feat_a[:, :a.window_size[0]])
                change_b = th.mean(feat_b[:, b.window_size[0]:]) - th.mean(feat_b[:, :b.window_size[0]])
                mean_a = th.mean(feat_a[:, a.window_size[0]:])
                mean_b = th.mean(feat_b[:, b.window_size[0]:])
                type_sim = max(
                    np.mean([
                        1 - th.abs(mean_a - mean_b), 1 - th.abs(change_a - change_b)
                    ]),
                    type_sim
                )
                # type_sim = np.nanmean(
                #     [pearsonr(feat_a[_, :].numpy(), feat_b[_, :].numpy())[0] for _ in range(len(feat_a))]
                # )
            if type_sim > 0:
                sim += type_sim
                cnt += 1
        return sim / (cnt + 1)


class GraphLibrary:
    def __init__(self, cdp: FDG, fault_ids: List[int], mp: MetricPreprocessor):
        self.fe = mp
        logger.info(f"building graph library by {fault_ids}")
        self.graphs = [
            SystemGraph(cdp, fault_id=_, feature_extractor=self.fe) for _ in tqdm(fault_ids)
        ]
        self.root_causes = [
            fault['root_cause_node'] for _, fault in cdp.failures_df.iloc[fault_ids].iterrows()
        ]
        assert len(self.graphs) == len(self.root_causes)

        self.cdp = cdp

    @profile
    def query(self, fault_id: int) -> List[str]:
        graph = SystemGraph(self.cdp, fault_id=fault_id, feature_extractor=self.fe)
        similarities = [SystemGraph.similarity(graph, _) for _ in self.graphs]
        indices = sorted(range(len(self.graphs)), key=lambda _: similarities[_], reverse=True)
        preds = [self.root_causes[_] for _ in indices]
        logger.info(
            f"|{fault_id:<5}|"
            f"{self.cdp.failures_df.iloc[fault_id]['root_cause_node']:<20}|"
            f"{preds[0]:<20} {similarities[indices[0]]:4f}|"
            f"{preds[1]:<20} {similarities[indices[1]]:4f}|"
            f"{preds[2]:<20} {similarities[indices[2]]:4f}|"
        )
        return preds
