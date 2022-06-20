from typing import List

import dgl
import torch as th
import torch.nn as nn
from pyprof import profile

from failure_instance_feature_extractor import FIFeatureExtractor
from DejaVu.models.interface import DejaVuModuleProtocol
from DejaVu.models.node_weight_predictor import NodeWeightPredictor


class MLP(DejaVuModuleProtocol):
    @profile
    def __init__(
            self,
            node_types: List[str],
            input_features: List[th.Size], feature_size: int,
            feature_projector_type: str = 'CNN',
            has_dropout: bool = False,
    ):
        super(MLP, self).__init__()
        self.node_types = node_types
        self.feature_projector = FIFeatureExtractor(
            failure_classes=self.node_types,
            feature_size=feature_size,
            input_tensor_size_list=input_features,
            feature_projector_type=feature_projector_type,
        )
        self.feature_size = feature_size

        self.predictor = NodeWeightPredictor(feature_size=feature_size, has_dropout=has_dropout)

    def forward(self, x: List[th.Tensor], graphs: List[dgl.DGLGraph]):
        feat = self.feature_projector(x)  # (batch_size, N, feature_size)

        return self.predictor(feat)  # (batch_size, N)


class MLPGraphClf(nn.Module):
    @profile
    def __init__(
            self,
            node_types: List[str],
            input_features: List[th.Size], feature_size: int,
            n_nodes: int,
            graph_adj,
            feature_projector_type: str = 'CNN'
    ):
        super(MLPGraphClf, self).__init__()
        self.n_nodes = n_nodes
        self.node_types = node_types
        self.feature_projector = FIFeatureExtractor(
            failure_classes=self.node_types,
            feature_size=feature_size,
            input_tensor_size_list=input_features,
            feature_projector_type=feature_projector_type
        )
        self.node_score_predictor = NodeWeightPredictor(feature_size=feature_size)

        attn_w = th.full((n_nodes, n_nodes), 0.)
        attn_w[th.arange(n_nodes), th.arange(n_nodes)] = th.ones(n_nodes)
        self.register_parameter(
            "attn_w",
            th.nn.Parameter(attn_w, requires_grad=True)
        )

        self.register_buffer(
            'graph_adj', graph_adj
        )

    def forward(self, x: List[th.Tensor]):
        feat = self.feature_projector(x)  # (batch_size, N, feature_size)
        node_weight = self.node_score_predictor(feat)
        # batch_size = feat.size()[0]

        # ret = th.zeros_like(node_weight)
        # for i in range(batch_size):
        #     A = monitor_rank_score(adj=self.graph_adj, node_weight=node_weight[i])
        #     ret[i, :] = A[th.arange(self.n_nodes, device=A.device), th.arange(self.n_nodes, device=A.device)]
        # return ret

        # return node_weight

        # mask = th.greater_equal(th.softmax(node_weight, dim=-1), 0.5 / self.n_nodes).detach()
        # return node_weight @ self.attn_w
        return node_weight @ self.attn_w

    def regularization(self):
        norm = th.norm(self.attn_w, p=1)
        # positive = th.sum(th.square(th.minimum(1 - th.abs(self.attn_w), th.zeros_like(self.attn_w))))
        sum_one = th.sum(th.square(
            th.sum(th.abs(self.attn_w), dim=0) - th.ones((self.n_nodes,), device=self.attn_w.device)
        ))
        # sparse = th.mean(th.sum(th.log(th.abs(self.attn_w) + 1e-8), dim=0))
        # sparse = th.count_nonzero(th.abs(self.attn_w) > 1e-4)
        return norm + sum_one * 1e2
