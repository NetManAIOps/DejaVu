from typing import List

import dgl
import torch as th
from dgl.nn.pytorch import GATConv
from einops import rearrange
from pyprof import profile
from torch.nn import ModuleList, Identity

from DejaVu.models import DejaVuModuleProtocol
from failure_instance_feature_extractor import FIFeatureExtractor
from DejaVu.models.node_weight_predictor import NodeWeightPredictor
from utils import SequentialModelBuilder


class GAT(DejaVuModuleProtocol):
    @profile
    def __init__(
            self,
            node_types: List[str],
            input_features: List[th.Size], feature_size: int,
            feature_projector_type: str = 'CNN', GAT_layers: int = 1,
            num_heads: int = 1, residual: bool = False,
            has_dropout: bool = False, shared_feature_mapper: bool = False,
    ):
        super().__init__()
        self.feature_projector = FIFeatureExtractor(
            failure_classes=node_types,
            feature_size=feature_size,
            input_tensor_size_list=input_features,
            feature_projector_type=feature_projector_type,
        )

        if shared_feature_mapper:
            self.feature_mapper = SequentialModelBuilder(
                input_shape=(-1, feature_size),
            ).add_linear(feature_size).add_activation('ReLU').add_linear(feature_size).add_activation('ReLU').build()
        else:
            self.feature_mapper = Identity()

        self.feature_size = feature_size

        self.num_heads = num_heads
        self.GAT_aggregators = ModuleList([
            GATConv(
                in_feats=feature_size if i == 0 else feature_size * num_heads,
                out_feats=feature_size,
                num_heads=num_heads,
                residual=residual
            )
            for i in range(GAT_layers)
        ])

        self.predictor = NodeWeightPredictor(feature_size=feature_size * num_heads, has_dropout=has_dropout)

    def forward(self, x: List[th.Tensor], graphs: List[dgl.DGLGraph]):
        feat = self.feature_projector(x)  # (batch_size, N, feature_size)
        batch_size, n_instances, _ = feat.size()

        # select the features of the instances that are retained in the graphs
        feat = th.cat([feat[i, g.ndata[dgl.NID]] for i, g in enumerate(graphs)])

        feat = (self.feature_mapper(feat) + feat) / 2

        batch_graph = dgl.batch(graphs)
        batch_graph.ndata['graph_id'] = dgl.broadcast_nodes(
            batch_graph, th.arange(len(graphs), device=batch_graph.device)[:, None]
        )[:, 0]

        agg_feat = feat
        for GAT_aggregator in self.GAT_aggregators:
            agg_feat = rearrange(GAT_aggregator(batch_graph, agg_feat), "N H F -> N (H F)")

        node_weights = self.predictor(agg_feat)  # (n_total_instances, feature_size)
        ret = th.zeros((batch_size, n_instances), dtype=x[0].dtype, device=x[0].device)
        ret[batch_graph.ndata['graph_id'], batch_graph.ndata[dgl.NID]] = node_weights
        return ret
