from math import prod

import dgl
import torch as th
from torch import nn as nn


class WrappedGraphNN(nn.Module):
    """
    Apply a graph neural network on a fixed graph
    """
    def __init__(self, module: nn.Module, graph: dgl.DGLGraph):
        super().__init__()
        self.module = module
        self.graph = graph
        self.dummy_param = nn.Parameter(th.empty(0))

    def forward(self, feat: th.Tensor) -> th.Tensor:
        """
        :param feat: a Tensor in shape `batch_shape + (n_nodes, n_feats)`
        :return: a tensor in shape `batch_shape + (n_nodes, Any)`
        """
        n_nodes = feat.size()[-2]
        n_feats = feat.size()[-1]
        if len(feat.size()) > 2:
            batch_shape = feat.size()[:-2]
            feat = feat.view(prod(batch_shape) * n_nodes, n_feats)
            graph = dgl.batch([self.graph.to(self.dummy_param.device)] * prod(batch_shape))
        else:
            batch_shape = tuple()
            graph = self.graph.to(self.dummy_param.device)
        out = self.module(graph, feat)
        out = out.view(batch_shape + (n_nodes, -1))
        return out
