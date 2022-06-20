# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# %load_ext autoreload
# %autoreload 2

import sys
import os
sys.path.insert(0, "/SSF")
from DejaVu.explib.read_model import read_model

# + tags=[]
from pathlib import Path
exp_dir = Path('/data/SSF/experiment_outputs/run_GAT_node_classification.py.2021-12-22T03:50:07.949451')

from DejaVu.models import get_GAT_model
model, y_probs, y_preds = read_model(
    exp_dir,
    get_GAT_model,
    override_config=dict(data_dir=Path("/SSF/data/A1"), flush_dataset_cache=False)
)

cdp = fdg = model.fdg
config = model.config
cache = model.cache

output_dir = config.output_dir / 'gnnexplainer_explain'
output_dir.mkdir(exist_ok=True, parents=True)
print(output_dir)

# +
from DejaVu.models import DejaVuModelInterface, GAT
from typing import Tuple, List
import dgl
import torch as th
from DejaVu.dataset import DejaVuDataset
from einops import rearrange
import copy


class WrappedGATDejaVuModel(th.nn.Module):
    def __init__(self, GAT_aggregators, predictor):
        super().__init__()
        self.GAT_aggregators = copy.deepcopy(GAT_aggregators)
        self.predictor = copy.deepcopy(predictor)
        
    def forward(self, graph: dgl.DGLGraph, feat: th.Tensor, eweight=None):
        with graph.local_scope():
            for GAT_aggregator in self.GAT_aggregators:
                feat = rearrange(GAT_aggregator(graph, feat), "N H F -> N (H F)")

            node_weights = self.predictor(feat)
            return rearrange(node_weights, "N -> N 1")
        


def wrap_gnnexplainer_model_from_DejaVu_model(model: DejaVuModelInterface) -> Tuple[WrappedGATDejaVuModel, List[dgl.DGLGraph], List[th.Tensor]]:
    fdg = model.fdg
    module: GAT = model.module
    assert isinstance(module, GAT)
    feature_projector = module.feature_projector
    # shared_feature_mapper = module.shared_feature_mapper
    GAT_aggregators = module.GAT_aggregators
    predictor = module.predictor
    wrapped_module = WrappedGATDejaVuModel(GAT_aggregators, predictor)
    
    dataset = DejaVuDataset(
        cdp=model.fdg,
        feature_extractor=model.metric_preprocessor,
        fault_ids=fdg.failure_ids,
        window_size=model.config.window_size,
        augmentation=model.config.augmentation,
        normal_data_weight=1. if model.config.augmentation else 0.,
        drop_edges_fraction=model.config.drop_FDG_edges_fraction,
        device=model.device,
    )
    
    collate_fn = model.get_collate_fn(1)
    graph_list = []
    feat_list = []
    for fid in fdg.failure_ids:
        _features_list, _, _, _graph_list = collate_fn([dataset[fid]])
        
        _feat = feature_projector(_features_list)  # (batch_size, N, feature_size)

        # select the features of the instances that are retained in the graphs
        _feat = th.cat([_feat[i, g.ndata[dgl.NID]] for i, g in enumerate(_graph_list)])

        # _feat = (feature_mapper(_feat) + _feat) / 2
        
        graph_list.append(_graph_list[0])
        feat_list.append(_feat.detach())
    
    return wrapped_module, graph_list, feat_list
    
wrapped_module, graph_list, feat_list = wrap_gnnexplainer_model_from_DejaVu_model(model)
# -



# + tags=[]
from graphviz import Digraph
import networkx as nx
failure_id=0

def plot_explain_graph(failure_id: int, num_hops=1) -> Digraph:
    from dgl.nn import GNNExplainer
    explainer = GNNExplainer(wrapped_module, num_hops=num_hops)

    rc = model.fdg.root_cause_instances_of(failure_id)[0]
    rc_gid = model.fdg.instance_to_gid(rc)
    print(f"{failure_id=} {model.fdg.root_cause_instances_of(failure_id)=}")
    new_center, sg, feat_mask, edge_mask = explainer.explain_node(
        rc_gid, 
        graph_list[failure_id], 
        feat_list[failure_id]
    )

    nid_dict = {i: nid.item() for i, nid in enumerate(sg.ndata[dgl.NID])}
    # print(nid_dict)
    for n in sg.nodes().tolist():
        n = model.fdg.gid_to_instance(nid_dict[n])

    nx_graph = nx.DiGraph()
    
    for u, v, weight in zip(*[_.tolist() for _ in sg.edges()], edge_mask):
        u = model.fdg.gid_to_instance(nid_dict[u])
        v = model.fdg.gid_to_instance(nid_dict[v])
        # print(u, v)
        if weight > 0.5 or u == rc or v == rc:
            color = f"gray{100 - int(weight * 100)}"
            nx_graph.add_edge(u, v, penwidth=f"{1 + weight}", color=color)
    nx_graph = nx_graph.subgraph([_ for _ in nx.weakly_connected_components(nx_graph) if rc in _][0])
            
    gv_graph = Digraph()
    for n in nx_graph.nodes():
        gv_graph.node(n, color="red" if n == rc else "cornflowerblue", style="filled", fontcolor="white")
    for u, v, data in nx_graph.edges(data=True):
        gv_graph.edge(u, v, **data)
    gv_graph.render(str(output_dir / f"{num_hops}" / f"{failure_id}_{rc}".replace(" ", "_")), format="pdf")
    return gv_graph

# plot_explain_graph(0, num_hops=5)
for num_hops in [1, 2, 3, 4, 5]:
    for fid in model.fdg.failure_ids:
        plot_explain_graph(fid, num_hops=num_hops)
# -

output_dir

sg.edges()

sg.ndata[dgl.NID]

zip(*[_.tolist() for _ in sg.edges()])


