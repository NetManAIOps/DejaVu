import dgl
import torch as th
from einops import rearrange

from DejaVu.dataset import DejaVuDataset
from DejaVu.models import DejaVuModelInterface


def find_similar_faults(dataset: DejaVuDataset, his_dataset: DejaVuDataset, model: DejaVuModelInterface, k: int = 5):
    # assert isinstance(model.module, GAT), f"{type(model.module)} is not supported"
    cdp = dataset.cdp
    assert dataset.cdp is his_dataset.cdp

    def collate_fn(batch_data):
        features_list, label, failure_id, graph = batch_data
        return (
            [v.type(th.float32).unsqueeze(0) for v in features_list],
            label.unsqueeze(0),
            [failure_id],
            [graph],
        )

    def get_feat(batch) -> th.Tensor:
        with th.no_grad():
            features, labels, failure_ids, graphs = collate_fn(batch)
            feat = model.module.feature_projector(features)  # (batch_size, N, feature_size)
            batch_size, n_instances, _ = feat.size()

            # select the features of the instances that are retained in the graphs
            feat = th.cat([feat[i, g.ndata[dgl.NID]] for i, g in enumerate(graphs)])

            batch_graph = dgl.batch(graphs)
            agg_feat = feat
            for GAT_aggregator in model.module.GAT_aggregators:
                agg_feat = rearrange(GAT_aggregator(batch_graph, agg_feat), "N H F -> N (H F)")
            return rearrange(agg_feat, "N F -> 1 N F")

    def get_score(_agg_feat: th.Tensor) -> th.Tensor:
        with th.no_grad():
            return th.sigmoid(model.module.predictor(_agg_feat))

    his_fault_ids = his_dataset.fault_ids.tolist()
    his_feats = th.cat([get_feat(his_dataset[_]) for _ in range(len(his_fault_ids))], dim=0)
    his_scores = get_score(his_feats)
    assert his_feats.size() == (
        len(his_fault_ids), cdp.n_failure_instances, model.module.feature_size * model.module.num_heads
    ), f"{his_feats.size()=} {get_feat(his_dataset[0]).size()}"
    assert his_scores.size() == (
        len(his_fault_ids), cdp.n_failure_instances,
    ), f"{his_scores.size()=}"

    fault_ids = dataset.fault_ids.tolist()
    feats = th.cat([get_feat(dataset[_]) for _ in range(len(fault_ids))], dim=0)
    scores = get_score(feats)
    assert feats.size() == (
        len(fault_ids), cdp.n_failure_instances, model.module.feature_size * model.module.num_heads
    ), f"{feats.size()=}"
    assert scores.size() == (
        len(fault_ids), cdp.n_failure_instances,
    ), f"{scores.size()=}"
    ret = {}
    for i, fault_id in enumerate(fault_ids):
        # node_type_dists = th.zeros(len(his_feats), len(cdp.node_types), dtype=th.float32)
        node_type_dists = th.zeros(len(his_feats), cdp.n_failure_instances, dtype=th.float32)
        for node_type_idx, node_type in enumerate(cdp.failure_classes):
            node_indices = th.tensor([cdp.instance_to_gid(_) for _ in cdp.failure_instances[node_type]])
            a = feats[None, i, node_indices, None, :]
            b = his_feats[:, None, node_indices, :]
            # a = scores[None, i, node_indices, None, None]
            # b = his_scores[:, None, node_indices, None]
            # node_type_dist = th.sqrt(th.mean(th.square(a - b), dim=-1))
            node_type_dist = th.mean(th.abs(a - b), dim=-1)
            min_node_type_dist = th.min(node_type_dist, dim=-1)
            # print(min_node_type_dist.indices, node_indices)
            # node_type_dists[:, node_type_idx] = th.mean(
            #     min_node_type_dist.values,
            #     dim=-1
            # )
            node_type_dists[:, node_indices] = min_node_type_dist.values

            # a = feats[None, i, node_indices[th.argsort(scores[i, node_indices], dim=-1, descending=True)]]
            # b = his_feats[
            #     th.arange(len(his_feats))[:, None],
            #     node_indices[th.argsort(his_scores[:, node_indices], dim=-1, descending=True)]
            # ]
            # node_type_dists[:, node_type_idx] = th.sum(
            #     th.sqrt(th.mean(th.square(a - b), dim=-1)) * scores[None, i, node_indices],
            #     dim=-1
            # )
        # dist = th.mean(node_type_dists, dim=-1)
        dist = th.sum(node_type_dists * scores[None, i, :], dim=-1)
        ret[fault_id] = [{"fault_id": his_fault_ids[idx], "distance": dist[idx]} for idx in th.argsort(dist)[:k]]
    return ret
