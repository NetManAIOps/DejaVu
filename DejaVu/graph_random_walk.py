from typing import List

import torch as th

from failure_dependency_graph import FDG


@th.jit.script
def get_node2vec_walk_sequence(
        adj: th.Tensor, length: int, p: float = 1 / 4, q: float = 1 / 4, debug: bool = False,
        undirected: bool = True
) -> th.Tensor:
    if undirected:
        adj = adj + adj.T
    n_nodes = len(adj)
    paths = th.zeros((n_nodes, length), dtype=th.long, device=adj.device)
    for start_node in th.arange(n_nodes, dtype=th.long, device=adj.device):
        last_node = -1
        curr = start_node
        pos = 0
        while pos < length:
            paths[start_node, pos] = curr
            pos += 1
            curr_neighbors = th.where(adj[curr, :] != 0)[0]
            if len(curr_neighbors) == 0:
                continue
            curr_transition_probability = th.zeros((len(curr_neighbors),), dtype=th.float32, device=adj.device)
            for nei_idx, curr_neighbor in enumerate(curr_neighbors):
                if last_node == -1:
                    curr_transition_probability[nei_idx] = 1.0
                elif curr_neighbor == last_node:  # distance==0, 1 / p
                    curr_transition_probability[nei_idx] = 1.0 / p
                elif adj[curr_neighbor, last_node]:  # distance == 1, 1
                    curr_transition_probability[nei_idx] = 1.0
                else:
                    curr_transition_probability[nei_idx] = 1.0 / q
            probability_cumulative = th.cumsum(curr_transition_probability, dim=0)
            last_node = curr
            random_choice_prob = th.rand(1).squeeze(0) * probability_cumulative[-1]
            if debug:
                __curr_neighbors_list: List[int] = curr_neighbors.tolist()
                __cul_prob_list: List[float] = probability_cumulative.tolist()
                print(
                    f"current node: {curr.item()} neighbors: {__curr_neighbors_list}, "
                    f"probs: {__cul_prob_list}, choice: {random_choice_prob.item()}"
                )
            curr = curr_neighbors[
                th.searchsorted(probability_cumulative, random_choice_prob)
            ]
    return paths


@th.jit.script
def get_random_walk_sequence(
        adj: th.Tensor, length: int, d: float = 0.85,
) -> th.Tensor:
    device = adj.device
    d = th.tensor(d, device=device)

    # normalize
    non_zero_rows, = th.where(th.any(th.greater(adj, 0), dim=-1))
    adj[non_zero_rows] = adj[non_zero_rows] / th.sum(adj[non_zero_rows], dim=1, keepdim=True)
    # add transition probability
    adj = d * adj + (1 - d) * th.full_like(adj, fill_value=1. / len(adj))

    adj = adj.to(device)

    n_nodes = len(adj)
    paths = th.zeros((n_nodes, length), dtype=th.long, device=device)
    for start_node in th.arange(n_nodes, dtype=th.long, device=device):
        curr = start_node.to(device)
        pos = 0
        while pos < length:
            paths[start_node, pos] = curr.to(device)
            pos += 1
            probability_cumulative = th.cumsum(adj[curr, :], dim=0)
            random_choice_prob = th.rand(1).squeeze(0) * probability_cumulative[-1]
            curr = th.arange(n_nodes, device=device)[
                th.searchsorted(probability_cumulative, random_choice_prob)
            ]
    return paths


# @th.jit.script
def monitor_rank_score(adj: th.Tensor, node_weight: th.Tensor, rho: float = 0.01, damping: float = 0.85):
    node_weight = th.softmax(node_weight, dim=0)
    A = monitor_rank_unnormed_adj(adj, node_weight, rho)

    A = A / th.sum(A, dim=-1, keepdim=True)

    M = damping * A + (1 - damping) * node_weight.view(1, -1)  # (n, n)
    pi = node_weight.view(1, -1)
    for i in range(100):
        pi = pi @ M
    # eig = th.eig(M.T, eigenvectors=True)
    # pi = eig.eigenvectors[:, th.argmax(th.sum(th.square(eig.eigenvalues), dim=1))]
    pi = pi.view(-1) / th.sum(pi)
    return pi


@th.jit.script
def monitor_rank_unnormed_adj(adj: th.Tensor, node_weight: th.Tensor, rho: float = 0.01, undirected: bool = True):
    n = len(adj)
    assert len(node_weight) == n
    device = adj.device

    if undirected:
        adj[:, :] = th.maximum(adj, adj.T)

    node_weight = th.softmax(node_weight, dim=0)

    A = adj * (node_weight.view(1, -1) / th.sum(node_weight))
    A = A + rho * A.T
    A[th.arange(n, device=device), th.arange(n, device=device)] = 0
    A[th.arange(n, device=device), th.arange(n, device=device)] = th.maximum(
        th.max(A, dim=0).values - th.max(A, dim=1).values,
        th.zeros((n,), device=device)
    )
    return A


@th.jit.script
def get_node_distance_matrix(adj: th.Tensor) -> th.Tensor:
    adj = adj + adj.T
    m = adj
    n_nodes: int = len(adj)
    dis: th.Tensor = th.full((n_nodes, n_nodes), fill_value=float('inf'), dtype=th.float32)
    dis[th.arange(n_nodes), th.arange(n_nodes)] = 0.
    i = 1.
    while i < n_nodes:
        x_indices, y_indices = th.where(th.logical_and(th.greater(dis, i), m >= 1))
        dis[x_indices, y_indices] = i
        i += 1.
        m = adj @ m
    return dis


def debug_plot_random_walk(cdp: FDG, walk_sequence: th.Tensor):
    for i in range(cdp.n_failure_instances):
        print("->".join(f"{cdp.gid_to_instance(_):<15}" for _ in walk_sequence[i]))
