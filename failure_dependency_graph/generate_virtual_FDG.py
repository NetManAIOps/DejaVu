import random
from typing import Set, FrozenSet, Dict

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from pyprof import profile


def generate_random_connected_graph(n: int, density: float = 0.1) -> Set[FrozenSet[int]]:
    edges: Set[FrozenSet[int]] = set()
    # first generate a spanning tree
    with profile("spanning tree"):
        for i in range(1, n):
            edges.add(frozenset({i, random.randint(0, i - 1)}))
    # then add other edges
    # candidate_edges = {frozenset({i, j}) for i, j in combinations(range(n), 2)} - edges
    with profile("add edges"):
        required_n_edges: int = int(n * (n - 1) // 2 * density)
        while len(edges) < required_n_edges:
            with profile("add edge"):
                with profile("random sample"):
                    u, v = np.random.randint(0, n, 2)
                if u != v:
                    edges.add(frozenset({u, v}))
    return edges


@profile
def generate_virtual_FDG(
        *,
        n_classes: int,
        n_instances: int,
        n_metrics: int,
        n_failures: int,
        FDG_density: float = 0.1,
):
    n_classes = int(n_classes)
    n_instances = int(n_instances)
    n_metrics = int(n_metrics)
    n_failures = int(n_failures)
    FDG_density = float(FDG_density)

    from failure_dependency_graph.failure_dependency_graph import FDG
    assert n_instances >= n_classes >= 1
    assert n_metrics >= n_instances
    if n_metrics % n_instances != 0:
        logger.warning(f"{n_metrics=} % {n_instances=} != 0")
    average_n_metrics: int = n_metrics // n_instances
    assert n_failures >= 1

    gids = np.arange(n_instances)
    failure_class_right_boundaries = np.concatenate([
        np.sort(np.random.choice(np.arange(1, n_instances), size=n_classes - 1, replace=False)),
        [n_instances]
    ])
    failure_class_left_boundaries = np.concatenate([[0], failure_class_right_boundaries[:-1]])

    graph = nx.DiGraph()
    edges: Set[FrozenSet[int]] = generate_random_connected_graph(n=n_instances, density=FDG_density)
    gid_to_instance: Dict[int, str] = {}
    for class_idx, (left, right) in enumerate(zip(failure_class_left_boundaries, failure_class_right_boundaries)):
        class_name: str = "".join([chr(int(_) + ord('A')) for _ in f"{class_idx:d}"])
        for local_id, gid in enumerate(gids[left:right]):
            instance_name = f"{class_name}{local_id:d}"
            gid_to_instance[gid] = instance_name
            graph.add_node(
                instance_name,
                **{
                    "type": class_name,
                    "metrics": [
                        f"{instance_name}##{_}" for _ in range(average_n_metrics)
                    ],
                }
            )
    for edge in edges:
        u, v = tuple(edge)
        graph.add_edge(gid_to_instance[u], gid_to_instance[v], **{"type": "Deployment"})

    failures_df = pd.DataFrame.from_dict({
        "timestamp": np.arange(7200, 7200 + 60 * 60 * n_failures, 60 * 60),
        "root_cause_node": [gid_to_instance[_] for _ in np.random.choice(gids, size=n_failures, replace=True)],
    })
    failures_df['node_type'] = failures_df['root_cause_node'].apply(lambda x: x[0])

    metric_df_parts = []
    metric_timestamps = np.arange(0, failures_df['timestamp'].max(), 60)
    for gid in gids:
        for metric in [f"{gid_to_instance[gid]}##{_}" for _ in range(average_n_metrics)]:
            metric_df_parts.append(pd.DataFrame.from_dict({
                "timestamp": metric_timestamps,
                "value": np.zeros(len(metric_timestamps)),
                "name": metric,
            }))
    metrics_df = pd.concat(metric_df_parts)

    return FDG(graph=graph, metrics=metrics_df, failures=failures_df)
