from typing import Dict, Tuple, Callable

import networkx as nx
import numpy as np
from diskcache import Cache
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view
from pyprof import profile

from random_walk_failure_instance.config import RandomWalkFailureInstanceConfig


def get_agg(aggregation_method: str) -> Callable[[np.ndarray], float]:
    if aggregation_method == "sum":
        agg = np.sum
    elif aggregation_method == "max":
        agg = np.max
    elif aggregation_method == "min":
        agg = np.min
    elif aggregation_method == "mean":
        agg = np.mean
    else:
        raise ValueError("Unknown aggregation method: {}".format(aggregation_method))
    return agg


def set_weight_by_FI_correlation_(graph: nx.DiGraph, corr_aggregation_method: str):
    agg = get_agg(corr_aggregation_method)
    del corr_aggregation_method
    for u, v in graph.edges():
        u_metrics = graph.nodes[u]["values"]
        v_metrics = graph.nodes[v]["values"]
        corr = np.corrcoef(np.concatenate([u_metrics, v_metrics], axis=0), rowvar=True)[:len(u_metrics), -len(v_metrics):]
        np.nan_to_num(corr, copy=False)
        graph[u][v]["weight"] = agg(np.abs(corr))
    return graph


def set_anomaly_score_(graph, window_size: Tuple[int, int] = (10, 10), aggregation_method: str = "sum"):
    agg = get_agg(aggregation_method)

    def get_anomaly_score(v: np.ndarray) -> float:
        # moving median
        w = window_size[1]
        sliding_windows = sliding_window_view(v, window_shape=w, axis=0)  # (n_windows, n_variables, window_size)
        median = np.full_like(v, np.nan)
        mad = np.full_like(v, np.nan)
        for i, window in enumerate(sliding_windows):
            median[i + w - 1, :] = np.median(window, axis=1)
            mad[i + w - 1, :] = np.median(np.abs(window - median[i + w - 1, :, None]))
        anomaly_score = np.zeros_like(v)
        anomaly_score[1:] = np.abs(v[1:, :] - median[:-1, :]) / np.maximum(mad[:-1, :], 1e-6)
        metric_scores = np.mean(anomaly_score[-w:], axis=0)
        return agg(metric_scores)

    for node, data in graph.nodes(data=True):
        data["anomaly_score"] = get_anomaly_score(data["values"].T)


@profile
def random_walk(graph: nx.DiGraph, config: RandomWalkFailureInstanceConfig, cache: Cache) -> Dict[str, float]:
    logger.info("Starting random walk")
    FI_list = list(graph.nodes())

    # get transition probability
    logger.info("Calculating correlation coefficient matrix")
    set_weight_by_FI_correlation_(graph, corr_aggregation_method=config.corr_aggregation_method)
    A = np.asarray(nx.adjacency_matrix(graph, nodelist=FI_list).todense())
    A = np.abs(A)
    # add backward edge
    A = A + config.backward_weight * A
    # add self loop
    A[np.diag_indices_from(A)] = np.maximum(np.max(A, axis=0) - np.max(A, axis=1), 0)

    A = A / np.maximum(np.sum(A, axis=1, keepdims=True), 1e-6)

    random_walk_graph = nx.relabel_nodes(nx.from_numpy_matrix(A, create_using=nx.DiGraph), {
        i: m for i, m in enumerate(FI_list)
    })
    logger.info("Calculating anomaly score")
    set_anomaly_score_(
        graph, window_size=config.window_size, aggregation_method=config.anomaly_score_aggregation_method
    )
    scores = nx.pagerank(
        random_walk_graph, weight="weight", personalization=nx.get_node_attributes(graph, 'anomaly_score')
    )
    logger.info("Finished random walk")

    return scores
