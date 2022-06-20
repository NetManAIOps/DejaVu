from typing import List, Dict, Tuple

import cdt.causality.graph as ci
import networkx as nx
import numpy as np
import pandas as pd
from diskcache import Cache
from loguru import logger
from networkx import DiGraph
from numpy.lib.stride_tricks import sliding_window_view
from pyprof import profile

from random_walk_single_metric.config import RandomWalkSingleMetricConfig


@profile
def get_causal_graph(metrics_df: pd.DataFrame, method: str) -> nx.DiGraph:
    if method == "GES":
        return ci.GES(verbose=True).create_graph_from_data(metrics_df)
    elif method == "LiNGAM":
        return ci.LiNGAM(verbose=True).create_graph_from_data(metrics_df)
    elif method == "fGES":

        from pycausal import search as s
        tetrad = s.tetradrunner()
        tetrad.run(algoId='fges', dfs=metrics_df, scoreId='sem-bic', dataType='continuous',
                   maxDegree=-1, faithfulnessAssumed=True, verbose=False)
        import re
        edge_regex = re.compile(r"(?P<u>.*\S)\s+(?P<type>--[->])\s+(?P<v>\S.*)")
        g = nx.DiGraph()
        for node in metrics_df.columns:
            g.add_node(node)
        for edge in tetrad.getEdges():
            match = edge_regex.match(edge)
            if not match:
                logger.warning(f"Could not parse edge: {edge}")
            u = match.group("u")
            v = match.group("v")
            if match.group("type") == "-->":
                g.add_edge(u, v)
            elif match.group("type") == "---":
                g.add_edge(u, v)
                g.add_edge(v, u)
            else:
                logger.warning(f"Unknown edge type: {match.group('type')} {edge}")
        return g
    else:
        raise ValueError("Unknown method: {}".format(method))


@profile
def get_sliding_window_correlation_coefficient_matrix(metrics: np.ndarray, window_size: int = 10):
    assert np.ndim(metrics) == 2
    coef_list = []
    sliding_windows = sliding_window_view(
        metrics, window_shape=window_size, axis=0
    )  # (n_windows, variable, window_size)
    for window in sliding_windows:
        coef_list.append(np.corrcoef(window, rowvar=True))
    coef = np.nanmean(coef_list, axis=0)
    return np.nan_to_num(coef, nan=0.0)


def get_anomaly_score(metrics_df: pd.DataFrame, window_size: Tuple[int, int] = (10, 10)) -> Dict[str, float]:
    # moving median
    w = window_size[1]
    v = metrics_df.values  # (n_timestamps, n_variables)
    sliding_windows = sliding_window_view(v, window_shape=w, axis=0)  # (n_windows, n_variables, window_size)
    median = np.full_like(v, np.nan)
    mad = np.full_like(v, np.nan)
    for i, window in enumerate(sliding_windows):
        median[i + w - 1, :] = np.median(window, axis=1)
        mad[i + w - 1, :] = np.maximum(np.median(np.abs(window - median[i + w - 1, :, None])), 0.5)
    anomaly_score = np.zeros_like(v)
    anomaly_score[1:] = np.abs(v[1:, :] - median[:-1, :]) / np.maximum(mad[:-1, :], 1e-6)
    metric_scores = np.mean(anomaly_score[-w:], axis=0)
    return {
        m: s for m, s in zip(metrics_df.columns, metric_scores)
    }


@profile
def random_walk(metrics: np.ndarray, metric_names: List[str], config: RandomWalkSingleMetricConfig, cache: Cache) -> Dict[str, float]:
    ##################################################################################
    # Preprocess
    ##################################################################################
    metrics = np.asarray(metrics)

    _, __idx = np.unique(metric_names, return_index=True)
    metric_names = np.array(metric_names)[__idx]
    metrics = metrics[__idx, :]

    # ignore unchanged metrics
    metric_mean = np.mean(metrics, axis=-1, keepdims=True)
    rdiff = np.mean(np.abs(metrics - metric_mean) / np.maximum(np.abs(metric_mean), 1e-6), axis=-1)
    unchanged_metric_scores = {
        metric_name: 0.0 for metric_name in metric_names[np.where(rdiff < config.unchanged_metric_threshold)]
    }
    metric_names = np.array(metric_names)[np.where(rdiff >= config.unchanged_metric_threshold)]
    metrics = metrics[np.where(rdiff >= config.unchanged_metric_threshold)]

    metrics_df = pd.DataFrame(metrics.T, columns=metric_names)
    ###################################################################################
    logger.info("Starting random walk")

    logger.info("Creating causal graph")
    if "causal_graph" not in cache or config.flush_causal_graph_cache:
        cache.set("causal_graph", get_causal_graph(metrics_df, method=config.CI_method))
    causal_graph: nx.DiGraph = cache.get("causal_graph")

    adj = nx.adjacency_matrix(causal_graph, metric_names, weight=None)

    # get transition probability
    logger.info("Calculating correlation coefficient matrix")
    A = get_sliding_window_correlation_coefficient_matrix(metrics_df.values, window_size=config.corr_window_size) * adj
    A = np.abs(A)
    # add backward edge
    A = A + config.backward_weight * A
    # add self loop
    A[np.diag_indices_from(A)] = np.maximum(np.max(A, axis=0) - np.max(A, axis=1), 0)

    A = A / np.maximum(np.sum(A, axis=1, keepdims=True), 1e-6)

    random_walk_graph = nx.relabel_nodes(nx.from_numpy_matrix(A, create_using=nx.DiGraph), {
        i: m for i, m in enumerate(metric_names)
    })
    anomaly_scores = get_anomaly_score(metrics_df, window_size=config.window_size)
    scores = nx.pagerank(random_walk_graph, weight="weight", personalization=anomaly_scores)
    logger.info("Finished random walk")

    scores.update(unchanged_metric_scores)
    return scores
