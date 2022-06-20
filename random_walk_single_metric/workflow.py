from pathlib import Path
from typing import Dict, List

from diskcache import Cache
from loguru import logger
from pyprof import profile
from tqdm import tqdm

from DejaVu.config import DejaVuConfig
from DejaVu.evaluation_metrics import get_evaluation_metrics_dict
from DejaVu.workflow import format_result_string
from failure_dependency_graph import FDGModelInterface, FDG
from random_walk_single_metric.config import RandomWalkSingleMetricConfig
from random_walk_single_metric.model import random_walk


def agg_scores_by_failure_instance(fdg: FDG, metric_scores: Dict[str, float], agg_method: str = "sum") -> Dict[str, float]:
    scores_dict: Dict[str, List[float]] = {
        fi: [] for fi in fdg.flatten_failure_instances
    }
    for metric_name, score in metric_scores.items():
        for FI in fdg.metric_to_FI_list_dict[metric_name]:
            scores_dict[FI].append(score)
    ret = {}
    for fi, scores in scores_dict.items():
        if agg_method == "sum":
            ret[fi] = sum(scores)
        elif agg_method == "mean":
            ret[fi] = sum(scores) / len(scores)
        elif agg_method == "min":
            ret[fi] = min(scores)
        elif agg_method == "max":
            ret[fi] = max(scores)
        else:
            raise ValueError(f"Unknown agg_method: {agg_method}")
    return ret


def workflow(config: RandomWalkSingleMetricConfig, in_jvm_context: bool = False):
    logger.info(
        f"\n================================================Config=============================================\n"
        f"{config!s}"
        f"\n===================================================================================================\n"
    )
    logger.info(f"reproducibility info: {config.get_reproducibility_info()}")
    logger.add(config.output_dir / 'log')
    config.save(str(config.output_dir / "config"))

    base = FDGModelInterface(config)
    fdg = base.fdg
    mp = base.metric_preprocessor

    y_trues = []
    y_preds = []

    if not in_jvm_context:
        from pycausal.pycausal import pycausal as pc
        pc = pc()
        pc.start_vm()
    try:
        with profile("random_walk_main", report_printer=lambda _: logger.info(f"\n{_}")) as profiler:
            for fid in tqdm(base.test_failure_ids):
                cache_dir = Path("/tmp/single_metric_random_walk_cache") / config.data_dir.relative_to("/") / f"{fid=}"
                logger.info(f"Cache dir: {cache_dir}")
                cache = Cache(
                    directory=str(cache_dir),
                    size_limit=int(1e10),
                )
                failure_ts = fdg.failure_at(fid)["timestamp"]
                metric_scores = random_walk(
                    mp.flatten_features[:, mp.get_timestamp_indices(failure_ts, window_size=config.window_size)],
                    mp.flatten_metric_names,
                    config=config,
                    cache=cache,
                )
                failure_instance_scores = agg_scores_by_failure_instance(
                    fdg, metric_scores, agg_method=config.score_aggregation_method
                )
                y_trues.append(set(fdg.root_cause_instances_of(fid)))
                y_preds.append(
                    sorted(list(failure_instance_scores.keys()), key=lambda x: failure_instance_scores[x], reverse=True))
    finally:
        if not in_jvm_context:
            pc.stop_vm()

    for y_true, y_pred in zip(y_trues, y_preds):
        logger.info(
            f"{';'.join(y_true):<30}"
            f"|{', '.join(y_pred[:5]):<50}"
        )
    metrics = get_evaluation_metrics_dict(y_trues, y_preds, max_rank=fdg.n_failure_instances)
    logger.info(format_result_string(
        metrics,
        profiler,
        DejaVuConfig().from_dict(args_dict=config.as_dict(), skip_unsettable=True)
    ))
    return y_trues, y_preds
