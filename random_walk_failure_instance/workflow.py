from pathlib import Path

from diskcache import Cache
from loguru import logger
from pyprof import profile
from tqdm import tqdm

from DejaVu.config import DejaVuConfig
from DejaVu.evaluation_metrics import get_evaluation_metrics_dict
from DejaVu.workflow import format_result_string
from failure_dependency_graph import FDGModelInterface
from random_walk_failure_instance.config import RandomWalkFailureInstanceConfig
from random_walk_failure_instance.model import random_walk


def workflow(config: RandomWalkFailureInstanceConfig):
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

    with profile("random_walk_main", report_printer=lambda _: logger.info(f"\n{_}")) as profiler:
        for fid in tqdm(base.test_failure_ids):
            cache_dir = Path("/tmp/failure_instance_random_walk_cache") / config.data_dir.relative_to("/") / f"{fid=}"
            logger.info(f"Cache dir: {cache_dir}")
            cache = Cache(
                directory=str(cache_dir),
                size_limit=int(1e10),
            )
            failure_ts = fdg.failure_at(fid)["timestamp"]
            graph = fdg.networkx_graph_at(fid).copy()
            for failure_class, class_metrics in zip(
                    fdg.failure_classes, mp(failure_ts, window_size=config.window_size)
            ):
                for failure_instance, instance_metrics in zip(
                        fdg.failure_instances[failure_class],
                        class_metrics,
                ):
                    assert instance_metrics.shape == (fdg.metric_number_dict[failure_class], sum(config.window_size))
                    graph.add_node(failure_instance, values=instance_metrics.numpy())
            failure_instance_scores = random_walk(
                graph,
                config=config,
                cache=cache,
            )
            y_trues.append(set(fdg.root_cause_instances_of(fid)))
            y_preds.append(
                sorted(list(failure_instance_scores.keys()), key=lambda x: failure_instance_scores[x], reverse=True))

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
