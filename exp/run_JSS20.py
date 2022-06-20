from diskcache import Cache

from JSS20.system_graph import GraphLibrary
from loguru import logger
from pyprof import profile

from SSF.config import SSFConfig
from failure_dependency_graph import FDG, split_failures_by_type
from SSF.similar_failure_evaluation import top_k_precision
from SSF.workflows import print_output_one_line_summary
from metric_preprocess import MetricPreprocessor


@profile('workflow')
def jss20(config: SSFConfig):
    logger.add(config.output_dir / 'log')
    fdg, real_paths = FDG.load(
        dir_path=config.data_dir,
        graph_path=config.graph_config_path, metrics_path=config.metrics_path, faults_path=config.faults_path,
        return_real_path=True,
        use_anomaly_direction_constraint=config.use_anomaly_direction_constraint
    )
    mp = MetricPreprocessor(fdg)
    train_ids, _, test_ids = split_failures_by_type(
        fdg.failures_df, fdg=fdg, split=config.dataset_split_ratio,
        train_set_sampling_ratio=config.train_set_sampling
    )
    # train_ids = [54, 28, 77]
    # test_ids = [69, 67, 6]
    graph_library = GraphLibrary(fdg, train_ids[:], mp=mp)
    y_trues = []
    y_preds = []
    for fid, (_, fault) in zip(test_ids, fdg.failures_df.iloc[test_ids[:]].iterrows()):
        y_trues.append({fdg.instance_to_class(_) for _ in fault['root_cause_node'].split(';')})
        y_preds.append([{fdg.instance_to_class(_) for _ in item.split(';')} for item in graph_library.query(fid)])
    metrics = {
        "Top-1  Precision": top_k_precision(k=1, predictions=y_preds, ground_truths=y_trues),
        "Top-2  Precision": top_k_precision(k=2, predictions=y_preds, ground_truths=y_trues),
        "Top-3  Precision": top_k_precision(k=3, predictions=y_preds, ground_truths=y_trues),
        "Top-5  Precision": top_k_precision(k=5, predictions=y_preds, ground_truths=y_trues),
        "Top-10 Precision": top_k_precision(k=10, predictions=y_preds, ground_truths=y_trues),
    }
    logger.info(metrics)
    return metrics


def main(config: SSFConfig):
    metrics = jss20(config)
    print_output_one_line_summary(metrics, config)


if __name__ == '__main__':
    with profile("Experiment", report_printer=logger.info):
        main(SSFConfig().parse_args())
