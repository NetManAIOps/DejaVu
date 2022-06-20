from loguru import logger
from pyprof import profile

from SSF.similar_failure_evaluation import top_k_precision
from SSF.workflows import print_output_one_line_summary
from failure_dependency_graph import FDG
from iSQUAD.config import ISQUADConfig
from iSQUAD.iSQ import ISQUARD
from metric_preprocess import MetricPreprocessor


@profile('workflow')
def exp_iSQ(config: ISQUADConfig):
    logger.add(config.output_dir / 'log')
    fdg = FDG.load(
        dir_path=config.data_dir,
        graph_path=config.graph_config_path, metrics_path=config.metrics_path, faults_path=config.faults_path,
        return_real_path=False,
        use_anomaly_direction_constraint=config.use_anomaly_direction_constraint
    )
    mp = MetricPreprocessor(fdg, granularity=60)
    isq = ISQUARD(fdg=fdg, config=config, mp=mp)
    y_preds = isq()
    y_preds = [[{fdg.gid_to_local_id(_)[0]} for _ in preds] for preds in y_preds]
    y_trues = []
    for fault_id in isq.test_fault_ids:
        y_trues.append(set(map(fdg.instance_to_class, fdg.failures_df.iloc[fault_id]["root_cause_node"].split(";"))))
    for y_true, y_pred in zip(y_trues, y_preds):
        logger.info(
            f"{';'.join(y_true):<30}"
            f"|{'; '.join([','.join(_) for _ in y_pred[:5]]):<50}"
        )
    metrics = {
        "Top-1  Precision": top_k_precision(k=1, predictions=y_preds, ground_truths=y_trues),
        "Top-2  Precision": top_k_precision(k=2, predictions=y_preds, ground_truths=y_trues),
        "Top-3  Precision": top_k_precision(k=3, predictions=y_preds, ground_truths=y_trues),
        "Top-5  Precision": top_k_precision(k=5, predictions=y_preds, ground_truths=y_trues),
        "Top-10 Precision": top_k_precision(k=10, predictions=y_preds, ground_truths=y_trues),
    }
    return metrics


def main(config: ISQUADConfig):
    metrics = exp_iSQ(config)
    print_output_one_line_summary(metrics, config)


if __name__ == '__main__':
    with profile("Experiment", report_printer=logger.info):
        main(ISQUADConfig().parse_args())
