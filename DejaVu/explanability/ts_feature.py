import os

# Disable CUDA for numba since it does not support cuda11.2 now
from pyprof import profile

from failure_dependency_graph import FDG
from metric_preprocess import MetricPreprocessor

os.environ['NUMBA_DISABLE_CUDA'] = "1"

from collections import defaultdict
from typing import Tuple, Dict

import pandas as pd
from loguru import logger
from tsfresh import extract_features
from tsfresh.feature_extraction import feature_calculators
from tsfresh.feature_extraction.settings import MinimalFCParameters, ComprehensiveFCParameters

__all__ = ['SimpleFCParameters', 'prepare_ts_feature_dataset', 'SimpleFCTypeFCParameters']


class SimpleFCParameters(MinimalFCParameters):
    def __init__(self):
        super().__init__()

        for fname, f in feature_calculators.__dict__.items():
            if fname in self and ('sum_values' in fname or 'length' in fname):
                del self[fname]


class SimpleFCTypeFCParameters(ComprehensiveFCParameters):
    def __init__(self):
        ComprehensiveFCParameters.__init__(self)

        for fname, f in feature_calculators.__dict__.items():
            if fname in self and (not hasattr(f, 'fctype') or f.fctype != 'simple'):
                logger.debug(f"del {fname=}")
                del self[fname]
            elif fname in self:
                logger.debug(f"keep {fname=}")


@profile
def prepare_ts_feature_dataset(
        fdg: FDG, fe: MetricPreprocessor, window_size: Tuple[int, int] = (10, 10),
        mode: str = 'full',
) -> Dict[str, pd.DataFrame]:
    """
    :param mode:
    :param fdg:
    :param fe:
    :param window_size:
    :return:
    """
    # ts_df: id, time, kind, value
    ts_df_records = defaultdict(list)
    for fault_id in range(len(fdg.failures_df)):
        fault = fdg.failures_df.iloc[fault_id]
        feat = fe(
            fault_ts=fault['timestamp'],
            window_size=window_size,
        )  # [(n_nodes, n_features, n_ts)]
        for i, node_type in enumerate(fdg.failure_classes):
            for j, node in enumerate(fdg.failure_instances[node_type]):
                for k, metric in enumerate(fdg.FC_metrics_dict[node_type]):
                    ts_df_records[node_type].extend(
                        [{
                            'id': f"{fault_id=}.{node=}", 'time': t, 'kind': metric, 'value': v.item(),
                        } for t, v in enumerate(feat[i][j, k, :])]
                    )
    ret = {}
    if mode == 'full':
        default_fc_parameters = None
    elif mode == 'simple':
        default_fc_parameters = SimpleFCParameters()
    elif mode == 'simple_fctype':
        default_fc_parameters = SimpleFCTypeFCParameters()
    elif mode == 'minimal':
        default_fc_parameters = MinimalFCParameters()
    else:
        raise RuntimeError(f"unknown ts feature extraction mode: {mode=}")
    for node_type in fdg.failure_classes:
        ts_df = pd.DataFrame.from_records(ts_df_records[node_type])
        feat_df = extract_features(ts_df, column_id='id', column_sort='time', column_kind='kind', column_value='value',
                                   default_fc_parameters=default_fc_parameters)
        logger.info(f"Features for {node_type=}: {feat_df.columns}")
        ret[node_type] = feat_df
    return ret
