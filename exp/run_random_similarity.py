import random
from functools import lru_cache
from typing import Dict

import pandas as pd
from loguru import logger

from SSF.config import SSFConfig
from SSF.embedding_model_interface.cnn_infograph import CNNInfoGraphEmbedding
from SSF.similar_failure_evaluation import top_k_precision


def main(config: SSFConfig):
    model = CNNInfoGraphEmbedding(config=config)
    fdg = model.fdg
    failures_df = fdg.failures_df
    train_failure_classes = [
        {
            fdg.instance_to_class(instance)
            for instance in failures_df.iloc[fid]['root_cause_node'].split(';')
        }
        for fid in range(len(model.train_failure_ids))
    ]
    test_failure_classes = [
        {
            fdg.instance_to_class(instance)
            for instance in failures_df.iloc[fid]['root_cause_node'].split(';')
        }
        for fid in range(len(model.test_failure_ids))
    ]
    top_1_precision = sum(
        [
            len([_ for _ in train_failure_classes if len(_.intersection(tfc)) > 0]) / len(train_failure_classes)
            for tfc in test_failure_classes
        ]
    ) / len(test_failure_classes)
    # the result is the same until the number of matches in training is less than k
    # https://www.wolframalpha.com/input/?i=simplify%28sum+i+%2F+k+*+C%28m%2C+i%29+*+C%28n+-+m%2C+k+-+i%29+%2F+C%28n%2C+k%29%2C+i+from+0+to+k%29
    logger.info(f"theoretical top-k precision: {top_1_precision * 100:3.2f}%")

    sampled_top_k_precision = []
    for _ in range(1000):
        sampled_prediction_failure_classes = []
        for i in model.test_failure_ids:
            __ret = train_failure_classes.copy()
            random.shuffle(__ret)
            sampled_prediction_failure_classes.append(__ret)
        sampled_top_k_precision.append({
            f"Top-{k:<2} Precision": top_k_precision(
                k=k, predictions=sampled_prediction_failure_classes, ground_truths=test_failure_classes
            ) for k in [1, 2, 3, 5, 10]
        })
    sampled_top_k_precision = pd.DataFrame.from_records(sampled_top_k_precision).mean().to_dict()

    logger.info(f"sample top-1  precision: {sampled_top_k_precision['Top-1  Precision'] * 100:3.2f}%")
    logger.info(f"sample top-2  precision: {sampled_top_k_precision['Top-2  Precision'] * 100:3.2f}%")
    logger.info(f"sample top-3  precision: {sampled_top_k_precision['Top-3  Precision'] * 100:3.2f}%")
    logger.info(f"sample top-5  precision: {sampled_top_k_precision['Top-5  Precision'] * 100:3.2f}%")
    logger.info(f"sample top-10 precision: {sampled_top_k_precision['Top-10 Precision'] * 100:3.2f}%")


if __name__ == '__main__':
    main(SSFConfig().parse_args())
