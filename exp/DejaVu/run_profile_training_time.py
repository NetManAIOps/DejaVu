import random
import shlex
from pprint import pprint

import pandas as pd
import pytorch_lightning as pl
import torch.cuda
from loguru import logger

from DejaVu.config import DejaVuConfig
from DejaVu.models import DejaVuModelInterface, get_GAT_model
from DejaVu.models.interface import CFLLoggerCallback
from failure_dependency_graph.virtual_metric_preprocessor import FakeMetricPreprocessor
from utils.callbacks import EpochTimeCallback


def count_epoch_time(n_instances: int, n_metrics: int, n_failures: int) -> float:
    config = DejaVuConfig().parse_args(
        shlex.split(
            f"--data_dir=/dev/generate_random_FDG/"
            f"{n_instances=}/{n_metrics=}/{n_failures=}/n_classes={min(10, n_instances)} "
            f"--max_epoch=25 --display_epoch_freq=1 "
            f"-fe=GRU -bal=True -H=4 -L=8 ",
        )
    )
    model = DejaVuModelInterface(config, get_GAT_model)
    setattr(model, 'metric_preprocessor', FakeMetricPreprocessor(model.fdg))
    epoch_time_callback = EpochTimeCallback()
    trainer = pl.Trainer(
        auto_lr_find=True,
        max_epochs=config.max_epoch,
        callbacks=[
            epoch_time_callback,
            CFLLoggerCallback(epoch_freq=config.display_epoch_freq, second_freq=config.display_second_freq),
        ],
        default_root_dir=str(config.output_dir),
        check_val_every_n_epoch=config.valid_epoch_freq,
        num_sanity_val_steps=-1,
        enable_progress_bar=False,
        gradient_clip_val=config.gradient_clip_val,
        auto_select_gpus=True if config.cuda else False,
        gpus=1 if config.cuda else 0,
    )
    trainer.fit(model)
    return epoch_time_callback.average_epoch_time()


def exp():
    records = []

    def __add(n_instances: int, n_metrics: int, n_failures: int):
        try:
            records.append({
                'n_instances': n_instances,
                'n_metrics': n_metrics,
                'n_failures': n_failures,
                'epoch_time': count_epoch_time(n_instances, n_metrics, n_failures),
                "gpu": torch.cuda.is_available(),
            })
            pprint(records[-1])
        except Exception as e:
            logger.exception(f"Failed to run experiment with {n_instances=} {n_metrics=} {n_failures=}", exception=e)

    jobs_list = []
    for _ in range(10):
        for i in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
            jobs_list.append((i, 2048, 100))
        for i in [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]:
            jobs_list.append((100, i, 100))
        for i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            jobs_list.append((100, 2048, i))

    random.shuffle(jobs_list)
    for job in jobs_list:
        __add(*job)

    print(pd.DataFrame.from_records(records).to_csv(index=False))


if __name__ == '__main__':
    exp()
