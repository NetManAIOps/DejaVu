import time
from collections import defaultdict
from typing import Callable, Any, Optional, List, Dict

import pytorch_lightning as pl
from loguru import logger
from pyprof import Profiler, current_profiler
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT


class TrainingProcessLoggerCallback(pl.Callback):
    def __init__(
            self, *,
            epoch_freq: int = 1,
            print_func: Callable[[str], Any] = logger.info,
            second_freq: float = 10,
            train_metrics: Optional[List[str]] = None,
            validation_metrics: Optional[List[str]] = None,
            metrics: Optional[List[str]] = None
    ):
        if metrics is None:
            metrics = []
        if validation_metrics is None:
            validation_metrics = ['val_loss']
        if train_metrics is None:
            train_metrics = ['loss']
        self.print = print_func
        self.epoch_freq = epoch_freq
        self.second_freq = second_freq
        self._last_display_time = time.time()
        self._need_display_on_this_epoch = False
        self._train_metrics = train_metrics + metrics
        self._validation_metrics = validation_metrics + metrics

        # By default, PL logs metrics on step in train_step and on epoch in validation/test_step.
        # So we only accumulate step metric values for train steps.
        self._train_step_metric_values = defaultdict(list)

    def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if pl_module.current_epoch % self.epoch_freq != 0 and time.time() - self._last_display_time < self.second_freq:
            self._need_display_on_this_epoch = False
        else:
            self._last_display_time = time.time()
            self._need_display_on_this_epoch = True

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            unused: Optional[int] = 0,
    ) -> None:
        for metric in self._train_metrics:
            if metric in trainer.callback_metrics:
                self._train_step_metric_values[metric].append(trainer.callback_metrics[metric])

    def on_train_epoch_end(
            self, trainer: 'pl.Trainer', pl_module, unused: Optional = None
    ) -> None:
        if not self._need_display_on_this_epoch:
            return
        self.print(
            f"epoch={pl_module.current_epoch:<5.0f} " +
            ' '.join(
                f"{metric}="
                f"{self._get_metric_value(metric, trainer.callback_metrics, self._train_step_metric_values):<6.2f}"
                for metric in self._train_metrics
            )
        )
        self._train_step_metric_values.clear()

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module) -> None:
        self.print(
            f"epoch={pl_module.current_epoch:<5.0f} " +
            ' '.join(
                f"{metric}="
                f"{self._get_metric_value(metric, trainer.callback_metrics, {}):<6.2f}"
                for metric in self._validation_metrics
            )
        )

    @staticmethod
    def _get_metric_value(metric, last_metric_values: Dict[str, float], step_metric_values: [str, List[float]]):
        if metric in step_metric_values:
            return sum(step_metric_values[metric]) / len(step_metric_values[metric])
        else:
            return last_metric_values.get(metric, -1)


class EpochTimeCallback(Callback):
    def __init__(self, printer: Callable[[str], Any] = print):
        super().__init__()
        self._profiler = Profiler("Epoch Time", parent=current_profiler())
        self._printer = printer

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._profiler.tic()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._profiler.toc()

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._printer(f"Average epoch time: {self.average_epoch_time():.2f}")

    def average_epoch_time(self) -> float:
        return self._profiler.average
