import copy
import io
import time
from datetime import datetime
from functools import reduce
from typing import Callable, Any, Optional

import pytorch_lightning as pl
import pytz
import torch as th
from loguru import logger
from pytorch_lightning.trainer.supporters import CombinedDataset

from DejaVu.dataset import DejaVuDataset
from DejaVu.evaluation_metrics import top_1_accuracy, top_2_accuracy, top_3_accuracy, top_k_accuracy, MAR, get_rank
from DejaVu.models.interface.model_interface import DejaVuModelInterface
from failure_dependency_graph import FDG


class CFLLoggerCallback(pl.Callback):
    def __init__(self, epoch_freq: int = 1, print_func: Callable[[str], Any] = logger.info,
                 second_freq: float = 10):
        self.print = print_func
        self.epoch_freq = epoch_freq
        self.second_freq = second_freq
        self._last_display_time = time.time()
        self._need_display_on_this_epoch = False

    def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if pl_module.current_epoch % self.epoch_freq != 0 and time.time() - self._last_display_time < self.second_freq:
            self._need_display_on_this_epoch = False
        else:
            self._last_display_time = time.time()
            self._need_display_on_this_epoch = True

    def on_train_epoch_end(
            self, trainer: 'pl.Trainer', pl_module: DejaVuModelInterface, unused: Optional = None
    ) -> None:
        if not self._need_display_on_this_epoch:
            return
        self.print(
            f"epoch={pl_module.current_epoch:<5.0f} "
            f"loss={trainer.callback_metrics.get('loss'):<10.4f}"
            f"A@1={trainer.callback_metrics.get('A@1', -1) * 100:<5.2f}% "
            f"A@2={trainer.callback_metrics.get('A@2', -1) * 100:<5.2f}% "
            f"A@3={trainer.callback_metrics.get('A@3', -1) * 100:<5.2f}% "
            f"A@5={trainer.callback_metrics.get('A@5', -1) * 100:<5.2f}% "
            f"MAR={trainer.callback_metrics.get('MAR', -1):<5.2f} "
        )

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: DejaVuModelInterface) -> None:
        self.print(
            f"epoch={pl_module.current_epoch:<5.0f} "
            f"val_loss={trainer.callback_metrics.get('val_loss', -1):<10.4f} "
            f"A@1={trainer.callback_metrics.get('A@1', -1) * 100:<5.2f}% "
            f"A@2={trainer.callback_metrics.get('A@2', -1) * 100:<5.2f}% "
            f"A@3={trainer.callback_metrics.get('A@3', -1) * 100:<5.2f}% "
            f"A@5={trainer.callback_metrics.get('A@5', -1) * 100:<5.2f}% "
            f"MAR={trainer.callback_metrics.get('MAR', -1):<5.2f} "
        )

    def on_test_epoch_end(self, trainer: 'pl.Trainer', pl_module: DejaVuModelInterface) -> None:
        labels_list = pl_module.labels_list
        preds_list = pl_module.preds_list
        output = io.StringIO()
        print(
            f"A@1={top_1_accuracy(labels_list, preds_list) * 100:<5.2f}% "
            f"A@2={top_2_accuracy(labels_list, preds_list) * 100:<5.2f}% "
            f"A@3={top_3_accuracy(labels_list, preds_list) * 100:<5.2f}% "
            f"A@5={top_k_accuracy(labels_list, preds_list, k=5) * 100:<5.2f}% "
            f"MAR={MAR(labels_list, preds_list, max_rank=pl_module.fdg.n_failure_instances):<5.2f} ",
            file=output
        )
        train_rc_ids = reduce(
            lambda a, b: a | b,
            [
                {pl_module.fdg.instance_to_gid(_) for _ in pl_module.fdg.root_cause_instances_of(fid)}
                for fid in pl_module.train_failure_ids
            ]
        )

        cdp: FDG = pl_module.fdg
        tz = pytz.timezone('Asia/Shanghai')

        print(
            f"|{'id':4}|{'':<5}|{'FR':<3}|{'AR':<3}|{'recurring':<9}|{'timestamp':<25}|"
            f"{'root cause':<70}|{'rank-1':<20}|{'rank-2':<20}|{'rank-3':<20}|",
            file=output
        )
        for preds, fault_id, labels in zip(preds_list, pl_module.test_failure_ids, labels_list):
            is_correct = preds[0] in labels
            ranks = get_rank(labels, preds)
            is_recurring = all([_ in train_rc_ids for _ in labels])
            print(
                f"|{fault_id:<4.0f}|"
                f"{'✅' if is_correct else '❌':<5}|"
                f"{min(ranks):3.0f}|"
                f"{sum(ranks) / len(ranks):3.0f}|"
                f"{is_recurring!s:<9}|"
                f"{datetime.fromtimestamp(cdp.failure_at(fault_id)['timestamp']).astimezone(tz).isoformat():<25}|"
                f"{','.join([cdp.gid_to_instance(_) for _ in labels]):<70}|"
                f"{cdp.gid_to_instance(preds[0]):<20}|"
                f"{cdp.gid_to_instance(preds[1]):<20}|"
                f"{cdp.gid_to_instance(preds[2]):<20}|",
                file=output
            )
        self.print(f"\n{output.getvalue()}")


class TestCallback(pl.Callback):
    def __init__(self, model: DejaVuModelInterface, second_freq: float = 30, epoch_freq: int = 100):
        super(TestCallback, self).__init__()
        self.epoch_freq = epoch_freq
        self.second_freq = second_freq
        self.trainer = pl.Trainer(
            callbacks=[
                CFLLoggerCallback()
            ]
        )
        self.model = copy.deepcopy(model)
        setattr(self.model, "metric_preprocessor", model.metric_preprocessor)

        self._last_time = time.time()
        self._last_epoch = 0

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: DejaVuModelInterface) -> None:
        epoch = pl_module.current_epoch
        if epoch - self._last_epoch > self.epoch_freq or time.time() - self._last_time > self.second_freq:
            self._last_time = time.time()
            self._last_epoch = epoch
            try:
                self.model.load_state_dict(pl_module.state_dict())
                logger.info(
                    f"\n=======================Start Test at Epoch {pl_module.current_epoch} ======================"
                )
                logger.info(f"load model from {trainer.checkpoint_callback.best_model_path}")
                self.trainer.test(
                    self.model, test_dataloaders=pl_module.test_dataloader(), verbose=False,
                    ckpt_path=trainer.checkpoint_callback.best_model_path
                )
                logger.info(
                    f"\n======================= End  Test at Epoch {pl_module.current_epoch} ======================"
                )
            except Exception as e:
                logger.error(f"Encounter Exception during test: {e!r}")
