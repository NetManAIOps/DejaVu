from typing import Any, List, Set, Optional, Callable

import dgl
import torch as th
from pyprof import profile

from DejaVu.config import DejaVuConfig
from DejaVu.dataset import DejaVuDataset
from DejaVu.evaluation_metrics import top_1_accuracy, top_2_accuracy, top_3_accuracy, top_k_accuracy, MAR
from DejaVu.models.interface.loss import binary_classification_loss
from failure_dependency_graph import FDGModelInterface, FDG


class DejaVuModuleProtocol(th.nn.Module):
    def forward(self, features: List[th.Tensor], graphs: List[dgl.DGLGraph]):
        raise NotImplementedError


class DejaVuModelInterface(FDGModelInterface[DejaVuConfig, DejaVuDataset]):
    def __init__(
            self, config: DejaVuConfig,
            get_model: Callable[[FDG, DejaVuConfig], DejaVuModuleProtocol],
    ):
        super().__init__(config)
        self._module = get_model(self.fdg, config)

        # temporary variables to save outputs
        self.preds_list: List[List[int]] = []
        self.labels_list: List[Set[int]] = []
        self.probs_list: List[List[float]] = []

    @property
    def module(self) -> th.nn.Module:
        return self._module

    def forward(self, *args, **kwargs) -> Any:
        return self.module(*args, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        self._train_dataset = DejaVuDataset(
            cdp=self.fdg,
            feature_extractor=self.metric_preprocessor,
            fault_ids=self.train_failure_ids * self.config.train_set_repeat,
            window_size=self.config.window_size,
            augmentation=self.config.augmentation,
            normal_data_weight=1. if self.config.augmentation else 0.,
            drop_edges_fraction=self.config.drop_FDG_edges_fraction,
            device=self.device,
        )
        self._validation_dataset = DejaVuDataset(
            cdp=self.fdg,
            feature_extractor=self.metric_preprocessor,
            fault_ids=self.validation_failure_ids,
            window_size=self.config.window_size,
            augmentation=False,
            drop_edges_fraction=self.config.drop_FDG_edges_fraction,
            device=self.device,
        )
        self._test_dataset = DejaVuDataset(
            cdp=self.fdg,
            feature_extractor=self.metric_preprocessor,
            fault_ids=self.test_failure_ids,
            window_size=self.config.window_size,
            augmentation=False,
            drop_edges_fraction=self.config.drop_FDG_edges_fraction,
            device=self.device,
        )

    def get_collate_fn(self, batch_size: int):
        if batch_size is None:
            @profile
            def collate_fn(batch_data):
                features_list, label, failure_id, graph = batch_data
                return [v.type(th.float32) for v in features_list], label, failure_id, graph
        else:
            @profile
            def collate_fn(batch_data):
                feature_list_list, labels_list, failure_id_list, graph_list = tuple(map(list, zip(*batch_data)))
                features_list = list(map(lambda _: th.stack(_).float(), zip(*feature_list_list)))
                # (n_node_types, batch_size, n_metrics, window_size)
                labels = th.stack(labels_list, dim=0)
                # (batch_size,)
                return features_list, labels, th.tensor(failure_id_list), graph_list
        return collate_fn

    def configure_optimizers(self):
        optimizer = th.optim.Adam(
            self.parameters(), lr=self.config.init_lr, weight_decay=self.config.weight_decay
        )
        return {
            'optimizer': optimizer,
        }

    @profile
    def training_step(self, batch_data, batch_idx):
        features, labels, failure_ids, graphs = batch_data
        probs: th.Tensor = self.forward(features, graphs)
        loss: th.Tensor = binary_classification_loss(
            probs, labels,
            gamma=0.,
        )
        if hasattr(self.module, "regularization"):
            loss += self.module.regularization() * 1e-2
        if hasattr(self.module.feature_projector, 'rec_loss'):
            rec_loss = self.module.feature_projector.rec_loss
            loss += th.mean(rec_loss) * self.config.rec_loss_weight
        self.log("loss", loss)
        valid_idx = th.where(th.any(labels, dim=1))[0]
        return {
            "loss": loss,
            "probs": probs.detach()[valid_idx],
            "labels": labels[valid_idx],
        }

    def training_epoch_end(self, outputs) -> None:
        probs = th.concat([out["probs"] for out in outputs])
        labels = th.concat([out["labels"] for out in outputs])
        label_list: List[Set[Any]] = [set(th.where(label >= 1)[0].tolist()) for label in labels]
        pred_list: List[List] = th.argsort(probs, dim=-1, descending=True).tolist()
        metrics = {
            "A@1": top_1_accuracy(label_list, pred_list),
            "A@2": top_2_accuracy(label_list, pred_list),
            "A@3": top_3_accuracy(label_list, pred_list),
            "A@5": top_k_accuracy(label_list, pred_list, k=5),
            "MAR": MAR(label_list, pred_list, max_rank=self.fdg.n_failure_instances),
        }
        self.log_dict(metrics)

    @profile
    def validation_step(self, batch, batch_idx):
        features, labels, failure_ids, graphs = batch
        probs = self.forward(features, graphs)
        loss = binary_classification_loss(probs, labels, gamma=0.)
        self.log("val_loss", loss)
        return {
            "val_loss": loss,
            "probs": probs.detach(),
            "labels": labels,
        }

    def validation_epoch_end(self, outputs) -> None:
        probs = th.concat([out["probs"] for out in outputs])
        labels = th.concat([out["labels"] for out in outputs])
        label_list: List[Set[Any]] = [set(th.where(label >= 1)[0].tolist()) for label in labels]
        pred_list: List[List] = th.argsort(probs, dim=-1, descending=True).tolist()
        metrics = {
            "A@1": top_1_accuracy(label_list, pred_list),
            "A@2": top_2_accuracy(label_list, pred_list),
            "A@3": top_3_accuracy(label_list, pred_list),
            "A@5": top_k_accuracy(label_list, pred_list, k=5),
            "MAR": MAR(label_list, pred_list, max_rank=self.fdg.n_failure_instances),
        }
        self.log_dict(metrics)

    @profile
    def test_step(self, batch, batch_idx):
        features, labels, failure_ids, graphs = batch
        probs = self.forward(features, graphs)
        return {
            "probs": probs.detach(),
            "labels": labels,
        }

    def test_epoch_end(self, outputs) -> None:
        probs = th.concat([out["probs"] for out in outputs])
        labels = th.concat([out["labels"] for out in outputs])
        label_list: List[Set[Any]] = [set(th.where(label >= 1)[0].tolist()) for label in labels]
        pred_list: List[List] = th.argsort(probs, dim=-1, descending=True).tolist()
        self.labels_list = label_list
        self.preds_list = pred_list
        self.probs_list = probs.tolist()
        metrics = {
            "A@1": top_1_accuracy(label_list, pred_list),
            "A@2": top_2_accuracy(label_list, pred_list),
            "A@3": top_3_accuracy(label_list, pred_list),
            "A@5": top_k_accuracy(label_list, pred_list, k=5),
            "MAR": MAR(label_list, pred_list, max_rank=self.fdg.n_failure_instances),
        }
        self.log_dict(metrics)
