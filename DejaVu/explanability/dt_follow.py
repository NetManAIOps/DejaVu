import copy
import json
from pathlib import Path
from subprocess import CalledProcessError
from typing import Tuple, Set, Dict, Any, Optional, List

import pandas as pd
import pytorch_lightning as pl
import torch as th
from einops import rearrange
from loguru import logger
from sklearn.tree import DecisionTreeClassifier

from DejaVu.dataset import DejaVuDataset, SKLEARN_DATASET_TYPE
from DejaVu.explanability.fe_decoder import FeatureExtractorDecoder
from DejaVu.models import DejaVuModelInterface
from failure_dependency_graph import FDG
from failure_instance_feature_extractor import FIFeatureExtractor
from failure_instance_feature_extractor.GRU import GRUFeatureModule


class FCEncoder(GRUFeatureModule):
    def __init__(self, module: GRUFeatureModule):
        super().__init__(
            input_size=th.Size((module.n_instances, module.x_dim, module.n_ts)),
            embedding_size=module.z_dim,
            num_layers=module.num_layers
        )
        assert isinstance(module, GRUFeatureModule)
        self.encoder = copy.deepcopy(module.encoder)

        self.unify_mapper = th.nn.Identity()

    def forward(self, x):
        z = super().forward(x)
        return rearrange(z, "batch instance z ts -> batch instance (z ts)")


class Encoder(th.nn.Module):
    def __init__(self, feature_projector: FIFeatureExtractor):
        super().__init__()
        self.features_projection_module_list = th.nn.ModuleList([
            FCEncoder(_) for _ in feature_projector.features_projection_module_list
        ])

    def forward(self, x: List[th.Tensor]):
        feat_list = []
        # https://discuss.pytorch.org/t/runtimeerror-can-not-iterate-over-a-module-list-or-tuple-with-a-value-that-does-not-have-a-statically-determinable-length/118555
        for idx, module in enumerate(self.features_projection_module_list):
            input_x = x[idx]
            feat_list.append(module(input_x))
        feat = th.cat(feat_list, dim=-2)  # (batch_size, N, feature_size)
        return feat


class FeatureExtractorDecoderModelInterface(pl.LightningModule):
    def __init__(self, model: DejaVuModelInterface, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.cdp = model.fdg
        self._encoder = [Encoder(model.module.feature_projector)]
        from metric_preprocess import get_input_tensor_size_list
        self.decoder = FeatureExtractorDecoder(
            get_input_tensor_size_list(fdg=self.cdp, window_size=model.config.window_size),
            embedding_size=model.config.FI_feature_dim,
            # embedding_size=model.config.FI_feature_dim,
        )

    def to(self, device: th.device):
        super().to(device)
        self.encoder.to(device)

    @property
    def encoder(self):
        return self._encoder[0]

    def configure_optimizers(self):
        from torch.optim import Adam
        return Adam(self.decoder.parameters(), lr=1e-1, weight_decay=0.)

    @staticmethod
    def loss(x: List[th.Tensor], rec_x: List[th.Tensor]) -> th.Tensor:
        loss = th.zeros(1, device=x[0].device, dtype=x[0].dtype)
        for _x, _rec_x in zip(x, rec_x):
            loss += th.mean(th.square(_x - _rec_x))
        return loss / th.tensor(len(x))

    def training_step(self, batch, batch_idx):
        feat, label, fid, graph = batch
        with th.no_grad():
            z = self.encoder(feat).detach()
        rec_x = self.decoder(z)
        return self.loss(feat, rec_x)

    def validation_step(self, batch, batch_idx):
        feat, label, fid, graph = batch
        with th.no_grad():
            z = self.encoder(feat).detach()
            rec_x = self.decoder(z)
        self.log('val_loss', self.loss(feat, rec_x), prog_bar=True)


def train_feature_extractor_decoder(
        model: DejaVuModelInterface
) -> FeatureExtractorDecoder:
    dataloader = model.train_dataloader()
    assert isinstance(dataloader.dataset, DejaVuDataset)
    decoder_model = FeatureExtractorDecoderModelInterface(model)
    trainer = pl.Trainer(
        auto_lr_find=True,
        max_epochs=10000,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=50,
                mode='min',
            ),
        ],
        num_sanity_val_steps=-1,
        gradient_clip_val=10.,
        auto_select_gpus=True if th.cuda.is_available() else False,
        check_val_every_n_epoch=10,
        gpus=1 if th.cuda.is_available() else 0,
    )
    trainer.fit(decoder_model, train_dataloaders=model.train_dataloader(), val_dataloaders=model.val_dataloader())
    return decoder_model.decoder


def select_useful_columns(
        model: DejaVuModelInterface,
        decoder: FeatureExtractorDecoder,
        mode: str = 'simple_fctype', threshold: float = 0.1,
) -> Tuple[Set[str], pd.Series, pd.DataFrame]:
    import numpy as np
    dataloader = model.train_dataloader_orig()
    encoder = Encoder(model.module.feature_projector)
    cdp = model.fdg
    logger.info("Get reconstructed X")
    feat, *_ = tuple(zip(*dataloader))
    feat = [th.cat(_, dim=0) for _ in list(zip(*feat))]
    feat = [_.to(model.device) for _ in feat]
    z = encoder(feat)
    rec_x = decoder(z)

    logger.info("Prepare Dataframe")
    orig_ts_df_records = []
    rec_ts_df_records = []
    rec_loss_df_records = []
    for i, fault_id in enumerate(dataloader.dataset.fault_ids):
        for j in range(cdp.n_failure_instances):
            node = cdp.gid_to_instance(j)
            node_type, typed_id = cdp.gid_to_local_id(j)
            node_type_idx = cdp.failure_classes.index(node_type)
            for k, metric in enumerate(cdp.FC_metrics_dict[node_type]):
                _feat = feat[node_type_idx][i, typed_id, k, :].detach().cpu().numpy()
                _rec_x = rec_x[node_type_idx][i, typed_id, k, :].detach().cpu().numpy()
                orig_ts_df_records.extend(
                    [{
                        'kind': f"{node}##{metric}", 'time': t, 'id': fault_id, 'value': v.item(),
                    } for t, v in enumerate(_feat)]
                )
                rec_ts_df_records.extend(
                    [{
                        'kind': f"{node}##{metric}", 'time': t, 'id': fault_id, 'value': v.item(),
                    } for t, v in enumerate(_rec_x)]
                )
                rec_loss_df_records.append(
                    {
                        "metric": f"{node}##{metric}",
                        "fid": fault_id,
                        "x": _feat,
                        "rec_x": _rec_x,
                        "loss": np.mean(np.square(_feat - _rec_x)),
                    }
                )
    import pandas as pd
    orig_df = pd.DataFrame.from_records(orig_ts_df_records)
    rec_df = pd.DataFrame.from_records(rec_ts_df_records)
    rec_loss_df = pd.DataFrame.from_records(rec_loss_df_records)
    del orig_ts_df_records, rec_ts_df_records
    from tsfresh import extract_features
    if mode == 'full':
        default_fc_parameters = None
    elif mode == 'simple':
        from DejaVu.explanability import SimpleFCParameters
        default_fc_parameters = SimpleFCParameters()
    elif mode == 'simple_fctype':
        from DejaVu.explanability.ts_feature import SimpleFCTypeFCParameters
        default_fc_parameters = SimpleFCTypeFCParameters()
    elif mode == 'minimal':
        from tsfresh.feature_extraction import MinimalFCParameters
        default_fc_parameters = MinimalFCParameters()
    else:
        raise RuntimeError(f"unknown ts feature extraction mode: {mode=}")

    logger.info("Get TS features")
    orig_ts_feats = extract_features(
        orig_df, column_id='id', column_sort='time', column_kind='kind', column_value='value',
        default_fc_parameters=default_fc_parameters,
    )
    rec_ts_feats = extract_features(
        rec_df, column_id='id', column_sort='time', column_kind='kind', column_value='value',
        default_fc_parameters=default_fc_parameters,
    )
    diff = 2 * np.mean(
        np.square(rec_ts_feats.values - orig_ts_feats), axis=0
    ) / np.mean(
        np.square(rec_ts_feats.values + orig_ts_feats), axis=0
    )
    useful_columns = set([_.split('##')[1] for _ in diff[diff < threshold].sort_values(ascending=True).index])
    logger.info(
        f"select {len(useful_columns)} useful features in {len({_.split('##')[1] for _ in diff.index})} features"
    )
    return useful_columns, diff, rec_loss_df


def dt_follow(
        cdp: FDG, output_dir: Path, sklearn_dataset: SKLEARN_DATASET_TYPE,
        useful_columns: Optional[Set[str]], y_probs: th.Tensor, prune: bool = False
) -> Dict[str, DecisionTreeClassifier]:
    import numpy as np
    dataset, (_, _, test_dataset_fault_ids) = sklearn_dataset
    del sklearn_dataset
    ret_dt_dict = {}
    for node_type in cdp.failure_classes:
        feature_names, (
            (train_x, _, train_fault_ids, train_node_names), _, (test_x, _, test_fault_ids, test_node_names)
        ) = dataset[node_type]
        class_names = ['reject', 'uncertain', 'accept']
        class_thresholds = [-1e-3, 0.25, 0.75]
        if useful_columns is not None:
            useful_feature_indices = [idx for idx, _ in enumerate(feature_names) if _ in useful_columns]
        else:
            useful_feature_indices = [idx for idx, _ in enumerate(feature_names)]
        logger.info(
            f"{node_type:20} "
            f"#features={len(feature_names):6.0f} "
            f"#useful features={len(useful_feature_indices):6.0f}"
        )
        if len(useful_feature_indices) == 0:
            logger.info(f"{node_type:20} no useful features")
            continue
        feature_names = [feature_names[_] for _ in useful_feature_indices]
        train_x = train_x[:, useful_feature_indices]
        test_x = test_x[:, useful_feature_indices]
        train_y = np.zeros(len(train_x), dtype=np.int32)
        for i, (fault_id, node_name) in enumerate(zip(train_fault_ids, train_node_names)):
            train_y[i] = np.searchsorted(class_thresholds, y_probs[fault_id, cdp.instance_to_gid(node_name)],
                                         side='right')
        if np.max(train_y) < len(class_names) - 1:  # no accept class
            continue
        test_y = np.zeros(len(test_x), dtype=np.int32)
        for i, (fault_id, node_name) in enumerate(zip(test_fault_ids, test_node_names)):
            test_y[i] = np.searchsorted(class_thresholds, y_probs[fault_id, cdp.instance_to_gid(node_name)],
                                        side='right')

        init_dt = DecisionTreeClassifier()
        init_dt.fit(train_x, train_y)

        if prune:
            from sklearn.metrics import accuracy_score
            _max_accuracy = -1
            for ccp_alpha in init_dt.cost_complexity_pruning_path(train_x, train_y).ccp_alphas:
                _dt = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
                _dt.fit(train_x, train_y)
                _test_accuracy = accuracy_score(test_y, _dt.predict(test_x))
                logger.info(f"{node_type:20} ccp_alpha={ccp_alpha:.3f} test_accuracy={_test_accuracy:.3f}")
                if _test_accuracy > _max_accuracy:
                    _max_accuracy = _test_accuracy
                    ret_dt_dict[node_type] = _dt
            logger.info(f"{node_type:20} max_accuracy={_max_accuracy:.3f}")
        else:
            ret_dt_dict[node_type] = init_dt
        from graphviz import Source
        from sklearn.tree import export_graphviz
        dot = Source(export_graphviz(
            ret_dt_dict[node_type], feature_names=feature_names, proportion=False, filled=True, class_names=class_names
        ))
        output_path = output_dir / f"{node_type=}"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        try:
            dot.render(output_path, cleanup=True, **{'format': 'pdf'})
        except CalledProcessError:
            pass

    # dt_y_probs = np.zeros((len(test_dataset_fault_ids), cdp.n_failure_instances), dtype=np.float32)
    # y_trues = []
    # for fault_id in test_dataset_fault_ids:
    #     y_trues.append(set(map(cdp.instance_to_gid, cdp.root_cause_instances_of(fault_id))))
    # for node_type in cdp.failure_classes:
    #     if node_type not in ret_dt_dict:
    #         continue
    #     dt = ret_dt_dict[node_type]
    #     feature_names, (
    #         _, _, (test_x, _, test_fault_ids, test_node_names)
    #     ) = dataset[node_type]
    #     if useful_columns is not None:
    #         useful_feature_indices = [idx for idx, _ in enumerate(feature_names) if _ in useful_columns]
    #     else:
    #         useful_feature_indices = [idx for idx, _ in enumerate(feature_names)]
    #     if len(useful_feature_indices) == 0:
    #         logger.info(f"{node_type:20} no useful features")
    #         continue
    #     test_x = test_x[:, useful_feature_indices]
    #     for fault_id, node_name, prob in zip(test_fault_ids, test_node_names, dt.predict_proba(test_x)):
    #         dt_y_probs[test_dataset_fault_ids.index(fault_id), cdp.instance_to_gid(node_name)] = 1 - prob[0].item()
    # y_preds = [np.arange(len(prob))[np.argsort(prob, axis=-1)[::-1]].tolist() for prob in dt_y_probs]
    # from DejaVu.evaluation_metrics import get_evaluation_metrics_dict
    # metrics = get_evaluation_metrics_dict(y_trues, y_preds)
    # logger.info(metrics)

    return ret_dt_dict


def lr_follow(
        cdp: FDG, output_dir: Path, sklearn_dataset: SKLEARN_DATASET_TYPE,
        useful_columns: Optional[Set[str]], y_probs: th.Tensor,
) -> Dict[str, Dict[str, float]]:
    import numpy as np
    dataset, (_, _, test_dataset_fault_ids) = sklearn_dataset
    del sklearn_dataset
    ret_dict = {}
    for node_type in cdp.failure_classes:
        feature_names, (
            (train_x, _, train_fault_ids, train_node_names), _, (test_x, _, test_fault_ids, test_node_names)
        ) = dataset[node_type]
        class_names = ['reject', 'uncertain', 'accept']
        class_thresholds = [-1e-3, 0.25, 0.75]
        if useful_columns is not None:
            useful_feature_indices = [idx for idx, _ in enumerate(feature_names) if _ in useful_columns]
        else:
            useful_feature_indices = [idx for idx, _ in enumerate(feature_names)]
        logger.info(
            f"{node_type:20} "
            f"#features={len(feature_names):6.0f} "
            f"#useful features={len(useful_feature_indices):6.0f}"
        )
        if len(useful_feature_indices) == 0:
            logger.info(f"{node_type:20} no useful features")
            continue
        feature_names = [feature_names[_] for _ in useful_feature_indices]
        train_x = train_x[:, useful_feature_indices]
        test_x = test_x[:, useful_feature_indices]
        train_y = np.zeros(len(train_x), dtype=np.int32)
        for i, (fault_id, node_name) in enumerate(zip(train_fault_ids, train_node_names)):
            train_y[i] = np.searchsorted(class_thresholds, y_probs[fault_id, cdp.instance_to_gid(node_name)],
                                         side='right')
        if np.max(train_y) < len(class_names) - 1:  # no accept class
            continue
        test_y = np.zeros(len(test_x), dtype=np.int32)
        for i, (fault_id, node_name) in enumerate(zip(test_fault_ids, test_node_names)):
            test_y[i] = np.searchsorted(class_thresholds, y_probs[fault_id, cdp.instance_to_gid(node_name)],
                                        side='right')

        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(penalty="l1", solver="liblinear")
        lr.fit(train_x, train_y)
        from sklearn.metrics import accuracy_score
        logger.info(f"{node_type:20} test_accuracy={accuracy_score(test_y, lr.predict(test_x)):.3f}")

        feature_importance_dict = {
            n: i for n, i in zip(feature_names, lr.coef_[-1])
        }
        with open(output_dir / f"{node_type}_feature_importance.json", 'w+') as f:
            json.dump(feature_importance_dict, f, indent=4)
        ret_dict[node_type] = feature_importance_dict
    return ret_dict
