from typing import runtime_checkable, Protocol

import numpy as np

from DejaVu.config import DejaVuConfig
from DejaVu.models import GAT, DejaVuModuleProtocol, MLP
from failure_dependency_graph import FDG
from metric_preprocess import get_input_tensor_size_list

__all__ = [
    'get_GAT_model'
]


def get_GAT_model(fdg: FDG, config: DejaVuConfig) -> DejaVuModuleProtocol:
    model = GAT(
        node_types=fdg.failure_classes,
        input_features=get_input_tensor_size_list(fdg=fdg, window_size=config.window_size),
        feature_size=config.FI_feature_dim,
        feature_projector_type=config.feature_projector_type,
        num_heads=config.GAT_num_heads,
        residual=config.GAT_residual,
        GAT_layers=config.GAT_layers,
        has_dropout=config.dropout,
        shared_feature_mapper=config.GAT_shared_feature_mapper,
    )
    return model


def get_DNN_model(cdp: FDG, config: DejaVuConfig) -> DejaVuModuleProtocol:
    model = MLP(
        node_types=cdp.failure_classes,
        input_features=get_input_tensor_size_list(fdg=cdp, window_size=config.window_size),
        feature_size=config.FI_feature_dim,
        feature_projector_type=config.feature_projector_type,
        has_dropout=config.dropout,
    )
    return model


@runtime_checkable
class ClassifierProtocol(Protocol):
    def fit(self, X, y):
        ...

    def predict_proba(self, X) -> np.ndarray:
        ...


def get_RF_model(cdp: FDG, config: DejaVuConfig) -> ClassifierProtocol:
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(verbose=0)
