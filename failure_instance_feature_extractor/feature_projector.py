from typing import List, Union, Type

import torch as th

from failure_instance_feature_extractor.AE import AEFeatureModule
from failure_instance_feature_extractor.CNN import CNNFeatureModule
from failure_instance_feature_extractor.CNN_AE import CNNAEFeatureModule
from failure_instance_feature_extractor.GRU import GRUFeatureModule
from failure_instance_feature_extractor.GRU_AE import GRUAEFeatureModule
from failure_instance_feature_extractor.GRU_VAE import GRUVAEFeatureModule


class FIFeatureExtractor(th.nn.Module):
    """
    从每个Failure Instance的指标提取统一的特征
    """
    def __init__(
            self,
            failure_classes: List[str],
            input_tensor_size_list: List[th.Size], feature_size: int,
            feature_projector_type: Union[str, Type[th.nn.Module]] = 'CNN'
    ):
        super().__init__()
        self.node_types = failure_classes
        self.features_projection_module_list = th.nn.ModuleList()

        for node_type, input_size in zip(failure_classes, input_tensor_size_list):
            if feature_projector_type == 'CNN':
                self.features_projection_module_list.append(CNNFeatureModule(
                    input_size, feature_size
                ))
            elif feature_projector_type == "GRU":
                self.features_projection_module_list.append(GRUFeatureModule(
                    input_size, feature_size, num_layers=1,
                ))
            elif feature_projector_type == 'AE':
                self.features_projection_module_list.append(AEFeatureModule(input_size, feature_size - 1))
            elif feature_projector_type == 'GRU_AE':
                self.features_projection_module_list.append(GRUAEFeatureModule(input_size, feature_size - 1))
            elif feature_projector_type == 'GRU_VAE':
                self.features_projection_module_list.append(GRUVAEFeatureModule(input_size, feature_size - 1))
            elif feature_projector_type == 'CNN_AE':
                self.features_projection_module_list.append(CNNAEFeatureModule(input_size, feature_size - 1))
            elif issubclass(feature_projector_type, th.nn.Module):
                self.features_projection_module_list.append(feature_projector_type(input_size, feature_size))
            else:
                raise RuntimeError(f"Unknown {feature_projector_type=}")
        if isinstance(feature_projector_type, str) and feature_projector_type.endswith("AE"):
            self.rec_loss = th.zeros(1, 1, 1)

    def forward(self, x: List[th.Tensor]):
        """
        :param x: [type_1_node_features in shape (n_nodes, n_metrics, n_timestamps), type_2_nodes_features, ]
        :return: (n_nodes, feature_size)
        """
        feat_list = []
        # https://discuss.pytorch.org/t/runtimeerror-can-not-iterate-over-a-module-list-or-tuple-with-a-value-that-does-not-have-a-statically-determinable-length/118555
        for idx, module in enumerate(self.features_projection_module_list):
            input_x = x[idx]
            feat_list.append(module(input_x))
        if hasattr(self, 'rec_loss'):
            self.rec_loss = th.cat([_.rec_loss for _ in self.features_projection_module_list], dim=1)
        feat = th.cat(feat_list, dim=-2)  # (batch_size, N, feature_size)
        return feat
