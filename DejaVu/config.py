from typing import Literal

from failure_dependency_graph import FDGBaseConfig


# noinspection PyPackageRequirements


class DejaVuConfig(FDGBaseConfig):
    # training parameters
    early_stopping_epoch_patience: int = 500

    checkpoint_metric: Literal['val_loss', "MAR"] = "val_loss"

    # 太小了会导致过拟合，效果下降；但是比较大的时候VAE是无法收敛的
    weight_decay: float = 1e-2

    ############################
    # FDG
    ############################
    drop_FDG_edges_fraction: float = 0.

    # model parameters
    dropout: bool = False
    augmentation: bool = False

    ################################################
    # Random Walk
    p: float = 1 / 4
    q: float = 1 / 4
    random_walk_length: int = 8

    ###############################################
    # GAT
    GAT_num_heads: int = 1
    GAT_residual: bool = True
    GAT_layers: int = 1
    GAT_shared_feature_mapper: bool = False

    ################################################
    # tsfresh
    ts_feature_mode: Literal['full', 'simple', 'minimal', 'simple_fctype'] = 'full'

    def configure(self) -> None:
        super().configure()
        self.add_argument("-aug", "--augmentation")
        self.add_argument("-bal", "--balance_train_set")
        self.add_argument("-H", "--GAT_num_heads")
        self.add_argument("-L", "--GAT_layers")
        self.add_argument("-tss", "--train_set_sampling")
