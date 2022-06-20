from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Literal

from loguru import logger
from tap import Tap


class FDGBaseConfig(Tap):
    # Input
    gradient_clip_val: float = 1.
    es: bool = True
    early_stopping_epoch_patience: int = 250
    weight_decay: float = 1e-2
    init_lr: float = 1e-2
    max_epoch: int = 3000
    test_second_freq: float = 30.
    test_epoch_freq: int = 100
    valid_epoch_freq: int = 10
    display_second_freq: float = 5
    display_epoch_freq: int = 10
    graph_config_path: Optional[Path] = None
    metrics_path: Optional[Path] = None
    faults_path: Optional[Path] = None
    use_anomaly_direction_constraint: bool = False
    data_dir: Path = Path("/SSF/data/aiops2020_phase2/")
    cache_dir: Path = Path('/tmp/SSF/.cache')  # 用本地文件系统能加快速度
    flush_dataset_cache: bool = True

    dataset_split_ratio: Tuple[float, float, float] = (0.4, 0.2, 0.4)
    train_set_sampling: float = 1.0  # 在训练集中只取前一部分，只用于测试训练集个数对结果的影响的实验
    train_set_repeat: int = 1
    balance_train_set: bool = False

    output_base_path: Path = Path('/SSF/output')
    output_dir: Path = None

    cuda: bool = True

    ################################################
    # FEATURE PROJECTION
    ################################################
    rec_loss_weight: float = 1.
    FI_feature_dim: int = 3
    feature_projector_type: Literal['CNN', 'AE', 'GRU_AE', 'CNN_AE', 'GRU_VAE', 'GRU'] = 'CNN'

    window_size: Tuple[int, int] = (10, 10)
    batch_size: int = 16
    test_batch_size: int = 128

    def process_args(self) -> None:
        if self.output_dir is None:
            import traceback
            caller_file_path = Path(traceback.extract_stack()[-3].filename)
            self.output_dir = Path(
                self.output_base_path / f"{caller_file_path.name}.{datetime.now().isoformat()}"
            ).resolve()
            self.output_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        import torch
        logger.info(f"{torch.cuda.is_available()=}")
        self.cuda = torch.cuda.is_available() and self.cuda

    def __init__(self, *args, **kwargs):
        super().__init__(*args, explicit_bool=True, **kwargs)

    def configure(self) -> None:
        self.add_argument("-z_dim", "--FI_feature_dim")
        self.add_argument("-fe", "--feature_projector_type")
        self.add_argument("-f", "--flush_dataset_cache")
