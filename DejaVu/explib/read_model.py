from pathlib import Path
from typing import Union, Callable, Tuple, Dict

import numpy as np

__all__ = ['read_model']

from DejaVu.models import DejaVuModuleProtocol, DejaVuModelInterface
from DejaVu.config import DejaVuConfig
from failure_dependency_graph import FDG


def read_model(
        exp_dir: Union[str, Path],
        get_model: Callable[[FDG, DejaVuConfig], DejaVuModuleProtocol],
        override_config: Dict=None
) -> Tuple[DejaVuModelInterface, np.ndarray, np.ndarray]:
    #############################################################################################################
    if override_config is None:
        override_config = {}
    import pytorch_lightning as pl
    from utils.load_model import best_checkpoint

    exp_dir = Path(exp_dir)

    config = DejaVuConfig().load(path=f"{exp_dir}/config")
    config.augmentation = False  # it modifies the train dataset

    for k, v in override_config.items():
        setattr(config, k, v)

    model = DejaVuModelInterface.load_from_checkpoint(
        str(best_checkpoint(exp_dir, debug=True)),
        config=config, get_model=get_model
    )
    model.setup()
    ###############################################################################################################
    from pytorch_lightning.trainer.supporters import CombinedLoader
    from DejaVu.models.interface.callbacks import CFLLoggerCallback

    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[CFLLoggerCallback()],
        check_val_every_n_epoch=config.valid_epoch_freq,
        num_sanity_val_steps=-1,
        progress_bar_refresh_rate=0,
    )
    trainer.train_dataloader = CombinedLoader(model.train_dataloader_orig())
    trainer.test(model, test_dataloaders=model.test_dataloader(), verbose=False)
    test_preds_list = model.preds_list
    test_probs_list = model.probs_list

    trainer.test(model, test_dataloaders=model.train_dataloader_orig(), verbose=False)
    train_preds_list = model.preds_list
    train_probs_list = model.probs_list

    trainer.test(model, test_dataloaders=model.val_dataloader(), verbose=False)
    valid_preds_list = model.preds_list
    valid_probs_list = model.probs_list

    fault_ids = np.concatenate(
        [_.dataset.fault_ids for _ in [model.train_dataloader_orig(), model.val_dataloader(), model.test_dataloader()]],
        axis=0)
    actual_train_length = len(model.train_dataloader_orig().dataset.fault_ids)
    # print(test_probs_list[-10:], test_preds_list[-10:])
    y_probs = np.concatenate([train_probs_list[:actual_train_length], valid_probs_list, test_probs_list], axis=0)
    y_preds = np.concatenate([train_preds_list[:actual_train_length], valid_preds_list, test_preds_list], axis=0)
    # assert np.all(y_probs[np.arange(len(y_probs)), y_preds[:, 0]] == np.max(y_probs, axis=1))
    # assert np.all(y_probs[-20:] == np.asarray(test_probs_list)[-20:])
    # assert np.all(np.argmax(y_probs, axis=-1) == y_probs[:, 0])
    assert len(y_probs) == len(y_preds) == len(fault_ids)
    idx_sort = np.argsort(fault_ids)
    y_probs = y_probs[idx_sort]
    y_preds = y_preds[idx_sort]
    fault_ids = fault_ids[idx_sort]
    _, idx_unique = np.unique(fault_ids, return_index=True)
    y_probs = y_probs[idx_unique]
    y_preds = y_preds[idx_unique]
    fault_ids = fault_ids[idx_unique]
    # assert np.all(y_probs[np.arange(len(y_probs)), y_preds[:, 0]] == np.max(y_probs, axis=1))
    # assert np.all(np.argmax(y_probs, axis=-1) == y_probs[:, 0])
    del fault_ids, idx_sort
    y_probs = 1 / (1 + np.exp(-y_probs))
    # assert np.all(np.argmax(y_probs, axis=-1) == y_probs[:, 0])
    assert np.all(y_probs[np.arange(len(y_probs)), y_preds[:, 0]] == np.max(y_probs, axis=1))

    return model, y_probs, y_preds


if __name__ == '__main__':
    from DejaVu.models import get_GAT_model
    read_model("/data/SSF/experiment_outputs/run_GAT_node_classification.py.2021-12-20T09:34:08.620832", get_GAT_model)