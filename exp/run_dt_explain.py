from pathlib import Path
from pprint import pformat
from loguru import logger

from DejaVu.dataset import prepare_sklearn_dataset
from DejaVu.explanability import select_useful_columns, dt_follow
from explib import read_model
import torch as th


def explain_exp(exp_dir: Path):
    if exp_dir.is_file():
        exp_dir = exp_dir.parent
    output_dir = exp_dir / 'DT_explain'
    logger.add(output_dir / "log")

    cdp, config, cache, model, y_probs, y_preds, [
        train_dataloader, _, _
    ] = read_model(
        exp_dir
    )

    useful_column_path = output_dir / 'useful_columns.txt'
    if useful_column_path.exists():
        with open(useful_column_path, 'r') as f:
            useful_columns = eval(f.read())
    else:
        decoder_path = output_dir / 'decoder.pt'
        if decoder_path.exists():
            decoder = th.load(decoder_path)
        else:
            decoder = None
        useful_columns, decoder, _ = select_useful_columns(
            train_dataloader, model, cdp, config.window_size, config.metric_feature_dim, threshold=0.1,
            decoder=decoder,
        )
        # save useful columns
        th.save(decoder, decoder_path)
        with open(useful_column_path, 'w+') as f:
            print(pformat(useful_columns), file=f)
    dt_follow(
        cdp, output_dir, prepare_sklearn_dataset(
            cdp, config, cache, mode='simple_fctype'
        ), useful_columns, y_probs,
    )


def main(exp_dir: Path):
    explain_exp(exp_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Explain a model globally with DT')
    parser.add_argument('path', type=Path, help='The path to the exp dir')
    args = parser.parse_args()
    main(args.path)
