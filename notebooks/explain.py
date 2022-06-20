# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
sys.path.insert(0, os.path.realpath(".."))
from DejaVu.explib.read_model import read_model

# %%
from pathlib import Path
exp_dir = Path('/data/SSF/experiment_outputs/run_GAT_node_classification.py.2021-12-22T03:50:07.949451')

# %% tags=[]
from DejaVu.models import get_GAT_model
model, y_probs, y_preds = read_model(
    exp_dir,
    get_GAT_model,
    override_config=dict(data_dir=Path("/SSF/data/A1"), flush_dataset_cache=False)
)

cdp = fdg = model.fdg
config = model.config
cache = model.cache

output_dir = config.output_dir / 'DT_explain'
output_dir.mkdir(exist_ok=True, parents=True)
print(output_dir)

# %%
from DejaVu.explanability.dt_follow import train_feature_extractor_decoder
import torch as th
decoder_path = output_dir / 'decoder.pt'
if not decoder_path.exists():
    th.save(train_feature_extractor_decoder(model=model), decoder_path)
decoder = th.load(decoder_path)

# %% tags=[]
from DejaVu.explanability import select_useful_columns
import torch as th
from pprint import pformat
import json
import torch
import pandas as pd

useful_columns_path = output_dir / 'useful_columns.txt'
if not useful_columns_path.exists() or True:
    useful_columns, diff, rec_loss_df = select_useful_columns(
        model=model, threshold=0.1,
        decoder=decoder
    )
    with open(useful_columns_path, 'w+') as f:
        json.dump(list(useful_columns), f)
    diff.to_pickle(output_dir / "diff.pkl")
with open(useful_columns_path, 'r') as f:
    useful_columns = set(json.load(f))
diff = pd.read_pickle(output_dir / "diff.pkl")

# %%
diff_df = pd.DataFrame(diff, columns=["diff"])
diff_df["metric_kind"] = diff_df.index.map(lambda _: _.split("__")[0].split("##")[1])
diff_df

# %%
import numpy as np
rec_loss_df["rel_loss"] = rec_loss_df.apply(lambda _: _["loss"] / np.maximum(np.mean(np.square(_["x"])), 1e-12), axis=1)
rec_loss_df["metric_kind"] = rec_loss_df["metric"].map(lambda _: _.split("##")[1])
rec_loss_df

# %%
metric_kind_rec_loss_df = rec_loss_df.groupby("metric_kind")[["rel_loss"]].median()
metric_kind_rec_loss_df

# %%
useful_columns

# %% tags=[]
import numpy as np
from loguru import logger
import torch as th

######################################
mode = 'simple_fctype'
######################################
dataloader = model.train_dataloader_orig()
encoder = model.module.feature_projector
logger.info("Get reconstructed X")
feat, *_ = tuple(zip(*dataloader))
feat = [th.cat(_, dim=0) for _ in list(zip(*feat))]
device = next(encoder.parameters()).device
feat = [_.to(device) for _ in feat]
decoder.to(device)
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
                    "x": _feat,
                    "fid": fault_id,
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

# %% tags=[]
import pickle
with open(output_dir / "orig_ts_feats.pkl", "wb+") as f:
    pickle.dump(orig_ts_feats, f)
with open(output_dir / "rec_ts_feats.pkl", "wb+") as f:
    pickle.dump(rec_ts_feats, f)
with open(output_dir / "rec_loss_df.pkl", "wb+") as f:
    pickle.dump(rec_loss_df, f)

# %%
import pickle
with open(output_dir / "orig_ts_feats.pkl", "rb") as f:
    orig_ts_feats = pickle.load(f)
with open(output_dir / "rec_ts_feats.pkl", "rb") as f:
    rec_ts_feats = pickle.load(f)
with open(output_dir / "rec_loss_df.pkl", "rb") as f:
    rec_loss_df = pickle.load(f)

# %%
rec_loss_df["metric_kind"] = rec_loss_df.metric.map(lambda _: _.split("##")[1])
grouped_rec_loss_df = rec_loss_df.groupby("metric_kind")[["loss"]].mean().sort_values(by="loss")
grouped_rec_loss_df

# %%
import numpy as np
diff = 2 * np.mean(
    np.square(rec_ts_feats.values - orig_ts_feats), axis=0
) / np.mean(
    np.square(rec_ts_feats.values + orig_ts_feats), axis=0
)
diff

# %%
diff_df = pd.DataFrame(diff, columns=["feat_diff"])
diff_df["metric_kind"] = diff_df.index.map(lambda _: _.split("__")[0].split("##")[1])
diff_df["feature_name"] = diff_df.index.map(lambda _: _.split("##")[1])
grouped_diff_df = diff_df.groupby(["feature_name", "metric_kind"])[["feat_diff"]].mean().reset_index()
grouped_diff_df

# %%
df = grouped_diff_df.join(grouped_rec_loss_df, on="metric_kind")
df["score"] = df.eval("feat_diff * loss")
df

# %%
useful_columns = set(df[(df.loss > 0.5) & (df.feat_diff < 0.5)]["feature_name"].values)

# %% tags=[]
from DejaVu.explanability import dt_follow
from DejaVu.dataset import prepare_sklearn_dataset
config.flush_dataset_cache=False
dt_follow(
    cdp, output_dir / "with_feature_selection", prepare_sklearn_dataset(
        cdp, config, cache, mode='simple_fctype'
    ), useful_columns, y_probs, prune=False,
)
dt_follow(
    cdp, output_dir / "without_feature_selection", prepare_sklearn_dataset(
        cdp, config, cache, mode='simple_fctype'
    ), None, y_probs, prune=False,
)

# %% [markdown]
# ## LR explain

# %%
lr_output_dir = (output_dir / ".." / "lr_explain").resolve()
lr_output_dir.mkdir(exist_ok=True, parents=True)
print(f"{lr_output_dir=}")

# %%
from DejaVu.explanability.dt_follow import lr_follow
from DejaVu.dataset import prepare_sklearn_dataset
config.flush_dataset_cache=False
feature_importance_dict = lr_follow(
    cdp, lr_output_dir, prepare_sklearn_dataset(
        cdp, config, cache, mode='simple_fctype'
    ), useful_columns, y_probs,
)

# %%
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import json
def plot_lr_explain_result(feature_importances, k=5):
    def parse(s):
        metric = s.split("__")[0]
        func = s.split("__")[1]
        params = {}
        for item in s.split("__")[2:]:
            k, v = item.split("_")
            if k == "min" and float(v) < -1e9:
                v = "-inf"
            if k == "max" and float(v) > 1e9:
                v = "inf"
            params[k] = v
        return f"{metric}: {func}({', '.join([f'{k}={v}' for k, v in params.items()])})"
    data = pd.DataFrame({"importance": pd.Series(feature_importances)}).sort_values(by="importance", ascending=False)[:k].reset_index()
    data["index"] = data["index"].map(parse)
    fig = plt.figure(dpi=300, figsize=(6, 1.2))
    sns.barplot(data=data, x="importance", y="index", palette="Blues_r")
    plt.ylabel(None)
    plt.xlabel("Feature Importance in Logistic Regression")

for failure_class in fdg.failure_classes:
    try:
        with open(lr_output_dir / f"{failure_class}_feature_importance.json", "r") as f:
            feature_importances = json.load(f)
    except FileNotFoundError:
        continue
    fig = plot_lr_explain_result(feature_importances)
    plt.savefig(lr_output_dir / f"{failure_class.replace(' ', '_')}.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

# %%
