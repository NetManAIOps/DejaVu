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
sys.path.insert(0, "/SSF")
from DejaVu.explib.read_model import read_model

# %% tags=[]
from pathlib import Path
exp_dir = Path('/data/SSF/experiment_outputs/run_GAT_node_classification.py.2021-12-22T03:50:07.949451')

from DejaVu.models import get_GAT_model
model, y_probs, y_preds = read_model(
    exp_dir,
    get_GAT_model,
    override_config=dict(data_dir=Path("/SSF/data/A1"), flush_dataset_cache=False)
)

cdp = fdg = model.fdg
config = model.config
cache = model.cache

output_dir = config.output_dir / 'LIME_explain'
output_dir.mkdir(exist_ok=True, parents=True)
print(output_dir)

# %%
from DejaVu.models import DejaVuModelInterface
from typing import Callable
from einops import rearrange, repeat
import torch as th
import numpy as np
def wrap_predictor_for_DejaVu_model(model: DejaVuModelInterface, failure_instance: str, failure_id: int, metric_local_id=None) -> Callable:
    if failure_id in model.train_failure_ids:
        dataset = model.train_dataset
    elif failure_id in model.validation_failure_ids:
        dataset = model.validation_dataset
    else:
        dataset = model.test_dataset
    features, _, _, g = dataset[dataset.fault_ids.tolist().index(failure_id)]
    global_id = model.fdg.instance_to_gid(failure_instance)
    failure_class, local_id = model.fdg.instance_to_local_id(failure_instance)
    failure_class_id = model.fdg.failure_classes.index(failure_class)
    
    def func(ts: np.ndarray):
        device = model.device
        model.to(device)
        new_features = []
        for fc_id in range(len(model.fdg.failure_classes)):
            feat = repeat(features[fc_id], "i m t -> b i m t", b=ts.shape[0]).to(device=device, dtype=th.float32)
            if fc_id == failure_class_id:
                if metric_local_id is None:
                    feat[:, local_id, :, :] = th.from_numpy(ts).to(device=device, dtype=th.float32)
                else:
                    feat[:, local_id, metric_local_id, :] = th.from_numpy(ts).to(device=device, dtype=th.float32)
            # print(feat.shape)
            new_features.append(feat)
        probs = th.sigmoid(model(new_features, [g]).detach()).numpy()[:, global_id]
        return np.vstack([1-probs, probs]).T
    return func


# %%
from DejaVu.explanability.lime_timeseries import LimeTimeSeriesExplainer
import pandas as pd
def get_explain(failure_id: int):
    rc_instance = model.fdg.root_cause_instances_of(failure_id)[0]
    rc_class = model.fdg.instance_to_class(rc_instance)
    _, local_id = model.fdg.instance_to_local_id(rc_instance)
    rc_class_id = model.fdg.failure_classes.index(rc_class)
    print(f"{failure_id=} {rc_instance=}")
    explainer = LimeTimeSeriesExplainer(class_names=['non-root-cause', 'root-cause'], signal_names=model.fdg.FC_metrics_dict[rc_class])
    num_features = 500  # how many feature contained in explanation
    num_slices = 1  # split time series
    if failure_id in model.train_failure_ids:
        dataset = model.train_dataset
    elif failure_id in model.validation_failure_ids:
        dataset = model.validation_dataset
    else:
        dataset = model.test_dataset
    ts = dataset[dataset.fault_ids.tolist().index(failure_id)][0][rc_class_id][local_id].numpy()
    exp = explainer.explain_instance(
        ts, wrap_predictor_for_DejaVu_model(model, rc_instance, failure_id), 
        num_features=num_features, num_samples=1000, num_slices=num_slices, 
        replacement_method='total_mean'
    )
    df = pd.DataFrame(exp.as_list(), columns=["feature", "importance"])
    df['feature'] = df['feature'].map(lambda _: _.split(" - ")[1])
    df = df.groupby(['feature']).sum().reset_index()
    df = df.sort_values(by="importance", ascending=False)
    return df

get_explain(73)

# %%
get_explain(14)

# %%
get_explain(20)

# %% tags=[]
from tqdm import tqdm
exp_df_list = []
for fid in tqdm([_ for _ in model.fdg.failure_ids if model.fdg.instance_to_class(model.fdg.root_cause_instances_of(_)[0]) == "OS Network"]):
    __df = get_explain(fid)
    __df["fid"] = fid
    exp_df_list.append(__df)
exp_df = pd.concat(exp_df_list)
exp_df

# %%
import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(dpi=300, figsize=(5, 1.2))
order = exp_df.groupby(["feature"])["importance"].mean().sort_values(ascending=False).index.tolist()[:5]
sns.barplot(data=exp_df[exp_df.feature.isin(order)], y="feature", x="importance", order=order, palette="Blues_r")
plt.ylabel(None)
plt.xlabel("Metric Importance by LIME")
plt.savefig(output_dir / "OS_Network.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(1, 2, dpi=300, figsize=(8, 1.4))
sns.barplot(data=get_explain(14).iloc[:5], y="feature", x="importance", palette="Blues_r", ax=axes[0])
axes[0].set_ylabel(None)
axes[0].set_xlabel("Metric Importances of $T_1$                   ")
# axes[0].yaxis.set_tick_params(labelsize=9)
# axes[0].xlabel(-0.0007, 5.5, "Feature Importance")
axes[0].ticklabel_format(style="sci", axis="x", scilimits=(-1, 1))

sns.barplot(data=get_explain(20).iloc[:5], y="feature", x="importance", palette="Blues_r", ax=axes[1])
axes[1].set_ylabel(None)
axes[1].set_xlabel("Metric Importances of $T_2$                   ")
axes[1].ticklabel_format(style="sci", axis="x", scilimits=(-1, 1))
# axes[0].text(-0.0008, 5.5, "Feature Importance by LIME")
# axes[0].set_xlabel("$T_1$")
# plt.ylabel(None)
# plt.xlabel("Feature Importance by LIME")
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig("/SSF/output/lime_local_interpretation_example.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()

# %%
