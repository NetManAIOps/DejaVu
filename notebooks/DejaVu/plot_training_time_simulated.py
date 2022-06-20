# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload 
# %autoreload 2
import sys
import pandas as pd
pd.options.display.max_rows=100
pd.options.display.min_rows=100
pd.options.display.max_colwidth=120
sys.path.insert(0, '/SSF')
import os
os.chdir('/SSF')

import io
from typing import Tuple

import pandas as pd
import requests

s = requests.get(
    "https://docs.google.com/spreadsheets/d/1zZZfsz7kz1VvzFQhO-gI5TLe6Rc70EooURBai_z8YT8/export?format=csv&id=1zZZfsz7kz1VvzFQhO-gI5TLe6Rc70EooURBai_z8YT8&gid=1195533860",
    proxies={
        "http": "http://cpu3:7890",
        "https": "http://cpu3:7890",
    },
    allow_redirects=True,
).content
print("download finished")
eval_rets = pd.read_csv(io.StringIO(s.decode('utf-8')))
eval_rets.rename(columns=lambda x: x.rstrip(" "), inplace=True)
eval_rets

# %%
eval_rets['gpu'] = eval_rets['gpu'].map(lambda x: {True: 'w/ GPU', False: 'w/o GPU'}[x])

# %%
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(7, 1.2), dpi=300, sharey=False)


sns.lineplot(
    data=eval_rets.loc[
        (eval_rets['n_metrics'] == 2048) & 
        (eval_rets['n_failures'] == 100),
    ],
    x='n_instances',
    y='epoch_time',
    hue='gpu',
    ax=axes[0],
    palette="Blues",
)
axes[0].legend(title="", fontsize='small')
axes[0].set_ylabel("Epoch Time (s)    ")
axes[0].set_xlabel("#Failure Instances")

sns.lineplot(
    data=eval_rets.loc[
        (eval_rets['n_instances'] == 100) & 
        (eval_rets['n_failures'] == 100),
    ],
    x='n_metrics',
    y='epoch_time',
    hue='gpu',
    ax=axes[1],
    palette="Blues",
)
axes[1].legend_.set_title("")
axes[1].set_ylabel(None)
axes[1].set_xlabel("#Metrics")
axes[1].legend_.set_visible(False)

sns.lineplot(
    data=eval_rets.loc[
        (eval_rets['n_metrics'] == 2048) & 
        (eval_rets['n_instances'] == 100),
    ],
    x='n_failures',
    y='epoch_time',
    hue='gpu',
    ax=axes[2],
    palette="Blues",
)
axes[2].legend_.set_visible(False)
axes[2].legend_.set_title("")
axes[2].set_ylabel(None)
axes[2].set_xlabel("#Failures")

plt.tight_layout(pad=0.2)
plt.savefig(
    "/SSF/output/plot_training_time_simulated/plot_training_time_simulated.pdf", 
    bbox_inches='tight', pad_inches=0
)
plt.show()
plt.close()

# %%
eval_rets.loc[
        (eval_rets['n_metrics'] == 10) & 
        (eval_rets['n_instances'] == 10),
    ]

# %%
