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

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

import sys
import os
sys.path.insert(0, '/SSF')
from DejaVu.explib import get_eval_results
import pandas as pd
pd.options.display.max_rows = 100
pd.options.display.min_rows = 100
eval_rets, melted = get_eval_results()
eval_rets

# %%
import re

REGEX = r'^GRU\+GAT-H(?P<H>\d+)-L(?P<L>\d+)\+BAL$'
def __match_GAT(name: str):
    if __match := re.match(REGEX, name):
        __gd = dict(__match.groupdict())
        # print(__gd)
        return int(__gd.get('H', 4)), int(__gd.get('L', 8))
    else:
        return 0, 0
    
def match(the_df):
    the_df = the_df.copy()
    the_df['H'] = the_df.Method.map(
        lambda name: __match_GAT(name)[0]
    )
    the_df['L'] = the_df.Method.map(
        lambda name: __match_GAT(name)[1]
    )
    for m in ['MAR', 'A@1', 'A@2', 'A@3', 'A@5']:
        the_df[m] = pd.to_numeric(the_df[m], errors='coerce')
    return the_df


# %%
df1 = match(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(REGEX, _) is not None)
) & (
    eval_rets.Dataset == 'AIOPS20-phase1'
)])
df1 = df1.groupby(['H', 'L', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean()
df2 = match(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(REGEX, _) is not None)
) & (
    eval_rets.Dataset == 'AIOPS20-phase2'
)])
df2 = df2.groupby(['H', 'L', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean()
dfa = (df1 * 32 + df2 * 46) / (32 + 46)
dfa = dfa.reset_index()
# dfa = pd.concat([df1, df2])
display(dfa)

# %%
df = match(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(REGEX, _) is not None)
) & (
    eval_rets.Dataset == 'AIOPS21-B'
)])

dfb = df.groupby(['H', 'L', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean().reset_index()
# dfb = df
dfb

# %%
df = match(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(REGEX, _) is not None)
) & (
    eval_rets.Dataset == 'CCB-Oracle'
)])

dfc = df.groupby(['H', 'L', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean().reset_index()
dfc

# %%
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
def plot_impact_of_head_and_layer(data):
    data = data.copy()
    fig = plt.figure(figsize=(0.5 * len(data['L'].unique()) + 0.5, 0.5 * len(data['H'].unique())), dpi=300)
    ax = plt.axes()
#     df['MAR'] = df['MAR'].map(np.log)
    sns.heatmap(data=data.pivot(index='H', columns='L', values='MAR'), cmap='Blues_r', vmin=1, vmax=data.MAR.max() * 1.5,)
    plt.xlabel("$L$")
    plt.ylabel("$H$")
    plt.close(fig)
    return fig


# %%
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
fig, axes = plt.subplots(1, 2, figsize=(6, 1.3), dpi=300, sharey=False)
sns.lineplot(data=dfa, x="L", hue="H", y="MAR", palette=['#a1dab4','#41b6c4','#2c7fb8','#253494'], ax=axes[0], marker=".")
axes[0].set_ylim([0.95, 3.1])
axes[0].text(-1, 0.22, "$L$")
axes[0].set_xlabel("(a) $\mathcal{A}$")
axes[0].legend_.set_visible(False)
axes[0].set_xticks([1, 2, 4, 8, 16])

sns.lineplot(data=dfb, x="L", hue="H", y="MAR", palette=['#a1dab4','#41b6c4','#2c7fb8','#253494'], ax=axes[1], marker=".")
axes[1].set_ylim([4.5, 7.2])
axes[1].set_xlabel("(b) $\mathcal{B}$")
axes[1].legend_.set_visible(False)
axes[1].set_xticks([1, 2, 4, 8, 16])

fig.legend(*axes[1].get_legend_handles_labels(), bbox_to_anchor=(1.08, 0.95), ncol=1, fontsize="small", title="$H$")

plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
fig.savefig('/SSF/output/impact_of_gat_architecture/impact_GAT.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close(fig)

# %%
fig = plot_impact_of_head_and_layer(dfa)
fig.savefig('/SSF/output/impact_of_gat_architecture/A.pdf', bbox_inches='tight', pad_inches=0)
display(fig)

# %%
fig = plot_impact_of_head_and_layer(dfb)
fig.savefig('/SSF/output/impact_of_gat_architecture/B.pdf', bbox_inches='tight', pad_inches=0)
display(fig)

# %%
fig = plot_impact_of_head_and_layer(dfc)
fig.savefig('/SSF/output/impact_of_gat_architecture/C.pdf', bbox_inches='tight', pad_inches=0)
display(fig)

# %%
