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

REGEX = r'^GRU\+GAT-H4-L8(\+z(?P<Z>\d+))?\+BAL$'
def __match_z_dim(name: str):
    if __match := re.match(REGEX, name):
        __gd = dict(__match.groupdict())
        # print(__gd)
        return 3 if __match['Z'] is None else int(__match['Z'])
    else:
        return 0
    
def match_z_dim(the_df):
    the_df = the_df.copy()
    the_df['Z'] = the_df.Method.map(
        lambda name: __match_z_dim(name)
    )
    for m in ['MAR', 'A@1', 'A@2', 'A@3', 'A@5']:
        the_df[m] = pd.to_numeric(the_df[m], errors='coerce')
    return the_df


# %%
df1 = match_z_dim(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(REGEX, _) is not None)
) & (
    eval_rets.Dataset == 'AIOPS20-phase1'
)])
df1 = df1.groupby(['Z', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean()
df2 = match_z_dim(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(REGEX, _) is not None)
) & (
    eval_rets.Dataset == 'AIOPS20-phase2'
)])

df2 = df2.groupby(['Z', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean()
dfa = (df1 * 32 + df2 * 46) / (32 + 46)
dfa = dfa.reset_index()
display(dfa)

# %%
df = match_z_dim(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(REGEX, _) is not None)
) & (
    eval_rets.Dataset == 'AIOPS21-B'
)])

dfb = df.groupby(['Z', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean().reset_index()

# %%
df = match_z_dim(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(REGEX, _) is not None)
) & (
    eval_rets.Dataset == 'CCB-Oracle'
)])

dfc = df.groupby(['Z', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean().reset_index()

# %%
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from explib import get_line_style, legend_db
import matplotlib.ticker as mticker
from math import ceil

def plot_impact_of_z(data, ax, fig, legend=False):
    data = data.copy()
    data = data[data.Z.isin({2, 3, 4, 8, 16, 32, 64})]
    data = data.sort_values('Z')
    p1, = ax.plot(data.Z, data.MAR, marker='+', ls='-', label='MAR', c=legend_db.colors[0])
    ylim = [int(data['MAR'].mean() / 2), int(ceil(data['MAR'].mean() * 1.5))]
    ax.set_ylim(ylim)
#     ax.set_ylabel('MAR')
#     ax.legend(bbox_to_anchor=(1.1, 1))
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 4, 8, 16, 32, 64], minor=False)
    ax.set_xticklabels(["2", "3", "4", "8", "16", "32", "64"], minor=False)
    ax.set_ylabel("MAR")
    ax.set_yticks(range(ylim[0] + 1, ylim[1], (ylim[1] - ylim[0]) // 4 + 1))
    ax.text(x=1.2, y=ylim[0] - (ylim[1] - ylim[0]) / 5, s="Z")
    
    ax2 = ax.twinx()
    p2, = ax2.plot(data.Z, data['A@1'], marker='.', ls='--', label='A@1', c=legend_db.colors[1])
    p3, = ax2.plot(data.Z, data['A@2'], marker='*', ls='-', label='A@2', c=legend_db.colors[2])
    p4, = ax2.plot(data.Z, data['A@3'], marker='+', ls='--', label='A@3', c=legend_db.colors[3])
    p5, = ax2.plot(data.Z, data['A@5'], marker='.', ls='-', label='A@5', c=legend_db.colors[4])
    ax2.set_ylabel("A@k")
#     ax.set_xticks([2, 3, 4, 8, 16, 32, 64], minor=False)
#     ax.set_xticklabels(["2", "3", "4", "8", "16", "32", "64"], minor=False)
#     ax.xaxis.set_major_locator(mticker.FixedLocator([2, 3, 4, 8, 16, 32, 64]))
    if legend:
        fig.legend(handles=[p1, p2, p3, p4, p5], bbox_to_anchor=(0.25, 0.88), loc='lower left', prop={'size': 8}, ncol=5)
    
#     fig.legend(loc="upper right", bbox_to_anchor=(1, 0.6), bbox_transform=ax.transAxes, ncol=3, fontsize='small')
#     plt.xticks([2, 3, 4, 5, 6], [2, 3, 4, 5, 6])
#     plt.xlim([2, 65])
#     plt.xscale("log")
#     plt.show()
    # plt.close(fig)
    return ax, ax2

fig, axes = plt.subplots(1, 3, figsize=(7, 1.4), dpi=300, sharey=False)
ax1, ax2 = plot_impact_of_z(dfa, axes[0], fig, legend=True)
# ax2.get_yaxis().set_visible(False)
ax3, ax4 = plot_impact_of_z(dfb, axes[1], fig)
# ax4.get_yaxis().set_visible(False)
ax5, ax6 = plot_impact_of_z(dfc, axes[2], fig)
# ax2.get_shared_y_axes().join(ax2, ax4, ax6)
ax1.set_xlabel("(a) $\mathcal{A}$")
ax3.set_xlabel("(b) $\mathcal{B}$")
ax5.set_xlabel("(c) $\mathcal{C}$")
plt.tight_layout()
plt.savefig("/SSF/output/impact_of_z_dim/impact_of_z_dim.pdf", bbox_inches='tight', pad_inches=0)
plt.show(fig)
plt.close(fig)

# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 1.5), dpi=300, sharey=False)
ax1, ax2 = plot_impact_of_z(dfa, axes[0], fig, legend=True)
ax3, ax4 = plot_impact_of_z(dfb, axes[1], fig)
ax1.set_xlabel("(a) $\mathcal{A}$")
ax3.set_xlabel("(b) $\mathcal{B}$")
plt.tight_layout()
plt.savefig("/SSF/output/impact_of_z_dim/impact_of_z_dim.pdf", bbox_inches='tight', pad_inches=0)
plt.show(fig)
plt.close(fig)

# %%
