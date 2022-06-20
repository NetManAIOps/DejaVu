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

REGEX = r'^(TSS(?P<tss>[\d.]+)\+)?GRU\+GAT-H4-L8?\+BAL$'
def __match_tss(name: str):
    if __match := re.match(REGEX, name):
        __gd = dict(__match.groupdict())
        return float(__gd['tss']) if 'tss' in __gd and __gd['tss'] is not None else 1.
    else:
        return 0
    
def match(the_df):
    the_df = the_df.copy()
    the_df['tss'] = the_df.Method.map(
        lambda name: __match_tss(name)
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
df1 = df1.groupby(['tss', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean()
df2 = match(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(REGEX, _) is not None)
) & (
    eval_rets.Dataset == 'AIOPS20-phase2'
)])

df2 = df2.groupby(['tss', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean()
dfa = (df1 * 32 + df2 * 46) / (32 + 46)
dfa = dfa.reset_index()
display(dfa)

# %%
df = match(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(REGEX, _) is not None)
) & (
    eval_rets.Dataset == 'AIOPS21-B'
)])

dfb = df.groupby(['tss', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean().reset_index()
display(dfb)

# %%
df = match(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(REGEX, _) is not None)
) & (
    eval_rets.Dataset == 'CCB-Oracle'
)])

dfc = df.groupby(['tss', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean().reset_index()
display(dfc)

# %%
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from explib import get_line_style, legend_db
import matplotlib.ticker as mticker

def plot_impact_of_tss(data, ax, fig, legend=False):
    data = data.copy()
    # data = data[data.Z.isin({2, 3, 4, 8, 16, 32, 64})]
    data = data.sort_values('tss')
    p1, = ax.plot(data.tss, data.MAR, marker='+', ls='-', label='MAR', c=legend_db.colors[0])
    # ax.set_ylim([data['MAR'].mean() / 2, data['MAR'].mean() * 1.5])
    ax.set_ylabel('MAR')
#     ax.legend(bbox_to_anchor=(1.1, 1))
    ax.set_xscale("log")
    # ax.set_xticks([2, 3, 4, 8, 16, 32, 64], minor=False)
    # ax.set_xticklabels(["2", "3", "4", "8", "16", "32", "64"], minor=False)
    
    ax2 = ax.twinx()
    p2, = ax2.plot(data.tss, data['A@1'], marker='.', ls='--', label='A@1', c=legend_db.colors[1])
    p3, = ax2.plot(data.tss, data['A@2'], marker='*', ls='-', label='A@2', c=legend_db.colors[2])
    p4, = ax2.plot(data.tss, data['A@3'], marker='+', ls='--', label='A@3', c=legend_db.colors[3])
    p5, = ax2.plot(data.tss, data['A@5'], marker='.', ls='-', label='A@5', c=legend_db.colors[4])
    ax2.set_ylabel('A@k')
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

fig, axes = plt.subplots(1, 3, figsize=(7, 1.2), dpi=300, sharey=False)
ax1, ax2 = plot_impact_of_tss(dfa, axes[0], fig, legend=True)
# ax2.get_yaxis().set_visible(False)
ax3, ax4 = plot_impact_of_tss(dfb, axes[1], fig)
# ax4.get_yaxis().set_visible(False)
ax5, ax6 = plot_impact_of_tss(dfc, axes[2], fig)
# ax2.get_shared_y_axes().join(ax2, ax4, ax6)
ax1.set_xlabel("(a) $\mathcal{A}$")
ax3.set_xlabel("(b) $\mathcal{B}$")
ax5.set_xlabel("(c) $\mathcal{C}$")
plt.tight_layout(pad=0.1)
plt.savefig("/SSF/output/impact_of_tss/impact_of_tss.pdf", bbox_inches='tight', pad_inches=0)
plt.show(fig)
plt.close(fig)

# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 1.5), dpi=300, sharey=False)
ax1, ax2 = plot_impact_of_tss(dfa, axes[0], fig, legend=True)
# ax2.get_yaxis().set_visible(False)
ax3, ax4 = plot_impact_of_tss(dfb, axes[1], fig)
# ax4.get_yaxis().set_visible(False)
ax1.set_xlabel("(a) $\mathcal{A}$")
ax3.set_xlabel("(b) $\mathcal{B}$")
plt.tight_layout()
plt.savefig("/SSF/output/impact_of_tss/impact_of_tss.pdf", bbox_inches='tight', pad_inches=0)
plt.show(fig)
plt.close(fig)

# %%
import re
def __match_tss(name: str):
    if __match := re.match(r'^GAT-H4-L8(\+tss(?P<tss>[0-9]+\.?[0-9]*))?\+AUG\+BAL$', name):
        __gd = dict(__match.groupdict())
        # print(__gd)
        return 1.0 if __match['tss'] is None else float(__match['tss'])
    else:
        return 0

def match_tss(the_df):
    the_df = the_df.copy()
    the_df['tss'] = the_df.Method.map(
        lambda name: __match_tss(name)
    )
    return the_df

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from explib import get_line_style, legend_db
import matplotlib.ticker as mticker

def plot_impact_of_tss(data):
    data = data.copy()
    data = data.sort_values('tss')
    fig = plt.figure(figsize=(4.5, 1.1), dpi=300)
    ax = plt.axes()
    p1, = ax.plot(data.tss, data.MAR, marker='+', ls='-', label='MAR', c=legend_db.colors[0])
    
    ax = ax.twinx()
    p2, = ax.plot(data.tss, data['A@1'], marker='.', ls='--', label='A@1', c=legend_db.colors[1])
    p3, = ax.plot(data.tss, data['A@2'], marker='*', ls='-', label='A@2', c=legend_db.colors[2])
    p4, = ax.plot(data.tss, data['A@3'], marker='+', ls='--', label='A@3', c=legend_db.colors[3])
    p5, = ax.plot(data.tss, data['A@5'], marker='.', ls='-', label='A@5', c=legend_db.colors[4])
    
    ax.legend(handles=[p1, p2, p3, p4, p5], bbox_to_anchor=(-0.05, 1), loc='lower left', prop={'size': 8}, ncol=5)
    
    plt.close(fig)
    return fig


# %%
df1 = match_tss(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(r'^GAT-H4-L8(\+tss(?P<tss>[0-9.]+))?\+AUG\+BAL$', _) is not None)
) & (
    eval_rets.Dataset == 'AIOPS20-phase1'
)])

df1 = df1.groupby(['tss', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean()
df2 = match_tss(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(r'^GAT-H4-L8(\+tss(?P<tss>[0-9.]+))?\+AUG\+BAL$', _) is not None)
) & (
    eval_rets.Dataset == 'AIOPS20-phase2'
)])

df2 = df2.groupby(['tss', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean()
df = (df1 * 32 + df2 * 46) / (32 + 46)
df = df.reset_index()
display(df)
fig = plot_impact_of_tss(df)
fig.savefig('../output/impact_of_tss/A.pdf', bbox_inches='tight', pad_inches=0)
display(fig)

# %%
df = match_tss(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(r'^GAT-H4-L8(\+tss(?P<tss>[0-9.]+))?\+AUG\+BAL$', _) is not None)
) & (
    eval_rets.Dataset == 'AIOPS21-B'
)])

df = df.groupby(['tss', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean().reset_index()
display(df)
fig = plot_impact_of_tss(df)
fig.savefig('../output/impact_of_tss/B.pdf', bbox_inches='tight', pad_inches=0)
display(fig)

# %%
df = match_tss(eval_rets[(
      eval_rets.Method.map(lambda _: re.match(r'^GAT-H4-L8(\+tss(?P<tss>[0-9.]+))?\+AUG\+BAL$', _) is not None)
) & (
    eval_rets.Dataset == 'CCB-Oracle'
)])

df = df.groupby(['tss', 'Method'])[['MAR', 'A@1', 'A@2', 'A@3', 'A@5']].mean().reset_index()
display(df)
fig = plot_impact_of_tss(df)
fig.savefig('../output/impact_of_tss/C.pdf', bbox_inches='tight', pad_inches=0)
display(fig)

# %%
