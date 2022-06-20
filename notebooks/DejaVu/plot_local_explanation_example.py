# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Load data pack and model

# %% tags=[]
# %load_ext autoreload
# %autoreload 2
import os
# Read Model
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.realpath(".."))
from explib import read_model

   
exp_dir = Path('/data/DejaVu/experiments_output/run_GAT_node_classification.py.2021-08-17T03:18:26.426390/')
cdp, config, cache, model, y_probs, y_preds, [
    train_dataloader, validation_dataloader, test_dataloader,
] = read_model(
    exp_dir
)
output_dir = "/DejaVu/output/plot_local_explanation_example"
output_dir.mkdir(exist_ok=True, parents=True)


from explib import plot_fault_metrics
fe=cache.get('FeatureExtractor')

# %%
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional, Union, Tuple

import numpy as np
from matplotlib import pyplot as plt
from pytz import timezone
import matplotlib.dates as mdates
from DejaVu.data import CFLDataPack, FeatureExtractor
from explib.legend import get_line_style


def parse_ts(__ts):
    return datetime.fromtimestamp(__ts).astimezone(timezone('Asia/Shanghai'))


def is_abnormal(his, cur) -> bool:
    median = np.median(his)
    mad = np.maximum(np.median(np.abs(his - median)), 1e-3)
    score = np.mean((cur[:5] - median) / mad)
    return score > 3 * 0.4


def plot_fault_metrics(
        fault, cdp: CFLDataPack, fe: FeatureExtractor, log_scale=False, plot_type=True, extra_metric_names=None,
        extra_node_names=None,
        skip_fault_rc=False, window_size=(10, 10), skip_normal=False, format_xdate=False,
        output_dir: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 3),
        show_xticks: bool = True,
        show_yticks: bool = True,
):
    if format_xdate:
        x_convert = parse_ts
    else:
        x_convert = lambda _: _
    pprint(fault)
    rc_nodes = fault['root_cause_node'].split(';')
    ts = fault['timestamp']
    print(f'fault time: {parse_ts(ts)}')
    node_types = fault['node_type'].split(';')
    fig = plt.figure(dpi=300, figsize=figsize)
    if plot_type:
        node_list = sum([cdp.nodes[node_type] for node_type in node_types], [])
    else:
        node_list = rc_nodes

    def __add_metric(__metric_name, __skip_normal=False):
        for _k, _v in cdp.node_metrics_dict.items():
            if __metric_name in _v:
                __node = _k
                break
        else:
            print(f"{__metric_name} not found")
            return
        __gid = cdp.node_name_to_gid(__node)
        __node_type, _typed_idx = cdp.gid_to_node_type_and_id(__gid)
        __y = fe(ts, window_size)[cdp.node_types.index(__node_type)][
            _typed_idx, cdp.node_metrics_dict[__node].index(__metric_name)].numpy()
        __x = list(map(x_convert, range(ts - 60 * window_size[0], ts + 60 * window_size[1], 60)))
        if __skip_normal and not is_abnormal(__y[:window_size[0]], __y[-window_size[1]:]):
            return
        if log_scale:
            __y = np.log(1 + np.abs(__y)) / np.log(10) * np.sign(__y)
        plt.plot(
            __x, __y, label=__metric_name.split('##')[1],
            **get_line_style(__metric_name.split('##')[1])
        )

    if not skip_fault_rc:
        for node in node_list:
            for metric_name in cdp.node_metrics_dict[node]:
                __add_metric(metric_name, skip_normal)

    if extra_metric_names is not None:
        for metric_name in extra_metric_names:
            __add_metric(metric_name)
    if extra_node_names is not None:
        for n in extra_node_names:
            for metric_name in cdp.node_metrics_dict[n]:
                __add_metric(metric_name, skip_normal)
    plt.title(f"{rc_nodes[0]}" if not log_scale else f"log({rc_nodes})")
    plt.axvline(x_convert(ts), color='red', linestyle='--', alpha=0.8, label='Failure occurs')
    plt.axvline(x_convert(ts + 10 * 60), color='C0', linestyle='-.', alpha=0.8, label='10min later')
    #     plt.axvline(parse_ts(ts + 5 * 60), color='red', linestyle='--', alpha=0.8)
    plt.legend(fontsize='small', ncol=4, loc='upper left', bbox_to_anchor=(0, 0.95))
    if format_xdate:
        myFmt = mdates.DateFormatter('%H:%M')
        plt.gca().xaxis.set_major_formatter(myFmt)
    if not show_xticks:
        plt.xticks([], [])
    if not show_yticks:
        plt.yticks([], [])
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(
            output_dir / f"name={fault.name}.ts={fault['timestamp']}.rc={fault['root_cause_node'].replace(';', '-')}.pdf".replace(' ', '_'),
            bbox_inches='tight',
            pad_inches=0
        )
    # plt.show()
    plt.close()
    return fig



# %%
from explib import legend_db
legend_db.clear()
plot_fault_metrics(
    cdp.failures_df.iloc[14], cdp=cdp, fe=fe, plot_type=False, log_scale=False,
    skip_fault_rc=False,
    extra_metric_names=[
    ],
#     window_size=(20, 60),
    skip_normal=False,
    extra_node_names=[
    ],
    format_xdate=True,
    output_dir=output_dir,
    figsize=(8.5, 1.8),
#     show_xticks=False,
)
plot_fault_metrics(
    cdp.failures_df.iloc[20], cdp=cdp, fe=fe, plot_type=False, log_scale=False,
    skip_fault_rc=False,
    extra_metric_names=[
    ],
#     window_size=(20, 60),
    skip_normal=False,
    extra_node_names=[
    ],
    format_xdate=True,
    output_dir=output_dir,
    figsize=(8.5, 1.8),
#     show_xticks=False,
)

# %%
plot_fault_metrics(
    cdp.failures_df.iloc[50], cdp=cdp, fe=fe, plot_type=False, log_scale=False,
    skip_fault_rc=True,
    extra_metric_names=[
        '##AAS_TOTAL', '##每秒登录数', '##现有连接数量', "##每秒执行数", 
#         "##index_contention_sql:ASHSQL",
#         "##enq: TM - contention",
#         "##log file sync",
#         "##PGA_ALLOCATED",
#         "##enq: TX - index contention",
    ],
#     window_size=(20, 60),
    skip_normal=True,
    extra_node_names=[
#         'os_018 Network'
#         'Execution', 
        "Contention",
#         "gc",
#         "LogFile",
#         "ASM",
#         'DBFile',
#         'CPU',
#         'Memory',
#         "Disk",
#         "Parse",
    ],
    format_xdate=True,
    output_dir=output_dir,
)

# %%
plot_fault_metrics(
    cdp.failures_df.iloc[14], cdp=cdp, fe=fe, plot_type=False, log_scale=False,
    window_size=(60, 10),
    skip_fault_rc=True,
    extra_metric_names=[
        "os_022##count",
    ],
)


# %% [markdown]
# ## Metric Value Distribution

# %%
def plot_metric(ts, metric_name, metric_df, window_size):
    fig = plt.figure(dpi=300)
    ts_list = list(range(ts, ts + window_size * 60))
    the_df = metric_df[(metric_df.name == metric_name) & (metric_df.timestamp.isin(ts_list))].sort_values(by='timestamp')
    plt.plot(list(map(parse_ts, the_df.timestamp)), the_df['value'].values, label=metric_name, marker='+')
    for _, fault in cdp.failures_df.iterrows():
        if fault['root_cause_node'] != 'os_021 Network':
            continue
        plt.axvline(parse_ts(fault.timestamp), color='r')
    plt.legend()
    plt.title(metric_name)
    plt.show()
    plt.close()
    
plot_metric(ts=cdp._metrics_df.timestamp.min(), metric_name='os_021##Sent_queue', metric_df=unnormed_cdp._metrics_df, window_size=1440 * 60)


# %%
def plot_metric_value_dist(metric_name, metric_df, log_scale=False):
    import seaborn as sns
    fig = plt.figure(dpi=300)
    sns.kdeplot(unnormed_cdp._metrics_df[unnormed_cdp._metrics_df.name == metric_name]['value'], log_scale=(log_scale, False))
    plt.xlabel(f'{metric_name}' if not log_scale else f'log({metric_name})')
    plt.show()
    plt.close()

plot_metric_value_dist('os_022##Sent_queue', unnormed_cdp._metrics_df, log_scale=False)

# %%
cdp._node_metrics_dict['db_007 State']

# %%
unnormed_cdp._metrics_df[unnormed_cdp._metrics_df.name == 'os_020##Sent_queue']['value']

# %%
cdp.failures_df.root_cause_node.unique()

# %%
