from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional, Union, Tuple

import numpy as np
from matplotlib import pyplot as plt
from pytz import timezone

from explib.legend import get_line_style
from failure_dependency_graph import FDG
from metric_preprocess import MetricPreprocessor


def parse_ts(__ts):
    return datetime.fromtimestamp(__ts).astimezone(timezone('Asia/Shanghai'))


def is_abnormal(his, cur) -> bool:
    median = np.median(his)
    mad = np.maximum(np.median(np.abs(his - median)), 1e-3)
    score = np.mean((cur[:5] - median) / mad)
    return score > 3 * 0.4


def plot_fault_metrics(
        fault, cdp: FDG, fe: MetricPreprocessor, log_scale=False, plot_type=True, extra_metric_names=None,
        extra_node_names=None,
        skip_fault_rc=False, window_size=(10, 10), skip_normal=False, format_xdate=False,
        output_dir: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 4),
        show_xticks: bool = True,
        show_yticks: bool = True,
        mark_other_failures: bool = True,
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
        node_list = sum([cdp.failure_instances[node_type] for node_type in node_types], [])
    else:
        node_list = rc_nodes

    def __add_metric(__metric_name, __skip_normal=False):
        for _k, _v in cdp.FI_metrics_dict.items():
            if __metric_name in _v:
                __node = _k
                break
        else:
            print(f"{__metric_name} not found")
            return
        __gid = cdp.instance_to_gid(__node)
        __node_type, _typed_idx = cdp.instance_to_local_id(__node)
        __y = fe(ts, window_size)[cdp.failure_classes.index(__node_type)][
            _typed_idx, cdp.FI_metrics_dict[__node].index(__metric_name)].cpu().numpy()
        __x = list(map(x_convert, range(ts - 60 * window_size[0], ts + 60 * window_size[1], 60)))
        if __skip_normal and not is_abnormal(__y[:window_size[0]], __y[-window_size[1]:]):
            return
        if log_scale:
            __y = np.log(1 + np.abs(__y)) / np.log(10) * np.sign(__y)
        plt.plot(
            __x, __y, label=__metric_name,
            **get_line_style(__metric_name.split('##')[1])
        )

    if not skip_fault_rc:
        for node in node_list:
            for metric_name in cdp.FI_metrics_dict[node]:
                __add_metric(metric_name, skip_normal)

    if extra_metric_names is not None:
        for metric_name in extra_metric_names:
            __add_metric(metric_name)
    if extra_node_names is not None:
        for n in extra_node_names:
            for metric_name in cdp.FI_metrics_dict[n]:
                __add_metric(metric_name, skip_normal)
    plt.title(f"{rc_nodes}" if not log_scale else f"log({rc_nodes})")
    plt.axvline(x_convert(ts), color='red', linestyle='--', alpha=0.8, label='Fault Occurs')
    plt.axvline(x_convert(ts + 10 * 60), color='red', linestyle='-.', alpha=0.8, label='10min After Fault')
    if mark_other_failures:
        all_failure_ts_list = [
            _ for _ in cdp.failures_df['timestamp'] if ts - 60 * window_size[0] <= _ <= ts + 60 * window_size[1]
        ]
        for other_ts in all_failure_ts_list:
            plt.axvline(x_convert(other_ts), color='red', linestyle='--', alpha=0.4)

    #     plt.axvline(parse_ts(ts + 5 * 60), color='red', linestyle='--', alpha=0.8)
    plt.legend(fontsize='small', ncol=2)
    if format_xdate:
        fig.autofmt_xdate()
    if not show_xticks:
        plt.xticks([], [])
    if not show_yticks:
        plt.yticks([], [])
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(
            output_dir / f"name={fault.name}.ts={fault['timestamp']}.rc={fault['root_cause_node'].replace(';', '-')}.pdf",
            bbox_inches='tight',
            pad_inches=0
        )
    # plt.show()
    plt.close()
    return fig
