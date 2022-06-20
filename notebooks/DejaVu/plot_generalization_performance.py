# -*- coding: utf-8 -*-
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
import pandas as pd
pd.options.display.max_rows=100
pd.options.display.min_rows=100
pd.options.display.max_colwidth=120
sys.path.insert(0, '/SSF')
import os
os.chdir('/SSF')

from DejaVu.explib import get_eval_results
eval_rets, _ = get_eval_results()

# %% tags=[]
from loguru import logger
from pathlib import Path
import pickle
import sys
from DejaVu.models import DejaVuModelInterface, get_GAT_model
from DejaVu.explib import read_model
from utils.load_model import best_checkpoint
import numpy as np

logger.remove()
logger.add(sys.stdout, level='ERROR')

def parse_pred_ret_by_recurring(exp_dir_lists, dataset):
    records = []
    exp_dir_lists = [Path(_) for _ in exp_dir_lists]
    exp_dir_lists = [_ if _.is_dir() else _.parent for _ in exp_dir_lists]
    
    model, _, _ = read_model(exp_dir_lists[0], get_GAT_model, override_config=dict(flush_dataset_cache=False))
    fdg = model.fdg
    module: DejaVuModelInterface = model.module
    trainer = model.trainer
    test_fault_ids = model.test_failure_ids
    train_fault_ids = model.train_failure_ids
    train_rc_nodes = set()
    for fault_id in train_fault_ids:
        train_rc_nodes |= set(fdg.root_cause_instances_of(fault_id))
    
    for exp_dir in exp_dir_lists:
        exp_records = []
        print(exp_dir)
        trainer.test(
            model, test_dataloaders=model.test_dataloader(), verbose=False,
            ckpt_path=str(best_checkpoint(exp_dir, debug=True)),
        )
        y_preds = np.asarray(model.preds_list)
#         print(y_preds.shape)
        for failt_id_idx, fault_id in enumerate(test_fault_ids):
            rc_nodes = set(fdg.root_cause_instances_of(fault_id))
            for n in rc_nodes:
                exp_records.append({
                    'Seen': n in train_rc_nodes,
                    'Rank': np.where(y_preds[failt_id_idx, :] == fdg.instance_to_gid(n))[0][0] + 1,
                    'fault_id': fault_id,
                    "node": n,
                })
        records.extend(exp_records)
        # break
    ret_df = pd.DataFrame.from_records(records)
    ret_df['Dataset'] = dataset
    return ret_df

OUTPUT_DIR = Path('/SSF/output/plot_generalization_performance/')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% pycharm={"name": "#%%\n"} tags=[]
df1 = pd.concat([
    parse_pred_ret_by_recurring(eval_rets.loc[
        (eval_rets.Method == 'GRU+GAT-H4-L8+BAL') & 
        (eval_rets.Dataset.str.startswith('AIOPS20')),
        "实验路径",
    ].values, dataset='A'),
    parse_pred_ret_by_recurring(eval_rets.loc[
        (eval_rets.Method == 'GRU+GAT-H4-L8+BAL') & 
        (eval_rets.Dataset.str.startswith('AIOPS21')),
        "实验路径",
    ].values, dataset='B'),
    parse_pred_ret_by_recurring(eval_rets.loc[
        (eval_rets.Method == 'GRU+GAT-H4-L8+BAL') & 
        (eval_rets.Dataset == 'CCB-Oracle'),
        "实验路径",
    ].values, dataset='C')
])
df1['Method'] = 'Ours'

# %%
df1.to_pickle(OUTPUT_DIR / "ours_rank.pkl")

# %%
from DejaVu.config import DejaVuConfig
from iSQUAD.iSQ import ISQUARD
from iSQUAD.config import ISQUADConfig
from failure_dependency_graph import FDG, FDGModelInterface
from diskcache import Cache
def parse_isq(config: ISQUADConfig, dataset):
    fdg, real_paths = FDG.load(
        dir_path=config.data_dir,
        graph_path=config.graph_config_path, metrics_path=config.metrics_path, faults_path=config.faults_path,
        return_real_path=True
    )
    dataset_cache_dir = config.cache_dir / ".".join(
        f"{k}={real_paths[k]}" for k in sorted(real_paths.keys())
    ).replace('/', '_')
    print(f"dataset_cache_dir={dataset_cache_dir}")
    cache = Cache(str(dataset_cache_dir), size_limit=int(1e10))
    del dataset_cache_dir, real_paths
    mp = FDGModelInterface.get_metric_preprocessor(fdg, cache, config)
    del cache
        
    isq = ISQUARD(fdg, config, mp)
    
    train_rc_nodes = set()
    for fault_id in isq.train_fault_ids:
        train_rc_nodes |= set(fdg.root_cause_instances_of(fault_id))
    
    y_preds = isq()
    records = []
    for fault_id, y_pred in zip(isq.test_fault_ids, y_preds):
#         print(y_pred)
        for n in set(fdg.root_cause_instances_of(fault_id)):
            if fdg.instance_to_gid(n) not in y_pred:
#                 assert n not in train_rc_nodes
                rank = len(y_pred) / 2 + fdg.n_failure_instances / 2
            else:
                rank = y_pred.index(fdg.instance_to_gid(n)) + 1
            records.append({
                'Seen': n in train_rc_nodes,
                "Rank": rank,
                "fault_id": fault_id,
                "node": n,
                "Method": "iSQUAD",
            })
    ret_df = pd.DataFrame.from_records(records)
    ret_df['Dataset'] = dataset
    return ret_df
df2 = pd.concat([
    parse_isq(ISQUADConfig().from_dict(dict(flush_dataset_cache=False, data_dir="/SSF/data/A1")), dataset='A'),
    parse_isq(ISQUADConfig().from_dict(dict(flush_dataset_cache=False, data_dir="/SSF/data/A2")), dataset='A'),
    parse_isq(ISQUADConfig().from_dict(dict(flush_dataset_cache=False, data_dir="/SSF/data/B")), dataset='B'),
    parse_isq(ISQUADConfig().from_dict(dict(flush_dataset_cache=False, data_dir="/SSF/data/C")), dataset='C'),
])

# %%
df2.to_pickle(OUTPUT_DIR / "isq_rank.pkl")

# %%
from DejaVu.config import DejaVuConfig
from failure_dependency_graph import FDG
from diskcache import Cache
from JSS20.system_graph import GraphLibrary
from failure_dependency_graph import split_failures_by_type

def parse_jss20(config: DejaVuConfig, dataset):
    cdp, real_paths = FDG.load(
        dir_path=config.data_dir,
        graph_path=config.graph_config_path, metrics_path=config.metrics_path, faults_path=config.faults_path,
        return_real_path=True
    )
    train_ids, _, test_ids = split_failures_by_type(
        cdp.failures_df, split=config.dataset_split_ratio,
        train_set_sampling_ratio=config.train_set_sampling,
        fdg=cdp,
    )
    dataset_cache_dir = config.cache_dir / ".".join(
        f"{k}={real_paths[k]}" for k in sorted(real_paths.keys())
    ).replace('/', '_')
    print(f"dataset_cache_dir={dataset_cache_dir}")
    cache = Cache(str(dataset_cache_dir), size_limit=int(1e10))
    del dataset_cache_dir, real_paths
    mp = FDGModelInterface.get_metric_preprocessor(cdp, cache, config)
    del cache
    graph_library = GraphLibrary(cdp, train_ids[:], mp=mp)
    
    train_rc_nodes = set()
    for fault_id in train_ids:
        train_rc_nodes |= set(cdp.failures_df.iloc[fault_id]['root_cause_node'].split(';'))
    
    records = []
    for fault_id  in test_ids:
        y_pred = graph_library.query(fault_id)
        for n in set(cdp.failures_df.iloc[fault_id]['root_cause_node'].split(';')):
            if n not in y_pred:
                rank = len(y_pred) / 2 + cdp.n_failure_instances / 2
            else:
                rank = y_pred.index(n) + 1
            records.append({
                'Seen': n in train_rc_nodes,
                "Rank": rank,
                "fault_id": fault_id,
                "node": n,
                "Method": "JSS'20",
            })
    ret_df = pd.DataFrame.from_records(records)
    ret_df['Dataset'] = dataset
    return ret_df
df3 = pd.concat([
    parse_jss20(DejaVuConfig().from_dict(dict(flush_dataset_cache=False, data_dir="/SSF/data/A1")), dataset='A'),
    parse_jss20(DejaVuConfig().from_dict(dict(flush_dataset_cache=False, data_dir="/SSF/data/A2")), dataset='A'),
    parse_jss20(DejaVuConfig().from_dict(dict(flush_dataset_cache=False, data_dir="/SSF/data/B")), dataset='B'),
    parse_jss20(DejaVuConfig().from_dict(dict(flush_dataset_cache=False, data_dir="/SSF/data/C")), dataset='C'),
])

# %%
df3.to_pickle(OUTPUT_DIR / "jss20_rank.pkl")

# %% tags=[]
from DejaVu.dataset import prepare_sklearn_dataset
from sklearn.tree import DecisionTreeClassifier
from pyprof import profile
from failure_dependency_graph import FDG

def parse_decision_trees(config: DejaVuConfig, dataset):
    dataset_name = dataset
    cdp, real_paths = FDG.load(
        dir_path=config.data_dir,
        graph_path=config.graph_config_path, metrics_path=config.metrics_path, faults_path=config.faults_path,
        return_real_path=True
    )
    dataset_cache_dir = config.cache_dir / ".".join(
        f"{k}={real_paths[k]}" for k in sorted(real_paths.keys())
    ).replace('/', '_')
    logger.info(f"dataset_cache_dir={dataset_cache_dir}")
    cache = Cache(str(dataset_cache_dir), size_limit=int(1e10))
    del dataset_cache_dir, real_paths

    dataset, (train_fault_ids, _, test_fault_ids) = prepare_sklearn_dataset(cdp, config, cache, mode=config.ts_feature_mode)

    y_probs = np.zeros((len(test_fault_ids), cdp.n_failure_instances), dtype=np.float32)
    y_trues = []
    for fault_id in test_fault_ids:
        y_trues.append(set(map(cdp.instance_to_gid, cdp.failures_df.iloc[fault_id]['root_cause_node'].split(';'))))
    models = {}
    for node_type in cdp.failure_classes:
        feature_names, (
            (train_x, train_y, _, _), _, (test_x, _, fault_ids, node_names)
        ) = dataset[node_type]
        model = DecisionTreeClassifier()
        with profile("Training"):
            model.fit(train_x, train_y)
        models[node_type] = model
        _y_probs = model.predict_proba(test_x)
        for fault_id, node_name, prob in zip(fault_ids, node_names, _y_probs):
            with profile("Inference for each failure"):
                y_probs[test_fault_ids.index(fault_id), cdp.instance_to_gid(node_name)] = 1 - prob[0].item()
    y_preds = [np.arange(len(prob))[np.argsort(prob, axis=-1)[::-1]].tolist() for prob in y_probs]
    
    train_rc_nodes = set()
    for fault_id in train_fault_ids:
        train_rc_nodes |= set(cdp.failures_df.iloc[fault_id]['root_cause_node'].split(';'))
    
    records = []
    for fault_id, y_pred in zip(test_fault_ids, y_preds):
#         print(y_pred)
        for n in set(cdp.failures_df.iloc[fault_id]['root_cause_node'].split(';')):
            if cdp.instance_to_gid(n) not in y_pred:
#                 assert n not in train_rc_nodes
                rank = len(y_pred) / 2 + cdp.n_failure_instances / 2
            else:
                rank = y_pred.index(cdp.instance_to_gid(n)) + 1
            records.append({
                'Seen': n in train_rc_nodes,
                "Rank": rank,
                "fault_id": fault_id,
                "node": n,
                "Method": "Decision Tree",
            })
    ret_df = pd.DataFrame.from_records(records)
    ret_df['Dataset'] = dataset_name
    return ret_df
            
df4 = pd.concat([
    parse_decision_trees(DejaVuConfig().from_dict(dict(flush_dataset_cache=False, data_dir="/SSF/data/A1")), dataset='A'),
    parse_decision_trees(DejaVuConfig().from_dict(dict(flush_dataset_cache=False, data_dir="/SSF/data/A2")), dataset='A'),
    parse_decision_trees(DejaVuConfig().from_dict(dict(flush_dataset_cache=False, data_dir="/SSF/data/B")), dataset='B'),
    parse_decision_trees(DejaVuConfig().from_dict(dict(flush_dataset_cache=False, data_dir="/SSF/data/C")), dataset='C'),
])

# %%
df4.to_pickle(OUTPUT_DIR / "dt_rank.pkl")

# %%
from pycausal.pycausal import pycausal as pc
pc = pc()
pc.start_vm()

# %% tags=[]
from random_walk_failure_instance import workflow as rw_fi_workflow
from random_walk_failure_instance import RandomWalkFailureInstanceConfig
from random_walk_single_metric import RandomWalkSingleMetricConfig
from random_walk_single_metric import workflow as rw_metric_workflow
from failure_dependency_graph import FDGModelInterface
from pathlib import Path
from functools import partial

def parse_random_walk(config, workflow_func, dataset_name, method_name):
    y_trues, y_preds = workflow_func(config)
    base = FDGModelInterface(config)
    fdg = base.fdg
    
    train_rc_nodes = set()
    for fault_id in base.train_failure_ids:
        train_rc_nodes |= set(fdg.root_cause_instances_of(fault_id))
    
    records = []
    for fault_id, y_pred in zip(base.test_failure_ids, y_preds):
        for n in set(fdg.root_cause_instances_of(fault_id)):
            if n not in y_pred:
                rank = len(y_pred) / 2 + fdg.n_failure_instances / 2
            else:
                rank = y_pred.index(n) + 1
            records.append({
                'Seen': n in train_rc_nodes,
                "Rank": rank,
                "fault_id": fault_id,
                "node": n,
                "Method": method_name,
            })
    ret_df = pd.DataFrame.from_records(records)
    ret_df['Dataset'] = dataset_name
    return ret_df

df5 = pd.concat([
    parse_random_walk(
        RandomWalkSingleMetricConfig().from_dict(dict(
            flush_dataset_cache=False, data_dir=Path("/SSF/data/A1"), output_dir=OUTPUT_DIR / "A1_RW@Metric", flush_causal_graph_cache=False
        )),
        partial(rw_metric_workflow, in_jvm_context=True),
        dataset_name='A',
        method_name="RW@Metric",
    ),
    parse_random_walk(
        RandomWalkSingleMetricConfig().from_dict(dict(
            flush_dataset_cache=False, data_dir=Path("/SSF/data/A2"), output_dir=OUTPUT_DIR / "A2_RW@Metric", flush_causal_graph_cache=False
        )),
        partial(rw_metric_workflow, in_jvm_context=True),
        dataset_name='A',
        method_name="RW@Metric",
    ),
    parse_random_walk(
        RandomWalkSingleMetricConfig().from_dict(dict(
            flush_dataset_cache=False, data_dir=Path("/SSF/data/B"), output_dir=OUTPUT_DIR / "B_RW@Metric", flush_causal_graph_cache=False
        )),
        partial(rw_metric_workflow, in_jvm_context=True),
        dataset_name='B',
        method_name="RW@Metric",
    ),
])

# %%
df5.to_pickle(OUTPUT_DIR / "rw_metric_rank.pkl")

# %%
df6 = pd.concat([
    parse_random_walk(
        RandomWalkFailureInstanceConfig().from_dict(dict(flush_dataset_cache=False, data_dir=Path("/SSF/data/A1"), output_dir=OUTPUT_DIR / "A1_RW@FI")),
        rw_fi_workflow,
        dataset_name='A',
        method_name="RW@FI",
    ),
    parse_random_walk(
        RandomWalkFailureInstanceConfig().from_dict(dict(flush_dataset_cache=False, data_dir=Path("/SSF/data/A2"), output_dir=OUTPUT_DIR / "A2_RW@FI")),
        rw_fi_workflow,
        dataset_name='A',
        method_name="RW@FI",
    ),
    parse_random_walk(
        RandomWalkFailureInstanceConfig().from_dict(dict(flush_dataset_cache=False, data_dir=Path("/SSF/data/B"), output_dir=OUTPUT_DIR / "B_RW@FI")),
        rw_fi_workflow,
        dataset_name='B',
        method_name="RW@FI",
    ),
])

# %%
df6.to_pickle(OUTPUT_DIR / "rw_fi_rank.pkl")

# %%
import pandas as pd
df1 = pd.read_pickle(OUTPUT_DIR / 'ours_rank.pkl')
df2 = pd.read_pickle(OUTPUT_DIR / 'isq_rank.pkl')
df3 = pd.read_pickle(OUTPUT_DIR / 'jss20_rank.pkl')
df4 = pd.read_pickle(OUTPUT_DIR / 'dt_rank.pkl')
df5 = pd.read_pickle(OUTPUT_DIR / 'rw_metric_rank.pkl')
df6 = pd.read_pickle(OUTPUT_DIR / 'rw_fi_rank.pkl')

# %% pycharm={"name": "#%%\n"}
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df = pd.concat([df1, df2, df3, df4, df5, df6])
df['log(Rank)'] = df['Rank'].map(np.log)
df['Seen'] = df['Seen'].map(lambda _: {True: 'Seen', False: 'Unseen'}[_])

df.replace({"Ours": "DéjàVu"}, inplace=True)

plt.figure(dpi=300, figsize=(4.5, 1.))
ax = sns.barplot(data=df[df.Dataset == 'A'], y='log(Rank)', x='Method', hue='Seen', hue_order=['Seen', 'Unseen'], palette='Blues')
plt.legend(title=False, fontsize='small')
plt.xlabel(None)
plt.ylabel("$\log_{10}(rank)$")
plt.xticks(rotation=15)
plt.yticks([0.0, 2.5, 5.0], )
plt.savefig(OUTPUT_DIR / 'A.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()

plt.figure(dpi=300, figsize=(4.5, 1.))
ax = sns.barplot(data=df[df.Dataset == 'B'], y='log(Rank)', x='Method', hue='Seen', hue_order=['Seen', 'Unseen'], palette='Blues')
plt.legend(title=False, loc='upper left', fontsize='small')
plt.xlabel(None)
plt.yticks([0.0, 2.5, 5.0])
plt.xticks(rotation=15)
plt.ylabel("$\log_{10}(rank)$")
plt.savefig(OUTPUT_DIR / 'B.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()

# %% [markdown]
# ##### 

# %%
