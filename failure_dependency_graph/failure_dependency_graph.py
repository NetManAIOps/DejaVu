from collections import defaultdict
from functools import lru_cache, reduce, cached_property
from itertools import groupby
from pathlib import Path
from pprint import pformat
from typing import Tuple, Dict, List, Set, Union, Callable, Optional

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import torch as th
from dgl import DGLHeteroGraph, heterograph
from loguru import logger
from networkx import DiGraph
from pyprof import profile

__all__ = [
    'FDG'
]

from failure_dependency_graph.generate_virtual_FDG import generate_virtual_FDG
from failure_dependency_graph.parse_yaml_graph_config import parse_yaml_graph_config
from utils import read_dataframe


class FDG:
    """
    A container for FDG, metrics and failures
    """
    _nx_failure_graphs: List[DiGraph]
    _nx_overall_graph: DiGraph

    @profile
    def __init__(
            self, *, metrics: pd.DataFrame, failures: pd.DataFrame,
            anomaly_direction_constraint: Dict[str, str] = None,
            graph: Optional[nx.DiGraph] = None, failure_graphs: Optional[List[nx.DiGraph]] = None,
    ):
        """
        :param graph: a networkx graph defines the failure dependency graph structure
            A node should have a type (str) and metrics (a list of str)
            A edge should have a type (str)
        :param metrics:
            The columns should be name, timestamp (in seconds) and value
        """
        assert set(metrics.columns) >= {'timestamp', 'value', "name"}, f"{metrics.columns=}"
        assert set(failures.columns) >= {'timestamp', 'root_cause_node'}, f"{failures.columns=}"
        assert graph is not None or failure_graphs is not None, "Either graph or failure_graphs should be provided"
        assert failure_graphs is None or len(failure_graphs) == len(failures), \
            f"The length of {len(failure_graphs)=} should be equal to the length of {len(failures)=}"
        if failure_graphs is None:
            failure_graphs = [graph.copy() for _ in range(len(failures))]
        if graph is None:
            graph = nx.compose_all(failure_graphs)
        self._nx_overall_graph = graph
        self._nx_failure_graphs = failure_graphs

        self._node_list: Dict[str, List[str]] = dict(map(
            lambda pair: (pair[0], [_[0] for _ in pair[1]]),
            groupby(sorted(
                graph.nodes(data=True), key=lambda _: _[1]['type']
            ), key=lambda _: _[1]['type'])
        ))
        self._node_to_idx: Dict[str, Dict[str, int]] = {
            node_type: {node: idx for idx, node in enumerate(node_list)}
            for node_type, node_list in self._node_list.items()
        }
        self._node_to_class: Dict[str, str] = dict(sum(
            [[(node, fc) for node in nodes] for fc, nodes in self._node_list.items()],
            []
        ))

        self._all_metrics_set: Set[str] = reduce(
            lambda a, b: a | b,
            [set(data['metrics']) for _, data in graph.nodes(data=True)],
            set()
        )
        self._metric_size_dict: Dict[str, int] = self.__get_metric_size_dict(graph)
        # 假设每个类型的指标都是相同的，传入的顺序也是相同的
        self._node_metrics_dict: Dict[str, List[str]] = self.__get_node_metrics_dict(graph)
        self._node_type_metrics_dict: Dict[str, List[str]] = self.__get_node_type_metrics_dict(
            self._node_metrics_dict, self._node_list
        )

        self._metrics_df = metrics[metrics.name.isin(self._all_metrics_set)]
        self._overall_hg = self.__get_hg(graph)
        logger.debug(f"All edge types: {pformat(self._overall_hg.etypes)}")
        self._node_types = self._overall_hg.ntypes
        assert self.overall_hg.ntypes == self.failure_classes

        self._faults_df = failures

        self._metric_mean_dict: Dict[str, float] = self._metrics_df.groupby('name')['value'].mean().to_dict()

        logger.info(f"the number of nodes: {self.n_failure_instances}")
        logger.info(f"all ({len(self.failure_classes)}) node types: {self.failure_classes}")
        logger.info(f"the number of metrics: "
                    f"{sum(self.metric_number_dict.values())} (metric_type) "
                    f"{sum([len(_) for _ in self.FI_metrics_dict.values()])} (name)")

        logger.info(f"The metrics of each node type: \n{pformat(self.FC_metrics_dict)}")

        self._anomaly_direction_constraint = {_.split('##')[1]: 'b' for _ in self._all_metrics_set}
        self._anomaly_direction_constraint.update(
            anomaly_direction_constraint if anomaly_direction_constraint is not None else {}
        )

    @staticmethod
    @profile
    def load(
            dir_path: Union[Path, str, None] = None, *,
            graph_path: Union[Path, str, None] = None,
            metrics_path: Union[Path, str, None] = None,
            faults_path: Union[Path, str, None] = None,
            use_anomaly_direction_constraint: bool = False,
            return_real_path: bool = False,
            loaded_FDG: Optional['FDG'] = None,
    ) -> Union['FDG', Tuple['FDG', Dict]]:
        """
        :param dir_path: read "graph.yaml", "metrics.norm.pkl", and "faults.csv" in this directory by default
        :param graph_path: overwrite the default graph path
        :param metrics_path: overwrite the default metrics path
        :param faults_path: overwrite the default faults path
        :param use_anomaly_direction_constraint: whether to use anomaly direction constraint
        :param return_real_path: whether to return the real path of the loaded files
        :param loaded_FDG: if not None, use the given FDG instead of loading from files
        :return: a FDG object,
            (optional) and a dict of the real paths of the loaded files whose keys are 'graph', 'metrics', and 'faults'
        """
        if dir_path is not None:
            dir_path = Path(dir_path)

            if str(dir_path).startswith("/dev/generate_random_FDG"):
                fdg = generate_virtual_FDG(
                    **{
                        _.split("=")[0]: float(_.split("=")[1])
                        for _ in str(dir_path.relative_to('/dev/generate_random_FDG')).split("/")
                    }
                )
                if return_real_path:
                    return fdg, {
                        'graph': "/dev/generate_random_FDG/graph",
                        'metrics': "/dev/generate_random_FDG/metrics",
                        "faults": "/dev/generate_random_FDG/faults"}
                else:
                    return fdg

            assert dir_path.is_dir() and dir_path.exists(), dir_path
        graph_path = dir_path / 'graph.yml' if graph_path is None else Path(graph_path)
        metrics_path = dir_path / "metrics.norm.pkl" if metrics_path is None else Path(metrics_path)
        faults_path = dir_path / "faults.csv" if faults_path is None else Path(faults_path)
        if not graph_path.exists():
            graph_path = None
        assert metrics_path.exists(), metrics_path
        assert faults_path.exists(), faults_path

        if not loaded_FDG:
            fdg = FDG._load_FDG(
                dir_path=dir_path,
                graph_path=graph_path,
                metrics_path=metrics_path,
                faults_path=faults_path,
                use_anomaly_direction_constraint=use_anomaly_direction_constraint,
            )
        else:
            fdg = loaded_FDG
        if return_real_path:
            return fdg, {'graph': graph_path, 'metrics': metrics_path, "faults": faults_path}
        else:
            return fdg

    @staticmethod
    def _load_FDG(
            dir_path: Path, graph_path: Optional[Path], faults_path: Path, metrics_path: Path,
            use_anomaly_direction_constraint: bool,
    ):
        if not use_anomaly_direction_constraint:
            anomaly_direction_constraint = {}
        else:
            try:
                with open(dir_path / "anomaly_direction_constraint.json", 'r') as f:
                    import json
                    anomaly_direction_constraint = json.load(f)
            except Exception as e:
                logger.error(e)
                anomaly_direction_constraint = {}
        failures_df = read_dataframe(faults_path)
        failures_graph_paths = [
            _ for _ in
            [dir_path / 'graphs' / f"graph_{t:.0f}.yml" for t in failures_df["timestamp"]]
            if _.exists()
        ]
        latest_source_modification_time = max(
            [
                _.stat().st_mtime
                for _ in failures_graph_paths + [graph_path, faults_path, metrics_path]
                if _ is not None and _.exists()
            ]
        )
        pickled_FDG_path: Path = dir_path / "FDG.pkl"
        is_loaded_from_pickle = False
        try:
            if pickled_FDG_path.exists() and pickled_FDG_path.stat().st_mtime >= latest_source_modification_time:
                logger.info(f"Loading FDG from {pickled_FDG_path}")
                with open(pickled_FDG_path, 'rb') as f:
                    fdg = pickle.load(f)
                    is_loaded_from_pickle = True
        except pickle.PickleError as e:
            logger.exception("Read pickled FDG error", exception=e)
        if not is_loaded_from_pickle:
            if len(failures_graph_paths) == len(failures_df):
                failure_graphs = [parse_yaml_graph_config(_) for _ in failures_graph_paths]
            else:
                failure_graphs = None
            logger.debug(f"Load CDP: {graph_path=} {metrics_path=} {faults_path=} {failures_graph_paths=}")
            fdg = FDG(
                graph=parse_yaml_graph_config(graph_path) if graph_path is not None else None,
                failure_graphs=failure_graphs,
                metrics=read_dataframe(metrics_path),
                failures=failures_df,
                anomaly_direction_constraint=anomaly_direction_constraint,
            )
            with open(pickled_FDG_path, 'wb') as f:
                pickle.dump(fdg, f, protocol=-1)
        else:
            fdg = fdg  # trick IDE
        return fdg

    #######################
    # Metrics
    #######################
    @property
    def metrics_df(self) -> pd.DataFrame:
        """
        :return: A dataframe contains the following columns: timestamp, name, value
        """
        return self._metrics_df

    @property
    def metric_mean_dict(self) -> Dict[str, float]:
        """
        :return: The average value of each metric
        """
        return self._metric_mean_dict

    @property
    def metric_number_dict(self) -> Dict[str, int]:
        """
        :return: The number of metrics of each node type
        """
        return self._metric_size_dict

    @property
    def FI_metrics_dict(self) -> Dict[str, List[str]]:
        """
        :return: The list of metrics of each failure instance
        """
        return self._node_metrics_dict

    @property
    def FC_metrics_dict(self) -> Dict[str, List[str]]:
        """
        :return: The list of metrics of each failure classes
        """
        return self._node_type_metrics_dict

    @cached_property
    def metric_to_FI_list_dict(self) -> Dict[str, List[str]]:
        """
        :return: The mapping from metrics to failure instances
        """
        ret = defaultdict(list)
        for FI, metrics in self.FI_metrics_dict.items():
            for metric in metrics:
                ret[metric].append(FI)
        return dict(ret)

    @property
    def anomaly_direction_constraint(self) -> Dict[str, str]:
        """
        :return:
            'u', upside the baseline is abnormal
            'd', downside the baseline is abnormal
            'b', both sides are abnormal
        """
        return self._anomaly_direction_constraint

    #######################
    # Nodes
    #######################
    @property
    def n_failure_instances(self) -> int:
        """
        :return: The number of failure instances
        """
        return self.overall_hg.number_of_nodes()

    @property
    def failure_instances(self) -> Dict[str, List[str]]:
        """
        :return: A dict mapping a failure class to its instances
        """
        return self._node_list

    @cached_property
    def flatten_failure_instances(self) -> List[str]:
        """
        :return: A list of failure instance names, where the indices are the gids of the instances
        """
        return sum([self.failure_instances[_] for _ in self.failure_classes], [])

    @property
    def failure_classes(self) -> List[str]:
        return self._node_types

    def instance_to_class(self, name: str) -> str:
        """
        Map a failure instance to its failure class
        :param name:
        :return:
        """
        return self._node_to_class[name]

    def gid_to_instance(self, gid: int) -> str:
        """
        Map a global id to a failure instance name
        :param gid: the global id of a failure instance
        :return: the name of the instance
        """
        node_type, node_typed_id = self.gid_to_local_id(gid)
        return self.failure_instances[node_type][node_typed_id]

    def instance_to_gid(self, name: str) -> int:
        """
        Map a failure instance name to a global id
        :param name:
        :return:
        """
        _node_type = self.instance_to_class(name)
        _local_index = self.failure_instances[_node_type].index(name)
        return self.local_id_to_gid(_node_type, _local_index)

    def gid_to_local_id(self, global_id: int) -> Tuple[str, int]:
        """
        Calculate the failure class and local id of a global id
        :param global_id:
        :return: (failure class, local id). Local id is the index of the failure instance in the failure class
        """
        resolver = _get_global_id_resolver(self.overall_hg)
        node_type, node_typed_id = resolver(global_id)
        return node_type, node_typed_id

    def local_id_to_gid(self, failure_class: str, local_id: int) -> int:
        """
        Calculate the global id of a failure class and local id
        :param failure_class:
        :param local_id: the local id of a failure instance inside the failure class
        :return:
        """
        getter = _get_global_id_getter(self.overall_hg)
        return getter(failure_class, local_id)

    def local_id_to_instance(self, failure_class: str, local_id: int) -> str:
        return self.gid_to_instance(self.local_id_to_gid(failure_class, local_id))

    def instance_to_local_id(self, name: str) -> Tuple[str, int]:
        return self.gid_to_local_id(self.instance_to_gid(name))

    #######################
    # DGLGraph
    #######################

    @property
    def overall_hg(self) -> DGLHeteroGraph:
        return self._overall_hg

    @cached_property
    def overall_homo(self) -> dgl.DGLGraph:
        return self.convert_to_homo(self.overall_hg)

    def overall_networkx(self) -> nx.DiGraph:
        return self._nx_overall_graph

    def networkx_graph_at(self, fid: int) -> nx.DiGraph:
        return self._nx_failure_graphs[fid]

    @lru_cache
    def hetero_graph_at(self, failure_id: int) -> DGLHeteroGraph:
        return self.__get_hg(self._nx_failure_graphs[failure_id])

    @lru_cache
    def homo_graph_at(self, failure_id: int) -> dgl.DGLGraph:
        return self.convert_to_homo(self.hetero_graph_at(failure_id))

    def convert_to_homo(self, hg: DGLHeteroGraph) -> dgl.DGLGraph:
        _gid_dict = {}
        _og: dgl.DGLGraph = dgl.to_homogeneous(hg)
        for _node, _ori_id, _ori_type in zip(_og.nodes(), _og.ndata['_ID'], _og.ndata['_TYPE']):
            _gid_dict[int(_node)] = self.local_id_to_gid(hg.ntypes[_ori_type], _ori_id)
        return dgl.graph(
            data=([_gid_dict[int(_)] for _ in _og.edges()[0]], [_gid_dict[int(_)] for _ in _og.edges()[1]]),
            num_nodes=self.overall_hg.number_of_nodes(),
            device=hg.device,
            idtype=hg.idtype,
        )

    #######################
    # Failures
    #######################
    @property
    def failures_df(self) -> pd.DataFrame:
        return self._faults_df

    def failure_at(self, fid: int):
        return self._faults_df.iloc[fid]

    def root_cause_instances_of(self, fid: int) -> List[str]:
        return self.failure_at(fid)['root_cause_node'].split(';')

    @property
    def failure_ids(self) -> List[int]:
        return list(range(len(self.failures_df)))

    @cached_property
    def timestamp_range(self) -> Tuple[int, int]:
        fault_ts_min = self.failure_timestamps()[0]
        fault_ts_max = self.failure_timestamps()[-1]
        valid_ts_min = self.valid_timestamps[0]
        valid_ts_max = self.valid_timestamps[-1]
        return min(valid_ts_min, fault_ts_min), max(valid_ts_max, fault_ts_max)

    @cached_property
    def valid_timestamps(self) -> np.ndarray:
        """
        至少一个指标是有值的时间戳
        :return:
        """
        return np.sort(self.metrics_df['timestamp'].unique())

    @lru_cache(maxsize=None)
    def failure_timestamps(self, duration=5, granularity=60, before_duration=0) -> np.ndarray:
        start = self.failures_df['timestamp'].unique().reshape(-1, 1)
        expand = (np.arange(-before_duration, duration + 1) * granularity).reshape(1, -1)
        return np.unique((start + expand).reshape(-1))

    @lru_cache(maxsize=None)
    def normal_timestamps(self, granularity=60, duration=10, before_duration=0) -> np.ndarray:
        return np.asarray(
            list(set(
                self.valid_timestamps[duration:-duration] if duration > 0 else self.valid_timestamps
            ) - set(
                self.failure_timestamps(
                    duration=duration, granularity=granularity, before_duration=before_duration
                )
            ))
        )

    ###########################
    # Init
    ############################
    @staticmethod
    @lru_cache(maxsize=None)
    def __get_metric_size_dict(graph: nx.DiGraph) -> Dict[str, int]:
        ret = {}
        for _, data in graph.nodes(data=True):
            if data['type'] not in ret:
                ret[data['type']] = data['metrics']
            else:
                assert list(map(
                    lambda _: _.split('##')[1],
                    ret[data['type']]
                )) == list(map(
                    lambda _: _.split('##')[1],
                    data['metrics']
                )), \
                    f"The metrics should be same for the nodes of each type: " \
                    f"{data['type']=} {ret[data['type']]=} {data['metrics']=}"
        return dict(map(lambda _: (_[0], len(_[1])), ret.items()))

    def __get_hg(self, graph: nx.DiGraph) -> DGLHeteroGraph:
        hg_data_dict = defaultdict(lambda: ([], []))
        for u, v, data in graph.edges(data=True):
            u_type = graph.nodes[u]['type']
            v_type = graph.nodes[v]['type']
            edge_type = (u_type, f"{u_type}-{data['type']}-{v_type}", v_type)
            hg_data_dict[edge_type][0].append(self._node_to_idx[u_type][u])
            hg_data_dict[edge_type][1].append(self._node_to_idx[v_type][v])
        _hg: DGLHeteroGraph = heterograph(
            {
                key: (th.tensor(src_list).long(), th.tensor(dst_list).long())
                for key, (src_list, dst_list) in hg_data_dict.items()
            }
        )
        del hg_data_dict
        return _hg

    @staticmethod
    def __get_node_metrics_dict(graph: nx.DiGraph):
        ret = {}
        for node, data in graph.nodes(data=True):
            ret[node] = data['metrics']
        return ret

    @staticmethod
    def __get_node_type_metrics_dict(
            node_metric_dict: Dict[str, List[str]], nodes_list: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        ret = {}
        for node_type, nodes in nodes_list.items():
            for node in nodes:
                metrics = list(map(lambda _: _.split('##')[1], node_metric_dict[node]))
                if node_type not in ret:
                    ret[node_type] = metrics
                assert ret[node_type] == metrics
        return ret


@lru_cache
@profile
def _get_global_id_getter(hg: DGLHeteroGraph) -> Callable[[str, int], int]:
    ptr = 0
    ntype_base_ptr = {}
    for ntype in hg.ntypes:
        if ntype in hg.ntypes:
            ntype_base_ptr[ntype] = ptr
        ptr += hg.number_of_nodes(ntype)
    del ptr

    def getter(node_type: str, node_id: int) -> id:
        return ntype_base_ptr[node_type] + node_id

    return getter


@lru_cache
@profile
def _get_global_id_resolver(hg: DGLHeteroGraph) -> Callable[[int], Tuple[str, int]]:
    id_to_type = {}
    ntype_base_ptr = {}
    ptr = 0
    for ntype in hg.ntypes:
        if ntype in hg.ntypes:
            ntype_base_ptr[ntype] = ptr
        new_ptr = ptr + hg.number_of_nodes(ntype)
        for _ in range(ptr, ptr + new_ptr):
            id_to_type[_] = ntype
        ptr = new_ptr
    del ptr, new_ptr

    def resolver(global_id: int) -> Tuple[str, int]:
        global_id = int(global_id)
        node_type = id_to_type[global_id]
        node_id = global_id - ntype_base_ptr[node_type]
        return node_type, node_id

    return resolver
