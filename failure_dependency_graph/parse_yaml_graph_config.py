import os
from itertools import product
from pathlib import Path
from pprint import pformat
from typing import Union, Optional, Any, Dict, List

import networkx as nx
from loguru import logger
from networkx.drawing.nx_agraph import write_dot
from yaml import load, CLoader as Loader


def insert_global_params(keys: List[str], global_params: Dict[str, Any], target: Dict[str, Any]):
    for key in keys:
        assert key in global_params
        target[key] = global_params[key]
    return target


def parse_yaml_graph_config(path: Union[str, Path], output_dir: Optional[Path] = None) -> nx.DiGraph:
    def metrics_sorted(metrics):
        return sorted(metrics, key=lambda _: _.split("##")[1])

    path = Path(path)
    logger.debug(f"parsing Graph from {path!s}")
    with open(path) as f:
        input_data = load(f, Loader=Loader)
    g = nx.DiGraph()
    global_params = {}
    for obj in input_data:
        if obj.get('class', "") == "global_params":
            global_params = obj
            break
    # parse nodes in the first pass
    for obj in input_data:
        try:
            if obj['class'] != "node":
                continue
            if "params" in obj or "global_params" in obj:
                params = obj.get("params", {})
                insert_global_params(obj.get('global_params', []), global_params, params)
                keys = list(params.keys())
                if obj.get('product', False):
                    for values in product(*params.values()):
                        node_id = obj["id"].format(**{k: v for k, v in zip(keys, values)})
                        node_metrics = [_.format(**{k: v for k, v in zip(keys, values)}) for _ in obj["metrics"]]
                        g.add_node(node_id, **{"metrics": metrics_sorted(node_metrics), "type": obj["type"]})
                else:
                    values_list = list(zip(*params.values()))
                    for values in values_list:
                        node_id = obj["id"].format(**{k: v for k, v in zip(keys, values)})
                        node_metrics = [_.format(**{k: v for k, v in zip(keys, values)}) for _ in obj["metrics"]]
                        g.add_node(node_id, **{"metrics": metrics_sorted(node_metrics), "type": obj["type"]})
            else:
                g.add_node(obj['id'], **{"metrics": metrics_sorted(obj['metrics']), "type": obj["type"]})
        except KeyError as e:
            logger.error(f"{e!r} obj={pformat(obj)}")

    def add_edge(_src, _dst, _attrs):
        if _src in g and _dst in g:  # ignore edges that don't exist
            g.add_edge(_src, _dst, **_attrs)
        else:
            logger.warning(f"ignoring edge {_src} -> {_dst}")

    # parse edges in the second pass
    for obj in input_data:
        try:
            if obj['class'] != "edge":
                continue
            if "params" in obj or "global_params" in obj:
                params = obj.get('params', {})
                insert_global_params(obj.get('global_params', []), global_params, params)
                keys = list(params.keys())
                if obj.get('product', False):
                    for values in product(*params.values()):
                        src = obj["src"].format(**{k: v for k, v in zip(keys, values)})
                        dst = obj["dst"].format(**{k: v for k, v in zip(keys, values)})
                        add_edge(src, dst, {"type": obj["type"]})
                else:
                    values_list = list(zip(*params.values()))
                    for values in values_list:
                        src = obj["src"].format(**{k: v for k, v in zip(keys, values)})
                        dst = obj["dst"].format(**{k: v for k, v in zip(keys, values)})
                        add_edge(src, dst, {"type": obj["type"]})
            else:
                add_edge(obj["src"], obj["dst"], {"type": obj["type"]})
        except KeyError as e:
            logger.error(f"{e!r} obj={pformat(obj)}")
    if output_dir is not None:
        dot_path = output_dir / f"{path.name}.graph"
        write_dot(g, dot_path)
        # noinspection SpellCheckingInspection
        os.system(f'dot -Tpdf -O \'{dot_path}\'')
    if not len(list(nx.weakly_connected_components(g))) <= 1:
        logger.warning(f"{path!s} is not a DAG: {list(nx.weakly_connected_components(g))=}")
    return g
