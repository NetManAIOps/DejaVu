import random
from typing import Dict, Tuple

import dgl
import torch as th
from loguru import logger

from failure_dependency_graph import FDG


class IncompleteFDGFactory:
    """
    Get a incomplete FDG for a given failure_id.
    The incomplete FDG is always the same with the same failure_id
    """
    _instances: Dict[Tuple, 'IncompleteFDGFactory'] = {}

    def __new__(cls, fdg: FDG, drop_edges_fraction: float):
        args = (fdg, drop_edges_fraction)
        if args not in cls._instances:
            cls._instances[args] = super().__new__(cls)
        return cls._instances[args]

    def __init__(self, fdg: FDG, drop_edges_fraction: float):
        self._fdg = fdg
        self._drop_edges_fraction = drop_edges_fraction
        self._cached_dgl_graphs: Dict[int, dgl.DGLGraph] = {}

    def __call__(self, failure_id: int):
        if failure_id not in self._cached_dgl_graphs:
            g: dgl.DGLGraph = self._fdg.homo_graph_at(failure_id).clone()
            dropped_edges = th.tensor([_ for _ in g.edges('eid') if random.random() < self._drop_edges_fraction])
            if len(dropped_edges) == g.number_of_edges():
                dropped_edges = dropped_edges[:-1]
                logger.warning("At least one edge should be preserved")
            if len(dropped_edges) > 0:
                logger.info(
                    f"Generate incomplete FDG for {failure_id=:4d}, {self._drop_edges_fraction=}, "
                    f"#original edges={g.number_of_edges():4d} #dropped edges={len(dropped_edges):4d}"
                )
                g.remove_edges(dropped_edges, store_ids=True)
            g.ndata[dgl.NID] = th.arange(self._fdg.n_failure_instances, device=g.device)
            g: dgl.DGLGraph = dgl.add_self_loop(dgl.to_bidirected(g, copy_ndata=True))
            self._cached_dgl_graphs[failure_id] = g
        return self._cached_dgl_graphs[failure_id]
