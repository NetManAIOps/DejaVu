from functools import reduce
from math import floor
from typing import Union, Tuple, List, Type, Callable, Dict

import dgl
import torch as th
from dgl.nn.pytorch import GATConv
from loguru import logger
from torch import nn as nn

from utils.conv_shape import conv_1d_output_shape, conv_2d_output_shape
from utils.layers import Reshape
from utils.wrapped_graph_nn import WrappedGraphNN


class SequentialModelBuilder:
    def __init__(self, input_shape: Union[th.Size, Tuple[int, ...]], debug=False):
        self._layer_cls: List[Union[Type[nn.Module], Callable[..., nn.Module]]] = []
        self._layer_kwargs: List[Dict] = []
        self._layer_args: List[List] = []
        self._output_shape: List[Tuple[int, ...]] = [input_shape]
        self._debug = debug

    @property
    def output_shapes(self):
        return self._output_shape

    @property
    def last_shape(self):
        return self._output_shape[-1]

    def build(self) -> nn.Module:
        layers = []
        if self._debug:
            logger.debug(f"=============================build layers==================================")
        for _cls, args, kwargs, shape in zip(
                self._layer_cls, self._layer_args, self._layer_kwargs, self._output_shape[1:]
        ):
            if self._debug:
                logger.debug(f"{_cls.__name__} {args=} {kwargs=} {shape=}")
            # noinspection PyArgumentList
            layers.append(_cls(*args, **kwargs))
        return nn.Sequential(
            *layers
        )

    def add_activation(self, activation='gelu') -> 'SequentialModelBuilder':
        if activation.lower() == 'relu':
            self._layer_cls.append(nn.ReLU)
        elif activation.lower() == 'gelu':
            self._layer_cls.append(nn.GELU)
        elif activation.lower() == 'softplus':
            self._layer_cls.append(nn.Softplus)
        else:
            raise RuntimeError(f"{activation=} unknown")
        self._layer_kwargs.append({})
        self._layer_args.append([])
        self._output_shape.append(self._output_shape[-1])
        return self

    def add_linear(
            self, out_features: int, bias: bool = True,
            device=None, dtype=None
    ) -> 'SequentialModelBuilder':
        self._layer_cls.append(nn.Linear)
        in_shape = self._output_shape[-1]
        self._layer_kwargs.append(
            dict(in_features=in_shape[-1], out_features=out_features, bias=bias,
                 device=device, dtype=dtype)
        )
        self._layer_args.append([])
        self._output_shape.append(in_shape[:-1] + (out_features,))
        return self

    def add_reshape(self, *args) -> 'SequentialModelBuilder':
        self._layer_cls.append(Reshape)
        self._layer_kwargs.append({})
        self._layer_args.append(list(args))
        self._output_shape.append(tuple(args))
        return self

    def add_flatten(self, start_dim: int = 1, end_dim: int = -1) -> 'SequentialModelBuilder':
        self._layer_cls.append(nn.Flatten)
        self._layer_kwargs.append(dict(start_dim=start_dim, end_dim=end_dim))
        self._layer_args.append(list())
        input_shape = self.last_shape
        if end_dim != -1:
            self._output_shape.append(
                input_shape[:start_dim] + (
                    reduce(lambda a, b: a * b, input_shape[start_dim:end_dim + 1], 1),
                ) + input_shape[end_dim + 1:]
            )
        else:
            self._output_shape.append(
                input_shape[:start_dim] + (
                    reduce(lambda a, b: a * b, input_shape[start_dim:], 1),
                )
            )
        return self

    def add_max_pool_1d(self, kernel_size, stride=None, padding=0, dilation=1) -> 'SequentialModelBuilder':
        if stride is None:
            stride = kernel_size
        self._layer_cls.append(nn.MaxPool1d)
        self._layer_kwargs.append(dict(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
        self._layer_args.append([])
        self._output_shape.append(
            self.last_shape[:-1] + (
                floor((self.last_shape[-1] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1),
            )
        )
        return self

    def add_conv_1d(
            self,
            out_channels: int,
            kernel_size: Tuple[int],
            stride=1,
            padding=0,
            bias: bool = True,
    ) -> 'SequentialModelBuilder':
        self._layer_cls.append(nn.Conv1d)
        in_shape = self._output_shape[-1]
        self._layer_kwargs.append(
            dict(
                in_channels=in_shape[-2], out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias,
            )
        )
        self._layer_args.append([])
        after_conv_size = conv_1d_output_shape(in_shape[-1], kernel_size, stride=stride, pad=padding)
        self._output_shape.append(in_shape[:-2] + (out_channels, after_conv_size))
        return self

    def add_conv_2d(
            self,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride=1,
            padding=0,
            bias: bool = True,
    ) -> 'SequentialModelBuilder':
        self._layer_cls.append(nn.Conv2d)
        in_shape = self._output_shape[-1]
        self._layer_kwargs.append(
            dict(
                in_channels=in_shape[-3], out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias,
            )
        )
        self._layer_args.append([])
        after_conv_size = conv_2d_output_shape(in_shape[-2:], kernel_size, stride=stride, pad=padding)
        self._output_shape.append(in_shape[:-3] + (out_channels,) + after_conv_size)
        return self

    def add_dropout(self, p: float = 0.5):
        self._layer_cls.append(nn.Dropout)
        self._layer_kwargs.append({})
        self._layer_args.append([p])
        self._output_shape.append(self.last_shape)
        return self

    def add_conv_transpose_1d(
            self,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            output_padding: int = 0
    ):
        self._layer_cls.append(nn.ConvTranspose1d)
        self._layer_kwargs.append(dict(
            in_channels=self.last_shape[-2],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        ))
        self._layer_args.append([])
        self._output_shape.append(
            self.last_shape[:-2] + (out_channels,) + (
                (self.last_shape[-1] - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1,
            )
        )
        return self

    def add_gat_conv_fixed_graph(
            self, graph: dgl.DGLGraph, in_feats: int, out_feats: int, num_heads: int, residual: bool,
            attn_drop: float = 0., feat_drop: float = 0.
    ):
        self._layer_cls.append(lambda *args, **kwargs: WrappedGraphNN(GATConv(*args, **kwargs), graph=graph))
        self._layer_kwargs.append(dict(
            in_feats=in_feats,
            out_feats=out_feats,
            num_heads=num_heads,
            residual=residual,
            attn_drop=attn_drop,
            feat_drop=feat_drop,
        ))
        self._layer_args.append([])
        self._output_shape.append(
            self.last_shape[:-1] + (out_feats * num_heads,)
        )
        return self
