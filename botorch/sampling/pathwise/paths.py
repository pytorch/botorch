#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Callable, Optional, Union

from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.pathwise.features import FeatureMap
from botorch.sampling.pathwise.utils import (
    TInputTransform,
    TOutputTransform,
    TransformedModuleMixin,
)
from torch import Tensor
from torch.nn import Module, ModuleDict, ModuleList, Parameter


class SamplePath(ABC, TransformedModuleMixin, Module):
    r"""Abstract base class for Botorch sample paths."""


class PathDict(SamplePath):
    r"""A dictionary of SamplePaths."""

    def __init__(
        self,
        paths: Optional[Mapping[str, SamplePath]] = None,
        join: Optional[Callable[[list[Tensor]], Tensor]] = None,
        input_transform: Optional[TInputTransform] = None,
        output_transform: Optional[TOutputTransform] = None,
    ) -> None:
        r"""Initializes a PathDict instance.

        Args:
            paths: An optional mapping of strings to sample paths.
            join: An optional callable used to combine each path's outputs.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
        """
        if join is None and output_transform is not None:
            raise UnsupportedError("Output transforms must be preceded by a join rule.")

        super().__init__()
        self.join = join
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.paths = (
            paths
            if isinstance(paths, ModuleDict)
            else ModuleDict({} if paths is None else paths)
        )

    def forward(self, x: Tensor, **kwargs: Any) -> Union[Tensor, dict[str, Tensor]]:
        out = [path(x, **kwargs) for path in self.paths.values()]
        return dict(zip(self.paths, out)) if self.join is None else self.join(out)

    def items(self) -> Iterable[tuple[str, SamplePath]]:
        return self.paths.items()

    def keys(self) -> Iterable[str]:
        return self.paths.keys()

    def values(self) -> Iterable[SamplePath]:
        return self.paths.values()

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[SamplePath]:
        yield from self.paths

    def __delitem__(self, key: str) -> None:
        del self.paths[key]

    def __getitem__(self, key: str) -> SamplePath:
        return self.paths[key]

    def __setitem__(self, key: str, val: SamplePath) -> None:
        self.paths[key] = val


class PathList(SamplePath):
    r"""A list of SamplePaths."""

    def __init__(
        self,
        paths: Optional[Iterable[SamplePath]] = None,
        join: Optional[Callable[[list[Tensor]], Tensor]] = None,
        input_transform: Optional[TInputTransform] = None,
        output_transform: Optional[TOutputTransform] = None,
    ) -> None:
        r"""Initializes a PathList instance.

        Args:
            paths: An optional iterable of sample paths.
            join: An optional callable used to combine each path's outputs.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
        """

        if join is None and output_transform is not None:
            raise UnsupportedError("Output transforms must be preceded by a join rule.")

        super().__init__()
        self.join = join
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.paths = (
            paths
            if isinstance(paths, ModuleList)
            else ModuleList({} if paths is None else paths)
        )

    def forward(self, x: Tensor, **kwargs: Any) -> Union[Tensor, list[Tensor]]:
        out = [path(x, **kwargs) for path in self.paths]
        return out if self.join is None else self.join(out)

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[SamplePath]:
        yield from self.paths

    def __delitem__(self, key: int) -> None:
        del self.paths[key]

    def __getitem__(self, key: int) -> SamplePath:
        return self.paths[key]

    def __setitem__(self, key: int, val: SamplePath) -> None:
        self.paths[key] = val


class GeneralizedLinearPath(SamplePath):
    r"""A sample path in the form of a generalized linear model."""

    def __init__(
        self,
        feature_map: FeatureMap,
        weight: Union[Parameter, Tensor],
        bias_module: Optional[Module] = None,
        input_transform: Optional[TInputTransform] = None,
        output_transform: Optional[TOutputTransform] = None,
    ):
        r"""Initializes a GeneralizedLinearPath instance.

        .. code-block:: text

            path(x) = output_transform(bias_module(z) + feature_map(z)^T weight),
            where z = input_transform(x).

        Args:
            feature_map: A map used to featurize the module's inputs.
            weight: A tensor of weights used to combine input features.
            bias_module: An optional module used to define additive offsets.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
        """
        super().__init__()
        self.feature_map = feature_map
        if not isinstance(weight, Parameter):
            self.register_buffer("weight", weight)
        self.weight = weight
        self.bias_module = bias_module
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        feat = self.feature_map(x, **kwargs)
        out = (feat @ self.weight.unsqueeze(-1)).squeeze(-1)
        return out if self.bias_module is None else out + self.bias_module(x)
