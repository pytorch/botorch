#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Containers to standardize inputs into models and acquisition functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from botorch.exceptions.errors import UnsupportedError
from torch import Tensor


@dataclass
class TrainingData:
    r"""Standardized container of model training data for models.

    Properties:
        Xs: A list of tensors, each of shape `batch_shape x n_i x d`,
            where `n_i` is the number of training inputs for the i-th model.
        Ys: A list of tensors, each of shape `batch_shape x n_i x 1`,
            where `n_i` is the number of training observations for the i-th
            (single-output) model.
        Yvars: A list of tensors, each of shape `batch_shape x n_i x 1`,
            where `n_i` is the number of training observations of the
            observation noise for the i-th  (single-output) model.
            If `None`, the observation noise level is unobserved.
    """

    Xs: List[Tensor]  # `batch_shape x n_i x 1`
    Ys: List[Tensor]  # `batch_shape x n_i x 1`
    Yvars: Optional[List[Tensor]] = None  # `batch_shape x n_i x 1`

    def __post_init__(self):
        self._is_block_design = all(torch.equal(X, self.Xs[0]) for X in self.Xs[1:])

    @classmethod
    def from_block_design(cls, X: Tensor, Y: Tensor, Yvar: Optional[Tensor] = None):
        r"""Construct a TrainingData object from a block design description.

        Args:
            X: A `batch_shape x n x d` tensor of training points (shared across
                all outcomes).
            Y: A `batch_shape x n x m` tensor of training observations.
            Yvar: A `batch_shape x n x m` tensor of training noise variance
                observations, or `None`.

        Returns:
            The `TrainingData` object (with `is_block_design=True`).
        """
        return cls(
            Xs=[X for _ in range(Y.shape[-1])],
            Ys=list(torch.split(Y, 1, dim=-1)),
            Yvars=None if Yvar is None else list(torch.split(Yvar, 1, dim=-1)),
        )

    @property
    def is_block_design(self) -> bool:
        r"""Indicates whether training data is a "block design".

        Block designs are designs in which all outcomes are observed
        at the same training inputs.
        """
        return self._is_block_design

    @property
    def X(self) -> Tensor:
        r"""The training inputs (block-design only).

        This raises an `UnsupportedError` in the non-block-design case.
        """
        if not self.is_block_design:
            raise UnsupportedError
        return self.Xs[0]

    @property
    def Y(self) -> Tensor:
        r"""The training observations (block-design only).

        This raises an `UnsupportedError` in the non-block-design case.
        """
        if not self.is_block_design:
            raise UnsupportedError
        return torch.cat(self.Ys, dim=-1)

    @property
    def Yvar(self) -> Optional[List[Tensor]]:
        r"""The training observations's noise variance (block-design only).

        This raises an `UnsupportedError` in the non-block-design case.
        """
        if self.Yvars is not None:
            if not self.is_block_design:
                raise UnsupportedError
            return torch.cat(self.Yvars, dim=-1)

    def __eq__(self, other: TrainingData) -> bool:
        # Check for `None` Yvars and unequal attribute lengths upfront.
        if self.Yvars is None or other.Yvars is None:
            if not (self.Yvars is other.Yvars is None):
                return False
        else:
            if len(self.Yvars) != len(other.Yvars):
                return False

        if len(self.Xs) != len(other.Xs) or len(self.Ys) != len(other.Ys):
            return False

        return (  # Deep-check equality of attributes.
            all(
                torch.equal(self_X, other_X)
                for self_X, other_X in zip(self.Xs, other.Xs)
            )
            and all(
                torch.equal(self_Y, other_Y)
                for self_Y, other_Y in zip(self.Ys, other.Ys)
            )
            and (
                self.Yvars is other.Yvars is None
                or all(
                    torch.equal(self_Yvar, other_Yvar)
                    for self_Yvar, other_Yvar in zip(self.Yvars, other.Yvars)
                )
            )
        )
