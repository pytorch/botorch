#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch
from botorch.models.model import Model
from botorch.utils.sampling import manual_seed
from torch import Tensor
from torch.nn import Module


class GPDraw(Module):
    r"""Convenience wrapper for sampling a function from a GP prior.

    This wrapper implicitly defines the GP sample as a self-updating function by keeping
    track of the evaluated points and respective base samples used during the
    evaluation.

    This does not yet support multi-output models.
    """

    def __init__(self, model: Model, seed: Optional[int] = None) -> None:
        r"""Construct a GP function sampler.

        Args:
            model: The Model defining the GP prior.
        """
        super().__init__()
        self._model = deepcopy(model)
        seed = torch.tensor(
            seed if seed is not None else torch.randint(0, 1000000, (1,)).item()
        )
        self.register_buffer("_seed", seed)

    @property
    def Xs(self) -> Tensor:
        """A `(batch_shape) x n_eval x d`-dim tensor of locations at which the GP was
        evaluated (or `None` if the sample has never been evaluated).
        """
        try:
            return self._Xs
        except AttributeError:
            return None

    @property
    def Ys(self) -> Tensor:
        """A `(batch_shape) x n_eval x d`-dim tensor of associated function values (or
        `None` if the sample has never been evaluated).
        """
        try:
            return self._Ys
        except AttributeError:
            return None

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the GP sample function at a set of points X.

        Args:
            X: A `batch_shape x n x d`-dim tensor of points

        Returns:
            The value of the GP sample at the `n` points.
        """
        if self.Xs is None:
            X_eval = X  # first time, no previous evaluation points
        else:
            X_eval = torch.cat([self.Xs, X], dim=-2)
        posterior = self._model.posterior(X=X_eval)
        event_shape = posterior.event_shape
        # re-use old samples
        bs_shape = event_shape[:-2] + X.shape[-2:-1] + event_shape[-1:]
        with manual_seed(seed=int(self._seed)):
            new_base_samples = torch.randn(bs_shape, device=X.device, dtype=X.dtype)
        seed = self._seed + 1
        if self.Xs is None:
            base_samples = new_base_samples
        else:
            base_samples = torch.cat([self._base_samples, new_base_samples], dim=-2)
        # TODO: Deduplicate repeated evaluations / deal with numerical degeneracies
        # that could lead to non-determinsitic evaluations. We could use SVD- or
        # eigendecomposition-based sampling, but we probably don't want to use this
        # by default for performance reasonse.
        Ys = posterior.rsample(torch.Size(), base_samples=base_samples)
        self.register_buffer("_Xs", X_eval)
        self.register_buffer("_Ys", Ys)
        self.register_buffer("_seed", seed)
        self.register_buffer("_base_samples", base_samples)
        return self.Ys[..., -(X.size(-2)) :, :]
