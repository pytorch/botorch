# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from botorch.acquisition.analytic import AcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.sampling.pathwise.posterior_samplers import get_matheron_path_model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


BATCH_SIZE_CHANGE_ERROR = """The batch size of PathwiseThompsonSampling should \
not change during a forward pass - was {}, now {}. Please re-initialize the \
acquisition if you want to change the batch size."""


class PathwiseThompsonSampling(AcquisitionFunction):
    r"""Single-outcome Thompson Sampling packaged as an (analytic)
    acquisition function. Querying the acquisition function gives the summed
    values of one or more draws from a pathwise drawn posterior sample, and thus
    it maximization yields one (or multiple) Thompson sample(s).

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> TS = PathwiseThompsonSampling(model)
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> None:
        r"""Single-outcome TS.

        Args:
            model: A fitted GP model.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
        """
        if model._is_fully_bayesian:
            raise NotImplementedError(
                "PathwiseThompsonSampling is not supported for fully Bayesian models",
            )

        super().__init__(model=model)
        self.batch_size: Optional[int] = None

    def redraw(self) -> None:
        self.samples = get_matheron_path_model(
            model=self.model, sample_shape=torch.Size([self.batch_size])
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the pathwise posterior sample draws on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk) x [num_models for fully bayesian]`-dim tensor of
            evaluations on the posterior sample draws.
        """
        batch_size = X.shape[-2]
        q_dim = -2

        # batch_shape x q x 1 x d
        X = X.unsqueeze(-2)
        if self.batch_size is None:
            self.batch_size = batch_size
            self.redraw()
        elif self.batch_size != batch_size:
            raise ValueError(
                BATCH_SIZE_CHANGE_ERROR.format(self.batch_size, batch_size)
            )

        # posterior_values.shape post-squeeze:
        # batch_shape x q x m
        posterior_values = self.samples(X).squeeze(-2)
        # sum over batch dim and squeeze num_objectives dim (-1)
        return posterior_values.sum(q_dim).squeeze(-1)
