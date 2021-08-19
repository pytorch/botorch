#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A wrapper around AquisitionFunctions to fix certain features for optimization.
This is useful e.g. for performing contextual optimization.
"""

from __future__ import annotations

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.utils import t_batch_mode_transform
from torch import Tensor
from torch.nn import Module


class ProximalAcquisitionFunction(AnalyticAcquisitionFunction):
    """A wrapper around AquisitionFunctions to add proximal weighting of the acquisition function. Acquisition
    function is weighted via a squared exponential centered at the last training point with varying lengthscales
    corresponding to 'proximal_weights'. Can only be used with single batch analytical acquisition functions.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)  # d = 2
        >>> EI = ExpectedImprovement(model, best_f=0.0)
        >>> proximal_weights = torch.ones(d)
        >>> EI_proximal = ProximalAcquisitionFunction(EI, proximal_weights)
        >>> eip = EI_proximal(test_X)  # d' = 3
    """

    def __init__(
        self,
        acq_function: AnalyticAcquisitionFunction,
        proximal_weights: Tensor,
    ) -> None:
        r"""Derived Acquisition Function by fixing a subset of input features.

        Args:
            acq_function: The base acquisition function, operating on input
                tensors `X_full` of feature dimension `d`.
            proximal_weights: Tensor used to bias locality along each axis, should be in the shape of ('d',).

        """
        Module.__init__(self)
        self.acq_func = acq_function

        self.register_buffer("proximal_weights", proximal_weights)

        # check to make sure that weights match the training data shape
        assert (
            self.proximal_weights.shape[0]
            == self.acq_func.model.train_inputs[0][-1].shape[-1]
        )

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor):
        r"""Evaluate base acquisition function with proximal weighting.

        Args:
            X: Input tensor of feature dimension `d` .

        Returns:
            Base acquisition function evaluated on tensor `X` multiplied by a squared exponenital centered at the last
            training point and is weighted according to the weighting of each dimension.
        """
        d = self.proximal_weights.shape[0]
        last_X = self.acq_func.model.train_inputs[0][-1].reshape(1, 1, -1)
        _unbroadcasted_scale_tril = torch.diag(
            torch.sqrt(self.proximal_weights)
        ).reshape(1, 1, d, d)

        diff = X - last_X
        M = _batch_mahalanobis(_unbroadcasted_scale_tril, diff)
        proximal_acq_weight = torch.exp(-0.5 * M)
        return self.acq_func(X) * proximal_acq_weight


def _batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for (sL, sx) in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (
        list(range(outer_batch_dims))
        + list(range(outer_batch_dims, new_batch_dims, 2))
        + list(range(outer_batch_dims + 1, new_batch_dims, 2))
        + [new_batch_dims]
    )
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = (
        torch.triangular_solve(flat_x_swap, flat_L, upper=False)[0].pow(2).sum(-2)
    )  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)
