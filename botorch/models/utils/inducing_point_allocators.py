#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
References

.. [burt2020svgp]
    David R. Burt and Carl Edward Rasmussen and Mark van der Wilk,
    Convergence of Sparse Variational Inference in Gaussian Process Regression,
    Journal of Machine Learning Research, 2020,
    http://jmlr.org/papers/v21/19-1015.html.

.. [chen2018dpp]
    Laming Chen and Guoxin Zhang and Hanning Zhou, Fast greedy MAP inference
    for determinantal point process to improve recommendation diversity,
    Proceedings of the 32nd International Conference on Neural Information
    Processing Systems, 2018, https://arxiv.org/abs/1709.05135.

.. [moss2023ipa]
    Henry B. Moss and Sebastian W. Ober and Victor Picheny,
    Inducing Point Allocation for Sparse Gaussian Processes
    in High-Throughput Bayesian Optimization,Proceedings of
    the 25th International Conference on Artificial Intelligence
    and Statistics, 2023, https://arxiv.org/pdf/2301.10123.pdf.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Union

import torch
from botorch.models.model import Model

from botorch.utils.probability.utils import ndtr as Phi, phi
from gpytorch.module import Module
from linear_operator.operators import LinearOperator
from torch import Tensor

NEG_INF = -(torch.tensor(float("inf")))


class InducingPointAllocator(ABC):
    r"""
    This class provides functionality to initialize the inducing point locations
    of an inducing point-based model, e.g. a `SingleTaskVariationalGP`.
    """

    @abstractmethod
    def allocate_inducing_points(
        inputs: Tensor,
        covar_module: Module,
        num_inducing: int,
        input_batch_shape: torch.Size,
    ) -> Tensor:
        """
        Initialize the `num_inducing` inducing point locations according to a
        specific initialization strategy.

        Args:
            inputs: A (*batch_shape, n, d)-dim input data tensor.
            covar_module: GPyTorch Module returning a LinearOperator kernel matrix.
            num_inducing: The maximun number (m) of inducing points (m <= n).
            input_batch_shape: The non-task-related batch shape.

        Returns:
            A (*batch_shape, m, d)-dim tensor of inducing point locations.
        """

        pass

    def _allocate_inducing_points(
        self,
        inputs: Tensor,
        covar_module: Module,
        num_inducing: int,
        input_batch_shape: torch.Size,
        quality_function: Callable[[Tensor], Tensor],
    ) -> Tensor:
        r"""
        Private method to allow inducing point allocators to support
        multi-task models and models with batched inputs.

        Args:
            inputs: A (*batch_shape, n, d)-dim input data tensor.
            covar_module: GPyTorch Module returning a LinearOperator kernel matrix.
            num_inducing: The maximun number (m) of inducing points (m <= n).
            input_batch_shape: The non-task-related batch shape.
            quality_function: A callable mapping `inputs` to scores representing
                the utility of allocating an inducing point to each input (of
                shape [n] ).

        Returns:
            A (*batch_shape, m, d)-dim tensor of inducing point locations.
        """

        train_train_kernel = covar_module(inputs).evaluate_kernel()

        # base case
        if train_train_kernel.ndimension() == 2:
            quality_scores = quality_function(inputs)
            inducing_points = _pivoted_cholesky_init(
                train_inputs=inputs,
                kernel_matrix=train_train_kernel,
                max_length=num_inducing,
                quality_scores=quality_scores,
            )
        # multi-task case
        elif train_train_kernel.ndimension() == 3 and len(input_batch_shape) == 0:
            quality_scores = quality_function(inputs)
            input_element = inputs[0] if inputs.ndimension() == 3 else inputs
            kernel_element = train_train_kernel[0]
            quality_scores = quality_function(input_element)
            inducing_points = _pivoted_cholesky_init(
                train_inputs=input_element,
                kernel_matrix=kernel_element,
                max_length=num_inducing,
                quality_scores=quality_scores,
            )
        # batched input cases
        else:
            batched_inputs = (
                inputs.expand(*input_batch_shape, -1, -1)
                if inputs.ndimension() == 2
                else inputs
            )
            reshaped_inputs = batched_inputs.flatten(end_dim=-3)
            inducing_points = []
            for input_element in reshaped_inputs:
                # the extra kernel evals are a little wasteful but make it
                # easier to infer the task batch size
                kernel_element = covar_module(input_element).evaluate_kernel()
                # handle extra task batch dimension
                kernel_element = (
                    kernel_element[0]
                    if kernel_element.ndimension() == 3
                    else kernel_element
                )
                quality_scores = quality_function(input_element)
                inducing_points.append(
                    _pivoted_cholesky_init(
                        train_inputs=input_element,
                        kernel_matrix=kernel_element,
                        max_length=num_inducing,
                        quality_scores=quality_scores,
                    )
                )
            inducing_points = torch.stack(inducing_points).view(
                *input_batch_shape, num_inducing, -1
            )

        return inducing_points


class QualityFunction(ABC):
    """
    A function that scores inputs with respect
    to a specific criterion.
    """

    @abstractmethod
    def __call__(inputs: Tensor) -> Tensor:  # [n, d] -> [n]
        """
        Args:
            inputs: inputs (of shape n x d)
        Returns:
            A tensor of quality scores for each input, of shape [n]
        """

        pass


class UnitQualityFunction(QualityFunction):
    """
    A function returning ones for each element. Using this quality function
    for inducing point allocation corresponds to allocating inducing points
    with the sole aim of minimizing predictive variance, i.e. the approach
    of [burt202svgp]_.
    """

    @torch.no_grad()
    def __call__(self, inputs:Tensor) -> Tensor: # [n, d]-> [n]
        """
        Args:
            inputs: inputs (of shape n x d)
        Returns:
            A tensor of ones for each input, of shape [n]
        """
        return torch.ones([inputs.shape[0]], dtype=inputs.dtype)



class ExpectedImprovementQualityFunction(QualityFunction):
    """
    A function measuring the quality of input points as their expected
    improvement with respect to a conservative baseline. Expectations
    are according to the model from the previous BO step. See [moss2023ipa]_
    for details and justification.
    """

    def __init__(self, model:Model, maximize: bool):
        r"""
        Args:
            model: The model fitted during the previous BO step. For now, this
                must be a single task model (i.e. num_outputs=1).
            maximize: Set True if we are performing function maximization, else
                set False.
        """
        if model.num_outputs != 1:
            raise NotImplementedError(
                "Multi-output models are currently not supported. "
            )
        self._model = model
        self._maximize = maximize


    @torch.no_grad()
    def __call__(self, inputs: Tensor) -> Tensor:  # [n, d] -> [n]
        """
        Args:
            inputs: inputs (of shape n x d)
        Returns:
            A tensor of quality scores for each input, of shape [n]
        """

        posterior = self._model.posterior(inputs)
        mean = posterior.mean.squeeze(-2).squeeze(
            -1
        )  # removing redundant dimensions
        sigma = posterior.variance.clamp_min(1e-12).sqrt().view(mean.shape)

        best_f = torch.max(mean) if self._maximize else torch.min(mean)
        u = (mean - best_f) / sigma if self._maximize else -(mean - best_f) / sigma
        return sigma * (phi(u) + u * Phi(u))




class GreedyVarianceReduction(InducingPointAllocator):
    r"""
    The inducing point allocator proposed by [burt2020svgp]_, that
    greedily chooses inducing point locations with maximal (conditional)
    predictive variance.
    """

    def allocate_inducing_points(
        self,
        inputs: Tensor,
        covar_module: Module,
        num_inducing: int,
        input_batch_shape: torch.Size,
    ) -> Tensor:
        """
        Greedily initialize `num_inducing` inducing points following [burt2020svgp]_.

        Args:
            inputs: A (*batch_shape, n, d)-dim input data tensor.
            covar_module: GPyTorch Module returning a LinearOperator kernel matrix.
            num_inducing: The maximun number (m) of inducing points (m <= n).
            input_batch_shape: The non-task-related batch shape.

        Returns:
            A (*batch_shape, m, d)-dim tensor of inducing point locations.
        """

        return self._allocate_inducing_points(
            inputs, covar_module, num_inducing, input_batch_shape, UnitQualityFunction()
        )


class GreedyImprovementReduction(InducingPointAllocator):
    r"""
    An inducing point allocator that greedily chooses inducing points with large
    predictive variance and that are in promising regions of the search
    space (according to the model form the previous BO step), see :cite:`moss2023IPA`.
    """

    def __init__(self, model: Model, maximize: bool):
        r"""

        Args:
            model: The model fitted during the previous BO step.
            maximize: Set True if we are performing function maximization, else
                set False.
        """
        self._model = model
        self._maximize = maximize

    def allocate_inducing_points(
        self,
        inputs: Tensor,
        covar_module: Module,
        num_inducing: int,
        input_batch_shape: torch.Size,
    ) -> Tensor:
        """
        Greedily initialize the `num_inducing` inducing points following the IMP-DPP
        strategy of [moss2023ipa]_.

        Args:
            inputs: A (*batch_shape, n, d)-dim input data tensor.
            covar_module: GPyTorch Module returning a LinearOperator kernel matrix.
            num_inducing: The maximun number (m) of inducing points (m <= n).
            input_batch_shape: The non-task-related batch shape.

        Returns:
            A (*batch_shape, m, d)-dim tensor of inducing point locations.
        """

        return self._allocate_inducing_points(
            inputs,
            covar_module,
            num_inducing,
            input_batch_shape,
            ExpectedImprovementQualityFunction(self._model,self._maximize),
        )


def _pivoted_cholesky_init(
    train_inputs: Tensor,
    kernel_matrix: Union[Tensor, LinearOperator],
    max_length: int,
    quality_scores: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    r"""
    A pivoted Cholesky initialization method for the inducing points,
    originally proposed in [burt2020svgp]_ with the algorithm itself coming from
    [chen2018dpp]_. Code is a PyTorch version from [chen2018dpp]_, based on
    https://github.com/laming-chen/fast-map-dpp/blob/master/dpp.py but with a small
    modification to allow the underlying DPP to be defined through its diversity-quality
    decomposition,as discussed by [moss2023ipa]_. This method returns a greedy
    approximation of the MAP estimate of the specified DPP, i.e. its returns a
    set of points that are highly diverse (according to the provided kernel_matrix)
    and have high quality (according to the provided quality_scores).

    Args:
        train_inputs: training inputs (of shape n x d)
        kernel_matrix: kernel matrix on the training
            inputs
        max_length: number of inducing points to initialize
        quality_scores: scores representing the quality of each candidate
            input (of shape [n])
        epsilon: numerical jitter for stability.

    Returns:
        max_length x d tensor of the training inputs corresponding to the top
        max_length pivots of the training kernel matrix
    """

    # this is numerically equivalent to iteratively performing a pivoted cholesky
    # while storing the diagonal pivots at each iteration
    # TODO: use gpytorch's pivoted cholesky instead once that gets an exposed list
    # TODO: ensure this works in batch mode, which it does not currently.

    # todo test for shape of quality function

    if quality_scores.shape[0] != train_inputs.shape[0]:
        raise ValueError(
            "_pivoted_cholesky_init requires a quality score for each of train_inputs"
        )

    item_size = kernel_matrix.shape[-2]
    cis = torch.zeros(
        (max_length, item_size), device=kernel_matrix.device, dtype=kernel_matrix.dtype
    )
    di2s = kernel_matrix.diag()
    scores = di2s * (quality_scores**2)
    selected_items = []
    selected_item = torch.argmax(scores)
    selected_items.append(selected_item)

    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        elements = kernel_matrix[..., selected_item, :]
        eis = (elements - torch.matmul(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s = di2s - eis.pow(2.0)
        di2s[selected_item] = NEG_INF
        scores = di2s * (quality_scores**2)
        selected_item = torch.argmax(scores)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)

    ind_points = train_inputs[torch.stack(selected_items)]

    return ind_points[:max_length,:]
