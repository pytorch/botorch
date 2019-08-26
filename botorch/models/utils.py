#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
Utiltiy functions for models.
"""

import warnings
from typing import List, Optional, Tuple, Any

import torch
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from torch import Tensor

from ..exceptions import InputDataError, InputDataWarning

from .models import SingleTaskGP
from .models import HeteroskedasticSingleTaskGP
from ..sampling import IIDNormalSampler
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.module import Module

from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.rbf_kernel import RBFKernel



def _make_X_full(X: Tensor, output_indices: List[int], tf: int) -> Tensor:
    r"""Helper to construct input tensor with task indices.

    Args:
        X: The raw input tensor (without task information).
        output_indices: The output indices to generate (passed in via `posterior`).
        tf: The task feature index.

    Returns:
        Tensor: The full input tensor for the multi-task model, including task
            indices.
    """
    index_shape = X.shape[:-1] + torch.Size([1])
    indexers = (
        torch.full(index_shape, fill_value=i, device=X.device, dtype=X.dtype)
        for i in output_indices
    )
    X_l, X_r = X[..., :tf], X[..., tf:]
    return torch.cat(
        [torch.cat([X_l, indexer, X_r], dim=-1) for indexer in indexers], dim=0
    )


def multioutput_to_batch_mode_transform(
    train_X: Tensor,
    train_Y: Tensor,
    num_outputs: int,
    train_Yvar: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    r"""Transforms training inputs for a multi-output model.

    Used for multi-output models that internally are represented by a
    batched single output model, where each output is modeled as an
    independent batch.

    Args:
        train_X: A `n x d` or `input_batch_shape x n x d` (batch mode) tensor of
            training features.
        train_Y: A `n x m` or `target_batch_shape x n x m` (batch mode) tensor of
            training observations.
        num_outputs: number of outputs
        train_Yvar: A `n x m` or `target_batch_shape x n x m` tensor of observed
            measurement noise.

    Returns:
        3-element tuple containing

        - A `input_batch_shape x m x n x d` tensor of training features.
        - A `target_batch_shape x m x n` tensor of training observations.
        - A `target_batch_shape x m x n` tensor observed measurement noise.
    """
    # make train_Y `batch_shape x m x n`
    train_Y = train_Y.transpose(-1, -2)
    # expand train_X to `batch_shape x m x n x d`
    train_X = train_X.unsqueeze(-3).expand(
        train_X.shape[:-2] + torch.Size([num_outputs]) + train_X.shape[-2:]
    )
    if train_Yvar is not None:
        # make train_Yvar `batch_shape x m x n`
        train_Yvar = train_Yvar.transpose(-1, -2)
    return train_X, train_Y, train_Yvar


def add_output_dim(X: Tensor, original_batch_shape: torch.Size) -> Tuple[Tensor, int]:
    r"""Insert the output dimension at the correct location.

    The trailing batch dimensions of X must match the original batch dimensions
    of the training inputs, but can also include extra batch dimensions.

    Args:
        X: A `(new_batch_shape) x (original_batch_shape) x n x d` tensor of features.
        original_batch_shape: the batch shape of the model's training inputs.

    Returns:
        2-element tuple containing

        - A `(new_batch_shape) x (original_batch_shape) x m x n x d` tensor of
        features.
        - The index corresponding to the output dimension.
    """
    X_batch_shape = X.shape[:-2]
    if len(X_batch_shape) > 0 and len(original_batch_shape) > 0:
        # check that X_batch_shape supports broadcasting or augments
        # original_batch_shape with extra batch dims
        error_msg = (
            "The trailing batch dimensions of X must match the trailing "
            "batch dimensions of the training inputs."
        )
        _mul_broadcast_shape(X_batch_shape, original_batch_shape, error_msg=error_msg)
    # insert `m` dimension
    X = X.unsqueeze(-3)
    output_dim_idx = max(len(original_batch_shape), len(X_batch_shape))
    return X, output_dim_idx


def check_no_nans(Z: Tensor) -> None:
    r"""Check that tensor does not contain NaN values.

    Raises an InputDataError if `Z` contains NaN values.

    Args:
        Z: The input tensor.
    """
    if torch.any(torch.isnan(Z)).item():
        raise InputDataError("Input data contains NaN values.")


def check_min_max_scaling(
    X: Tensor, strict: bool = False, atol: float = 1e-2, raise_on_fail: bool = False
) -> None:
    r"""Check that tensor is normalized to the unit cube.

    Args:
        X: A `batch_shape x n x d` input tensor. Typically the training inputs
            of a model.
        strict: If True, require `X` to be scaled to the unit cube (rather than
            just to be contained within the unit cube).
        atol: The tolerance for the boundary check. Only used if `strict=True`.
        raise_on_fail: If True, raise an exception instead of a warning.
    """
    with torch.no_grad():
        Xmin, Xmax = torch.min(X, dim=-1)[0], torch.max(X, dim=-1)[0]
        msg = None
        if strict and max(torch.abs(Xmin).max(), torch.abs(Xmax - 1).max()) > atol:
            msg = "scaled"
        if torch.any(Xmin < -atol) or torch.any(Xmax > 1 + atol):
            msg = "contained"
        if msg is not None:
            msg = (
                f"Input data is not {msg} to the unit cube. "
                "Please consider min-max scaling the input data."
            )
            if raise_on_fail:
                raise InputDataError(msg)
            warnings.warn(msg, InputDataWarning)


def check_standardization(
    Y: Tensor,
    atol_mean: float = 1e-2,
    atol_std: float = 1e-2,
    raise_on_fail: bool = False,
) -> None:
    r"""Check that tensor is standardized (zero mean, unit variance).

    Args:
        Y: The input tensor of shape `batch_shape x n x m`. Typically the
            train targets of a model. Standardization is checked across the
            `n`-dimension.
        atol_mean: The tolerance for the mean check.
        atol_std: The tolerance for the std check.
        raise_on_fail: If True, raise an exception instead of a warning.
    """
    with torch.no_grad():
        Ymean, Ystd = torch.mean(Y, dim=-2), torch.std(Y, dim=-2)
        if torch.abs(Ymean).max() > atol_mean or torch.abs(Ystd - 1).max() > atol_std:
            msg = (
                "Input data is not standardized. Please consider scaling the "
                "input to zero mean and unit variance."
            )
            if raise_on_fail:
                raise InputDataError(msg)
            warnings.warn(msg, InputDataWarning)


def fit_most_likely_HeteroskedasticGP(
    train_X: Tensor,
    train_Y: Tensor,
    covar_module: Optional[Module] = None,
    num_var_samples: int = 100,
    max_iter: int = 10,
    atol_mean: float = 1e-04,
    atol_var: float = 1e-04,
 ) -> HeteroskedasticSingleTaskGP:
    r"""Fit the Most Likely Heteroskedastic GP.

    The original algorithm is described in 
    http://people.csail.mit.edu/kersting/papers/kersting07icml_mlHetGP.pdf

    Args:
        train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
            features.
        train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
            training observations.  
        covar_module: The covariance (kernel) matrix for the initial homoskedastic GP.
            If omitted, use the RBFKernel.
        num_var_samples: Number of samples to draw from posterior when estimating noise.
        max_iter: Maximum number of iterations used when fitting the model.
        atol_mean: The tolerance for the mean check.
        atol_std: The tolerance for the var check.
    Returns:
        HeteroskedasticSingleTaskGP Model fit using the "most-likely" procedure.
    """

    if covar_module is None:
        covar_module = ScaleKernel(RBFKernel())

    # CANNOT CHECK RIGHT NOW BECAUSE NEED TO FIRST ADD BATCH DIMENSION
    # check to see if input Tensors are normalized and standardized
    # check_min_max_scaling(train_X)
    # check_standardization(train_Y)

    # fit initial homoskedastic model used to estimate noise levels
    homo_model = SingleTaskGP(train_X=train_X, train_Y=train_Y,
                              covar_module=covar_module)
    homo_model.likelihood.noise_covar.register_constraint("raw_noise",
                                                          GreaterThan(1e-5))
    homo_mll = gpytorch.mlls.ExactMarginalLogLikelihood(homo_model.likelihood,
                                                        homo_model)
    botorch.fit.fit_gpytorch_model(homo_mll)

    # get estimates of noise 
    homo_mll.eval()
    with torch.no_grad():
        homo_posterior = homo_mll.model.posterior(train_X.clone())
        homo_predictive_posterior = homo_mll.model.posterior(train_X.clone(),
                                                             observation_noise=True)
    sampler = IIDNormalSampler(num_samples=num_var_samples, resample=True)
    predictive_samples = sampler(homo_predictive_posterior)
    observed_var = 0.5 * ((predictive_samples - train_Y.reshape(-1,1))**2).mean(dim=0)

    # save mean and variance to check if they change later
    saved_mean = homo_posterior.mean
    saved_var = homo_posterior.variance

    for i in range(max_iter): 

        # now train hetero model using computed noise
        hetero_model = HeteroskedasticSingleTaskGP(train_X=train_X, train_Y=train_Y,
                                                   train_Yvar=observed_var)
        hetero_mll = gpytorch.mlls.ExactMarginalLogLikelihood(hetero_model.likelihood,
                                                              hetero_model)
        try:
            botorch.fit.fit_gpytorch_model(hetero_mll)
        except Exception as e:
            msg = f'Fitting failed on iteration {i}. Returning the current MLL'
            warnings.warn(msg, e)
            return saved_hetero_mll

        hetero_mll.eval()
        with torch.no_grad():
            hetero_posterior = hetero_mll.model.posterior(train_X.clone())
            hetero_predictive_posterior = hetero_mll.model.posterior(train_X.clone(),
                                                                     observation_noise=True)
           
        new_mean = hetero_posterior.mean
        new_var = hetero_posterior.variance
        
        mean_equality = torch.all(torch.lt(torch.abs(torch.add(saved_mean, -new_mean)), atol_mean))
        max_change_in_means = torch.max(torch.abs(torch.add(saved_mean, -new_mean)))

        var_equality = torch.all(torch.lt(torch.abs(torch.add(saved_var, -new_var)), atol_var))
        max_change_in_var = torch.max(torch.abs(torch.add(saved_var, -new_var)))
        
        if mean_equality and var_equality:
            return hetero_mll
        else:
            saved_hetero_mll = hetero_mll
              
        saved_mean = new_mean
        saved_var = new_var
        
        # get new noise estimate
        sampler = IIDNormalSampler(num_samples=num_var_samples, resample=True)
        predictive_samples = sampler(hetero_predictive_posterior)
        observed_var = 0.5 * ((predictive_samples - train_Y.reshape(-1,1))**2).mean(dim=0)
                  
    msg = f'Did not reach convergence after {max_iter} iterations. Returning the current MLL.'
    warnings.warn(msg)
    return hetero_mll