#!/usr/bin/env python3

from numbers import Number
from typing import Union, Tuple

import torch
from gpytorch import Module
from torch import Tensor
from torch.distributions import Normal


"""
Single-valued acquisition functions supporting t-batch evaluation.
These functions use analytical expressions and do not rely on MC-sampling.
"""


def expected_improvement(
    X: Tensor, model: Module, best_f: Union[float, Tensor]
) -> Tensor:
    """Single-outcome expected improvement (assumes maximization)

    Args:
        X: Either a `n x d` or a `b x n x d` (batch mode) Tensor of `n` individual
            design points in `d` dimensions each. Design points are evaluated
            independently (i.e. covariance across the different points is not
            considered)
        model: A fitted single-outcome GP model (must be in batch mode if X is)
        best_f: Either a scalar or a one-dim tensor with `b` elements (batch mode)
            representing the best function value observed so far (assumed noiseless)

    Returns:
        A one-dim tensor with `n` elements or a `b x n` tensor (batch mode)
        corresponding to the expected improvement values of the respective design
        points

    """
    if isinstance(best_f, Number):
        best_f = torch.tensor(best_f, dtype=X.dtype, device=X.device)
    best = best_f.view(X.shape[:-2] + torch.Size([1]))
    model.eval()
    posterior = model(X)
    mean, variance = _get_mean_and_variance(posterior)
    sigma = variance.sqrt()
    u = (mean - best).clamp_min_(0) / sigma
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei = sigma * (updf + u * ucdf)
    return ei


def posterior_mean(X: Tensor, model: Module) -> Tensor:
    """Single-outcome posterior mean

    Args:
        X: Either a `n x d` or a `b x n x d` (batch mode) Tensor of `n` individual
            design points in `d` dimensions each. Design points are evaluated
            independently (i.e. covariance across the different points is not
            considered)
        model: A fitted single-outcome GP model (must be in batch mode if X is)

    Returns:
        A one-dim tensor with `n` elements or a `b x n` tensor (batch mode)
        corresponding to the posterior mean of the respective design points

    """
    model.eval()
    posterior = model(X)
    return posterior.mean


def probability_of_improvement(
    X: Tensor, model: Module, best_f: Union[float, Tensor]
) -> Tensor:
    """Single-outcome probabiltiy of improvement (assumes maximization)

    Args:
        X: Either a `n x d` or a `b x n x d` (batch mode) Tensor of `n` individual
            design points in `d` dimensions each. Design points are evaluated
            independently (i.e. covariance across the different points is not
            considered)
        model: A fitted single-outcome GP model (must be in batch mode if X is)
        best_f: Either a scalar or a one-dim tensor with `b` elements (batch mode)
            representing the best function value observed so far (assumed noiseless)

    Returns:
        A one-dim tensor with `n` elements or a `b x n` tensor (batch mode)
        corresponding to the expected improvement values of the respective design
        points

    """
    if isinstance(best_f, Number):
        best_f = torch.tensor(best_f, dtype=X.dtype, device=X.device)
    best = best_f.view(X.shape[:-2] + torch.Size([1]))
    model.eval()
    posterior = model(X)
    mean, variance = _get_mean_and_variance(posterior)
    sigma = variance.sqrt()
    u = (mean - best) / sigma
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    return normal.cdf(u)


def upper_confidence_bound(
    X: Tensor, model: Module, beta: Union[float, Tensor]
) -> Tensor:
    """Single-outcome upper confidence bound (assumes maximization)

    Args:
        X: Either a `n x d` or a `b x n x d` (batch mode) Tensor of `n` individual
            design points in `d` dimensions each. Design points are evaluated
            independently (i.e. covariance across the different points is not
            considered)
        model: A fitted single-outcome GP model (must be in batch mode if X is)
        beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
            representing the trade-off parameter between mean and covariance

    Returns:
        A one-dim tensor with `n` elements or a `b x n` tensor (batch mode)
        corresponding to the upper confidence bound values of the respective design
        points

    """
    if isinstance(beta, Number):
        beta = torch.tensor(beta, dtype=X.dtype, device=X.device)
    beta = beta.view(X.shape[:-2] + torch.Size([1]))
    model.eval()
    posterior = model(X)
    mean, variance = _get_mean_and_variance(posterior)
    return mean + (beta * variance).sqrt()


def max_value_entropy_search(X: Tensor, model: Module, num_samples: int) -> Tensor:
    raise NotImplementedError()


def _get_mean_and_variance(posterior: Module) -> Tuple[Tensor, Tensor]:
    mu = posterior.mean
    covariance = posterior.lazy_covariance_matrix
    return mu, covariance.diag()
