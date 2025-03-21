#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
The following code is from the repository vbll (https://github.com/VectorInstitute/vbll) which is under the MIT license
The code is from the paper "Variational Bayesian Last Layers" by Harrison et al., ICLR 2024
"""

from typing import Callable
from dataclasses import dataclass
import warnings

import numpy as np

import torch
import torch.nn as nn


# following functions/classes are from https://github.com/VectorInstitute/vbll/blob/main/vbll/utils/distributions.py
def get_parameterization(p):
    if p in cov_param_dict:
        return cov_param_dict[p]
    else:
        raise ValueError("Must specify a valid covariance parameterization.")


def tp(M):
    return M.transpose(-1, -2)


def sym(M):
    return (M + tp(M)) / 2.0


# Credit to https://github.com/brentyi/fannypack/blob/2888aa5d969824ac1e1a528264674ece3f4703f9/fannypack/utils/_math.py
def cholesky_inverse(u: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """Alternative to `torch.cholesky_inverse()`, with support for batch dimensions.

    Relevant issue tracker: https://github.com/pytorch/pytorch/issues/7500

    Args:
        u: Triangular Cholesky factor. Shape should be `(*, N, N)`.
        upper (bool, optional): Whether to consider the Cholesky factor as a lower or
            upper triangular matrix.

    Returns:
        torch.Tensor:
    """
    if u.dim() == 2 and not u.requires_grad:
        return torch.cholesky_inverse(u, upper=upper)
    return torch.cholesky_solve(
        torch.eye(u.size(-1), dtype=torch.float64).expand(u.size()), u, upper=upper
    )


class Normal(torch.distributions.Normal):
    def __init__(self, loc, chol):
        super(Normal, self).__init__(loc, chol)

    @property
    def mean(self):
        return self.loc

    @property
    def var(self):
        return self.scale**2

    @property
    def chol_covariance(self):
        return torch.diag_embed(self.scale)

    @property
    def covariance_diagonal(self):
        return self.var

    @property
    def covariance(self):
        return torch.diag_embed(self.var)

    @property
    def precision(self):
        return torch.diag_embed(1.0 / self.var)

    @property
    def logdet_covariance(self):
        return 2 * torch.log(self.scale).sum(-1)

    @property
    def logdet_precision(self):
        return -2 * torch.log(self.scale).sum(-1)

    @property
    def trace_covariance(self):
        return self.var.sum(-1)

    @property
    def trace_precision(self):
        return (1.0 / self.var).sum(-1)

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (self.var.unsqueeze(-1) * (b**2)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((b**2) / self.var.unsqueeze(-1)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __add__(self, inp):
        if isinstance(inp, Normal):
            new_cov = self.var + inp.var
            return Normal(
                self.mean + inp.mean, torch.sqrt(torch.clip(new_cov, min=1e-12))
            )
        elif isinstance(inp, torch.Tensor):
            return Normal(self.mean + inp, self.scale)
        else:
            raise NotImplementedError(
                "Distribution addition only implemented for diag covs"
            )

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def squeeze(self, idx):
        return Normal(self.loc.squeeze(idx), self.scale.squeeze(idx))


class DenseNormal(torch.distributions.MultivariateNormal):
    def __init__(self, loc, cholesky):
        super(DenseNormal, self).__init__(loc, scale_tril=cholesky)

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        return self.scale_tril

    @property
    def covariance(self):
        return self.scale_tril @ tp(self.scale_tril)

    @property
    def inverse_covariance(self):
        warnings.warn(
            "Direct matrix inverse for dense covariances is O(N^3), consider using eg inverse weighted inner product"
        )
        return tp(torch.linalg.inv(self.scale_tril)) @ torch.linalg.inv(self.scale_tril)

    @property
    def logdet_covariance(self):
        return 2.0 * torch.diagonal(self.scale_tril, dim1=-2, dim2=-1).log().sum(-1)

    @property
    def trace_covariance(self):
        return (self.scale_tril**2).sum(-1).sum(-1)  # compute as frob norm squared

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((tp(self.scale_tril) @ b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (torch.linalg.solve(self.scale_tril, b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def squeeze(self, idx):
        return DenseNormal(self.loc.squeeze(idx), self.scale_tril.squeeze(idx))


class LowRankNormal(torch.distributions.LowRankMultivariateNormal):
    def __init__(self, loc, cov_factor, diag):
        super(LowRankNormal, self).__init__(loc, cov_factor=cov_factor, cov_diag=diag)

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        raise NotImplementedError()

    @property
    def covariance(self):
        return self.cov_factor @ tp(self.cov_factor) + torch.diag_embed(self.cov_diag)

    @property
    def inverse_covariance(self):
        raise NotImplementedError()

    @property
    def logdet_covariance(self):
        # Apply Matrix determinant lemma
        term1 = torch.log(self.cov_diag).sum(-1)
        arg1 = tp(self.cov_factor) @ (self.cov_factor / self.cov_diag.unsqueeze(-1))
        term2 = torch.linalg.det(
            arg1 + torch.eye(arg1.shape[-1], dtype=torch.float64)
        ).log()
        return term1 + term2

    @property
    def trace_covariance(self):
        # trace of sum is sum of traces
        trace_diag = self.cov_diag.sum(-1)
        trace_lowrank = (self.cov_factor**2).sum(-1).sum(-1)
        return trace_diag + trace_lowrank

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        diag_term = (self.cov_diag.unsqueeze(-1) * (b**2)).sum(-2)
        factor_term = ((tp(self.cov_factor) @ b) ** 2).sum(-2)
        prod = diag_term + factor_term
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        raise NotImplementedError()

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def squeeze(self, idx):
        return LowRankNormal(
            self.loc.squeeze(idx),
            self.cov_factor.squeeze(idx),
            self.cov_diag.squeeze(idx),
        )


class DenseNormalPrec(torch.distributions.MultivariateNormal):
    """A DenseNormal parameterized by the mean and the cholesky decomp of the precision matrix.

    This function also includes a recursive_update function which performs a recursive
    linear regression update with effecient cholesky factor updates.
    """

    def __init__(self, loc, cholesky, validate_args=False):
        prec = cholesky @ tp(cholesky)
        super(DenseNormalPrec, self).__init__(
            loc, precision_matrix=prec, validate_args=validate_args
        )
        self.tril = cholesky

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        raise NotImplementedError()

    @property
    def covariance(self):
        warnings.warn(
            "Direct matrix inverse for dense covariances is O(N^3), consider using eg inverse weighted inner product"
        )
        return cholesky_inverse(self.tril)

    @property
    def inverse_covariance(self):
        return self.precision_matrix

    @property
    def logdet_covariance(self):
        return -2.0 * torch.diagonal(self.tril, dim1=-2, dim2=-1).log().sum(-1)

    @property
    def trace_covariance(self):
        return (
            (torch.inverse(self.tril) ** 2).sum(-1).sum(-1)
        )  # compute as frob norm squared

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (torch.linalg.solve(self.tril, b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((tp(self.tril) @ b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def squeeze(self, idx):
        return DenseNormalPrec(self.loc.squeeze(idx), self.tril.squeeze(idx))


cov_param_dict = {
    "dense": DenseNormal,
    "dense_precision": DenseNormalPrec,
    "diagonal": Normal,
    "lowrank": LowRankNormal,
}


# following functions/classes are from https://github.com/VectorInstitute/vbll/blob/main/vbll/layers/regression.py
def gaussian_kl(p, q_scale):
    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean**2).sum(-1).sum(-1) / q_scale
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)
    return 0.5 * (mse_term + trace_term + logdet_term)  # currently exclude constant


@dataclass
class VBLLReturn:
    predictive: Normal | DenseNormal
    train_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    val_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ood_scores: None | Callable[[torch.Tensor], torch.Tensor] = None


class Regression(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        regularization_weight,
        parameterization="dense",
        prior_scale=1.0,
        wishart_scale=1e-2,
        cov_rank=None,
        dof=1.0,
    ):
        """
        Variational Bayesian Linear Regression

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        regularization_weight : float
            Weight on regularization term in ELBO
        parameterization : str
            Parameterization of covariance matrix. Currently supports {'dense', 'diagonal', 'lowrank', 'dense_precision'}
        prior_scale : float
            Scale of prior covariance matrix
        wishart_scale : float
            Scale of Wishart prior on noise covariance
        dof : float
            Degrees of freedom of Wishart prior on noise covariance
        """
        super(Regression, self).__init__()

        self.wishart_scale = wishart_scale
        self.dof = (dof + out_features + 1.0) / 2.0
        self.regularization_weight = regularization_weight
        self.dtype = torch.float64  # NOTE: not in the original source code

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_scale = prior_scale * (1.0 / in_features)

        # noise distribution
        self.noise_mean = nn.Parameter(
            torch.zeros(out_features, dtype=self.dtype), requires_grad=False
        )
        self.noise_logdiag = nn.Parameter(
            torch.randn(out_features, dtype=self.dtype) * (np.log(wishart_scale))
        )

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(
            torch.randn(out_features, in_features, dtype=self.dtype)
        )

        if parameterization == "diagonal":
            self.W_logdiag = nn.Parameter(
                torch.randn(out_features, in_features, dtype=self.dtype)
                - 0.5 * np.log(in_features)
            )
        elif parameterization == "dense":
            self.W_logdiag = nn.Parameter(
                torch.randn(out_features, in_features, dtype=self.dtype)
                - 0.5 * np.log(in_features)
            )
            self.W_offdiag = nn.Parameter(
                torch.randn(out_features, in_features, in_features, dtype=self.dtype)
                / in_features
            )
        elif parameterization == "dense_precision":
            self.W_logdiag = nn.Parameter(
                torch.randn(out_features, in_features, dtype=self.dtype)
                + 0.5 * np.log(in_features)
            )
            self.W_offdiag = nn.Parameter(
                torch.randn(out_features, in_features, in_features, dtype=self.dtype)
                * 0.0
            )
        elif parameterization == "lowrank":
            self.W_logdiag = nn.Parameter(
                torch.randn(out_features, in_features, dtype=self.dtype)
                - 0.5 * np.log(in_features)
            )
            self.W_offdiag = nn.Parameter(
                torch.randn(out_features, in_features, cov_rank, dtype=self.dtype)
                / in_features
            )

    def W(self):
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist == Normal:
            cov = self.W_dist(self.W_mean, cov_diag)
        elif (self.W_dist == DenseNormal) or (self.W_dist == DenseNormalPrec):
            tril = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            cov = self.W_dist(self.W_mean, tril)
        elif self.W_dist == LowRankNormal:
            cov = self.W_dist(self.W_mean, self.W_offdiag, cov_diag)

        return cov

    def noise(self):
        return Normal(self.noise_mean, torch.exp(self.noise_logdiag))

    def forward(self, x):
        out = VBLLReturn(
            self.predictive(x), self._get_train_loss_fn(x), self._get_val_loss_fn(x)
        )
        return out

    def predictive(self, x):
        return (self.W() @ x[..., None]).squeeze(-1) + self.noise()

    def _get_train_loss_fn(self, x):
        def loss_fn(y):
            # construct predictive density N(W @ phi, Sigma)
            W = self.W()
            noise = self.noise()
            pred_density = Normal((W.mean @ x[..., None]).squeeze(-1), noise.scale)
            pred_likelihood = pred_density.log_prob(y)

            trace_term = 0.5 * (
                (W.covariance_weighted_inner_prod(x.unsqueeze(-2)[..., None]))
                * noise.trace_precision
            )

            kl_term = gaussian_kl(W, self.prior_scale)
            wishart_term = (
                self.dof * noise.logdet_precision
                - 0.5 * self.wishart_scale * noise.trace_precision
            )
            total_elbo = torch.mean(pred_likelihood - trace_term)
            total_elbo += self.regularization_weight * (wishart_term - kl_term)
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            # compute log likelihood under variational posterior via marginalization
            logprob = self.predictive(x).log_prob(y).sum(-1)  # sum over output dims
            return -logprob.mean(0)  # mean over batch dim

        return loss_fn
