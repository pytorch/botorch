#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from math import pi
from typing import List, Optional

import torch
from botorch.models.converter import batched_to_model_list
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.utils.sampling import manual_seed
from gpytorch.kernels import Kernel, RBFKernel, MaternKernel, ScaleKernel
from gpytorch.utils.cholesky import psd_safe_cholesky
from torch import Tensor
from torch.distributions import MultivariateNormal
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
        base_sample_shape = posterior.base_sample_shape
        # re-use old samples
        bs_shape = base_sample_shape[:-2] + X.shape[-2:-1] + base_sample_shape[-1:]
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


class RandomFourierFeatures(Module):
    """A class that represents Random Fourier Features."""

    def __init__(self, kernel: Kernel, input_dim: int, num_rff_features: int) -> None:
        r"""Initialize RandomFourierFeatures.

        Args:
            kernel: the GP kernel
            input_dim: the input dimension to the GP kernel
            num_rff_features: the number of fourier features
        """
        if not isinstance(kernel, ScaleKernel):
            base_kernel = kernel
            outputscale = torch.tensor(
                1.0,
                dtype=base_kernel.lengthscale.dtype,
                device=base_kernel.lengthscale.device,
            )
        else:
            base_kernel = kernel.base_kernel
            outputscale = kernel.outputscale.detach().clone()
        if not isinstance(base_kernel, (MaternKernel, RBFKernel)):
            raise NotImplementedError("Only Matern and RBF kernels are supported.")
        elif len(base_kernel.batch_shape) > 0:
            raise NotImplementedError("Batched kernels are not supported.")
        super().__init__()
        self.register_buffer("outputscale", outputscale)

        self.register_buffer("lengthscale", base_kernel.lengthscale.detach().clone())
        self.register_buffer(
            "weights",
            self._get_weights(
                base_kernel=base_kernel,
                input_dim=input_dim,
                num_rff_features=num_rff_features,
            ),
        )
        # initialize uniformly in [0, 2 * pi]
        self.register_buffer(
            "bias",
            2
            * pi
            * torch.rand(
                num_rff_features,
                dtype=base_kernel.lengthscale.dtype,
                device=base_kernel.lengthscale.device,
            ),
        )

    def _get_weights(
        self, base_kernel: Kernel, input_dim: int, num_rff_features: int
    ) -> Tensor:
        r"""Sample weights for RFF.

        Args:
            kernel: the GP base kernel
            input_dim: the input dimension to the GP kernel
            num_rff_features: the number of fourier features

        Returns:
            A `input_dim x num_rff_features`-dim tensor of weights
        """
        weights = torch.randn(
            input_dim,
            num_rff_features,
            dtype=base_kernel.lengthscale.dtype,
            device=base_kernel.lengthscale.device,
        )
        if isinstance(base_kernel, MaternKernel):
            gamma_dist = torch.distributions.Gamma(base_kernel.nu, base_kernel.nu)
            gamma_samples = gamma_dist.sample(torch.Size([1, num_rff_features])).to(
                weights
            )
            weights = torch.rsqrt(gamma_samples) * weights
        return weights

    def forward(self, X: Tensor) -> Tensor:
        r"""Get fourier basis features for the provided inputs."""
        X_scaled = torch.div(X, self.lengthscale)
        outputs = torch.cos(X_scaled @ self.weights + self.bias)
        return (
            torch.sqrt(torch.tensor(2.0) * self.outputscale / self.weights.shape[-1])
            * outputs
        )


def get_deterministic_model(
    weights: List[Tensor], bases: List[RandomFourierFeatures]
) -> GenericDeterministicModel:
    """Get a deterministic model using the provided weights and bases for each output.

    Args:
        weights: a list of weights with `m` elements
        bases: a list of RandomFourierFeatures with `m` elements.

    Returns:
        A deterministic model.
    """

    def evaluate_gp_sample(X):
        return torch.stack([basis(X) @ w for w, basis in zip(weights, bases)], dim=-1)

    return GenericDeterministicModel(f=evaluate_gp_sample, num_outputs=len(weights))


def get_weights_posterior(X: Tensor, y: Tensor, sigma_sq: float) -> MultivariateNormal:
    r"""Sample bayesian linear regression weights.

    Args:
        X: a `n x num_rff_features`-dim tensor of inputs
        y: a `n`-dim tensor of outputs
        sigma_sq: the noise variance

    Returns:
        The posterior distribution over the weights.
    """
    with torch.no_grad():
        A = X.T @ X + sigma_sq * torch.eye(X.shape[-1], dtype=X.dtype, device=X.device)
        # mean is given by: m = S @ x.T @ y, where S = A_inv
        # compute inverse of A using solves
        # covariance is A_inv * sigma
        L_A = psd_safe_cholesky(A)
        # solve L_A @ u = I
        Iw = torch.eye(L_A.shape[0], dtype=X.dtype, device=X.device)
        u = torch.triangular_solve(Iw, L_A, upper=False).solution
        # solve L_A^T @ S = u
        A_inv = torch.triangular_solve(u, L_A.T).solution
        m = A_inv @ X.T @ y
        L = psd_safe_cholesky(A_inv * sigma_sq)
        return MultivariateNormal(loc=m, scale_tril=L)


def get_gp_samples(
    model: Model, num_outputs: int, n_samples: int, num_rff_features: int = 500
) -> List[GenericDeterministicModel]:
    r"""Sample functions from GP posterior using RFF.

    Args:
        model: the model
        num_outputs: the number of outputs
        n_samples: the number of sampled functions to draw
        num_rff_features: the number of random fourier features

    Returns:
        A list of sampled functions.
    """
    if num_outputs > 1:
        if not isinstance(model, ModelListGP):
            models = batched_to_model_list(model).models
    else:
        models = [model]
    if isinstance(models[0], MultiTaskGP):
        raise NotImplementedError

    weights = []
    bases = []
    for m in range(num_outputs):
        train_X = models[m].train_inputs[0]
        # get random fourier features
        basis = RandomFourierFeatures(
            kernel=models[m].covar_module,
            input_dim=train_X.shape[-1],
            num_rff_features=num_rff_features,
        )
        bases.append(basis)
        phi_X = basis(train_X)
        # sample weights from bayesian linear model
        mvn = get_weights_posterior(
            X=phi_X,
            y=models[m].train_targets,
            sigma_sq=models[m].likelihood.noise.mean().item(),
        )
        weights.append(mvn.sample(torch.Size([n_samples])))
        # construct a determinisitic, multi-output model for each sample
    models = [
        get_deterministic_model(
            weights=[weights[m][i] for m in range(num_outputs)],
            bases=bases,
        )
        for i in range(n_samples)
    ]
    return models
