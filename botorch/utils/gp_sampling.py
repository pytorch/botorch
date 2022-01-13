#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from functools import partial
from math import pi
from typing import List, Optional

import torch
from botorch.models.converter import batched_to_model_list
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.utils.sampling import manual_seed
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel
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

    def __init__(
        self,
        kernel: Kernel,
        input_dim: int,
        num_rff_features: int,
        sample_shape: Optional[torch.Size] = None,
    ) -> None:
        r"""Initialize RandomFourierFeatures.

        Args:
            kernel: The GP kernel.
            input_dim: The input dimension to the GP kernel.
            num_rff_features: The number of fourier features.
            sample_shape: The shape of a single sample. For a single-element
                `torch.Size` object, this is simply the number of RFF draws.
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
        self.sample_shape = torch.Size() if sample_shape is None else sample_shape
        self.register_buffer(
            "weights",
            self._get_weights(
                base_kernel=base_kernel,
                input_dim=input_dim,
                num_rff_features=num_rff_features,
                sample_shape=self.sample_shape,
            ),
        )
        # initialize uniformly in [0, 2 * pi]
        self.register_buffer(
            "bias",
            2
            * pi
            * torch.rand(
                *self.sample_shape,
                num_rff_features,
                dtype=base_kernel.lengthscale.dtype,
                device=base_kernel.lengthscale.device,
            ),
        )

    def _get_weights(
        self,
        base_kernel: Kernel,
        input_dim: int,
        num_rff_features: int,
        sample_shape: Optional[torch.Size] = None,
    ) -> Tensor:
        r"""Sample weights for RFF.

        Args:
            kernel: The GP base kernel.
            input_dim: The input dimension to the GP kernel.
            num_rff_features: The number of fourier features.
            sample_shape: The sample shape of weights.
        Returns:
            A `(sample_shape) x input_dim x num_rff_features`-dim tensor of weights
        """
        sample_shape = torch.Size() if sample_shape is None else sample_shape
        weights = torch.randn(
            *sample_shape,
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
        """
        Get fourier basis features for the provided inputs.
        Note that if `sample_shape` has been passed, then the rightmost
        subset of the batch shape of the input should be `sample_shape`.

        Args:
            X: input tensor of shape `(batch_shape) x n x input_dim`

        Returns:
            A Tensor of shape `(batch_shape) x n x rff`
        """
        self._check_forward_X_shape_compatibility(X)
        # X is of shape (batch_shape_minus_sample_shape) x (sample_shape) x n x d
        # weights is of shape (sample_shape) x d x num_rff
        X_scaled = torch.div(X, self.lengthscale)
        batchmatmul = X_scaled @ self.weights
        bias = self.bias
        # bias is of shape (sample_shape) x num_rff
        # batchmatmul is of shape
        # (batch_shape_minus_sample_shape) x (sample_shape) x n x num_rff
        outputs = torch.cos(batchmatmul + bias.unsqueeze(-2))
        return torch.sqrt(2.0 * self.outputscale / self.weights.shape[-1]) * outputs

    def _check_forward_X_shape_compatibility(self, X: Tensor) -> None:
        len_sample_shape = len(self.sample_shape)
        full_batch_shape_X = X.shape[:-2]
        len_full_batch_shape_X = len(full_batch_shape_X)
        num_trail_dims = len_full_batch_shape_X - len_sample_shape
        # if there is no batch dimension, then we forward pass through every RFF sample
        # otherwise, we check if the rightmost subset matches sample_shape
        if (
            len_full_batch_shape_X
            and full_batch_shape_X[num_trail_dims:] != self.sample_shape
        ):
            raise ValueError(
                "the batch shape of X is expected to follow the pattern: "
                f"`... x {tuple(self.sample_shape)}`"
            )


def get_deterministic_model_multi_samples(
    weights: List[Tensor],
    bases: List[RandomFourierFeatures],
) -> GenericDeterministicModel:
    """
    Get a batched deterministic model that batch evaluates `n_samples` function
    samples. This supports multi-output models as well.

    Args:
        weights: a list of weights with `num_outputs` elements. Each weight is of
            shape `(batch_shape_input) x n_samples x num_rff_features`, where
            `(batch_shape_input)` is the batch shape of the inputs used to obtain the
            posterior weights.
        bases: a list of RandomFourierFeatures with `num_outputs` elements. Each
            basis has a sample shape of `n_samples`.
        n_samples: the number of function samples.

    Returns:
        A batched `GenericDeterministicModel`s that batch evaluates `n_samples`
        function samples.
    """

    def evaluate_gps_X(X, weights, bases):
        list_of_outputs = []
        for w, basis in zip(weights, bases):
            # This ensures that the outermost batch dimension is the sample dimension
            # via basis._check_forward_X_shape_compatibility.
            phi_X = basis(X)
            # X.shape == (batch_shape_X_minus_n_samples) x n_samples x n x d
            # phi_X.shape == (batch_shape_X_minus_n_samples) x n_samples x n x num_rff
            # weights[0].shape == (batch_shape_input) x n_samples x num_rff
            # if X doesn't have a batch shape,
            # then phi_X.shape == n_samples x n x num_rff
            pending_dims = [1] * max(len(X.shape[:-2]) - 1, 0)
            # the below view operation is inserting batch_shape_X_minus_n_samples
            # dimensions in w to enable broadcasted matmul.
            w_unsqueezed = w.view(*w.shape[:-2], *pending_dims, *w.shape[-2:], 1)
            list_of_outputs.append((phi_X @ w_unsqueezed).squeeze(-1))

        return torch.stack(list_of_outputs, dim=-1)

    return GenericDeterministicModel(
        f=partial(evaluate_gps_X, weights=weights, bases=bases),
        num_outputs=len(weights),
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
        X: a `(batch_shape) x n x num_rff_features`-dim tensor of inputs
        y: a `(batch_shape) x n`-dim tensor of outputs
        sigma_sq: the noise variance

    Returns:
        The posterior distribution over the weights.
    """
    with torch.no_grad():
        X_trans = X.transpose(-2, -1)
        A = X_trans @ X + sigma_sq * torch.eye(
            X.shape[-1], dtype=X.dtype, device=X.device
        )
        # mean is given by: m = S @ x.T @ y, where S = A_inv
        # compute inverse of A using solves
        # covariance is A_inv * sigma
        L_A = psd_safe_cholesky(A)
        # solve L_A @ u = I
        Iw = torch.eye(L_A.shape[-1], dtype=X.dtype, device=X.device)
        u = torch.triangular_solve(Iw, L_A, upper=False).solution
        # solve L_A^T @ S = u
        A_inv = torch.triangular_solve(u, L_A.transpose(-2, -1)).solution
        m = (A_inv @ X_trans @ y.unsqueeze(-1)).squeeze(-1)
        L = psd_safe_cholesky(A_inv * sigma_sq)
        return MultivariateNormal(loc=m, scale_tril=L)


def get_gp_samples(
    model: Model, num_outputs: int, n_samples: int, num_rff_features: int = 500
) -> GenericDeterministicModel:
    r"""Sample functions from GP posterior using RFFs. The returned
    `GenericDeterministicModel` effectively wraps `num_outputs` models,
    each of which has a batch shape of `n_samples`. Refer
    `get_deterministic_model_multi_samples` for more details.

    Args:
        model: The model.
        num_outputs: The number of outputs.
        n_samples: The number of functions to be sampled IID.
        num_rff_features: The number of random Fourier features.

    Returns:
        A batched `GenericDeterministicModel` that batch evaluates `n_samples`
        sampled functions.
    """
    if num_outputs > 1:
        if not isinstance(model, ModelListGP):
            models = batched_to_model_list(model).models
        else:
            models = model.models
    else:
        models = [model]
    if isinstance(models[0], MultiTaskGP):
        raise NotImplementedError

    weights = []
    bases = []
    for m in range(num_outputs):
        train_X = models[m].train_inputs[0]
        train_targets = models[m].train_targets
        # get random fourier features
        # sample_shape controls the number of iid functions.
        basis = RandomFourierFeatures(
            kernel=models[m].covar_module,
            input_dim=train_X.shape[-1],
            num_rff_features=num_rff_features,
            sample_shape=torch.Size([n_samples]),
        )
        bases.append(basis)
        # TODO: when batched kernels are supported in RandomFourierFeatures,
        # the following code can be uncommented.
        # if train_X.ndim > 2:
        #    batch_shape_train_X = train_X.shape[:-2]
        #    dataset_shape = train_X.shape[-2:]
        #    train_X = train_X.unsqueeze(-3).expand(
        #        *batch_shape_train_X, n_samples, *dataset_shape
        #    )
        #    train_targets = train_targets.unsqueeze(-2).expand(
        #        *batch_shape_train_X, n_samples, dataset_shape[0]
        #    )
        phi_X = basis(train_X)
        # Sample weights from bayesian linear model
        # 1. When inputs are not batched, train_X.shape == (n, d)
        # weights.sample().shape == (n_samples, num_rff_features)
        # 2. When inputs are batched, train_X.shape == (batch_shape_input, n, d)
        # This is expanded to (batch_shape_input, n_samples, n, d)
        # to maintain compatibility with RFF forward semantics
        # weights.sample().shape == (batch_shape_input, n_samples, num_rff_features)
        mvn = get_weights_posterior(
            X=phi_X,
            y=train_targets,
            sigma_sq=models[m].likelihood.noise.mean().item(),
        )
        weights.append(mvn.sample())

    # TODO: Ideally support RFFs for multi-outputs instead of having to
    # generate a basis for each output serially.
    return get_deterministic_model_multi_samples(weights=weights, bases=bases)
