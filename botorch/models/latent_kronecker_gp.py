#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
References

.. [lin2024scaling]
    J. A. Lin, S. Ament, M. Balandat, E. Bakshy. Scaling Gaussian Processes
    for Learning Curve Prediction via Latent Kronecker Structure. NeurIPS 2024
    Bayesian Decision-making and Uncertainty Workshop.

.. [lin2023sampling]
    J. A. Lin, J. Antorán, s. Padhy, D. Janz, J. M. Hernández-Lobato, A. Terenin.
    Sampling from Gaussian Process Posterior using Stochastic Gradient Descent.
    Advances in Neural Information Processing Systems 2023.
"""

import contextlib
import warnings
from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import FantasizeMixin, Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.latent_kronecker import LatentKroneckerGPPosterior
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means import Mean, ZeroMean

from gpytorch.models.exact_gp import ExactGP
from gpytorch.module import Module
from linear_operator import settings
from linear_operator.operators import (
    ConstantDiagLinearOperator,
    KroneckerProductLinearOperator,
    MaskedLinearOperator,
)
from linear_operator.utils.warnings import PerformanceWarning
from torch import Tensor


class MinMaxStandardize(Standardize):
    r"""Standardize outcomes (zero mean, unit variance),
    centered about the minimum (or maximum) instead of the mean.
    Otherwise equivalent to 'Standardize'.
    """

    def __init__(
        self,
        m: int = 1,
        use_min: bool = False,
        outputs: list[int] | None = None,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        min_stdv: float = 1e-8,
    ) -> None:
        r"""Standardize outcomes (zero mean, unit variance).

        Args:
            m: The output dimension.
            use_min: Whether to use the minimum or maximum (instead of the mean).
            outputs: Which of the outputs to standardize. If omitted, all
                outputs will be standardized.
            batch_shape: The batch_shape of the training targets.
            min_stddv: The minimum standard deviation for which to perform
                standardization (if lower, only de-mean the data).
        """
        super().__init__(
            m=m, outputs=outputs, batch_shape=batch_shape, min_stdv=min_stdv
        )
        self._use_min = use_min

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Standardize outcomes.

        If the module is in train mode, this updates the module state (i.e. the
        mean/std normalizing constants). If the module is in eval mode, simply
        applies the normalization using the module state.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).
            X: A `batch_shape x n x d`-dim tensor of training inputs (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        if self.training:
            if Y.shape[:-2] != self._batch_shape:
                raise RuntimeError(
                    f"Expected Y.shape[:-2] to be {self._batch_shape}, matching "
                    "the `batch_shape` argument to `Standardize`, but got "
                    f"Y.shape[:-2]={Y.shape[:-2]}."
                )
            if Y.size(-1) != self._m:
                raise RuntimeError(
                    f"Wrong output dimension. Y.size(-1) is {Y.size(-1)}; expected "
                    f"{self._m}."
                )
            if Y.shape[-2] < 1:
                raise ValueError(f"Can't standardize with no observations. {Y.shape=}.")

            elif Y.shape[-2] == 1:
                stdvs = torch.ones(
                    (*Y.shape[:-2], 1, Y.shape[-1]), dtype=Y.dtype, device=Y.device
                )
            else:
                stdvs = Y.std(dim=-2, keepdim=True)
            stdvs = stdvs.where(stdvs >= self._min_stdv, torch.full_like(stdvs, 1.0))
            means = (
                Y.min(dim=-2, keepdim=True).values
                if self._use_min
                else Y.max(dim=-2, keepdim=True).values
            )
            if self._outputs is not None:
                unused = [i for i in range(self._m) if i not in self._outputs]
                means[..., unused] = 0.0
                stdvs[..., unused] = 1.0
            self.means = means
            self.stdvs = stdvs
            self._stdvs_sq = stdvs.pow(2)
            self._is_trained = torch.tensor(True)

        Y_tf = (Y - self.means) / self.stdvs
        Yvar_tf = Yvar / self._stdvs_sq if Yvar is not None else None
        return Y_tf, Yvar_tf


class LatentKroneckerGP(GPyTorchModel, ExactGP, FantasizeMixin):
    r"""
    A multi-task GP model which uses Kronecker structure despite missing entries.

    Leverages pathwise conditioning and iterative linear system solvers to
    efficiently draw samples from the GP posterior. See [lin2024scaling]_
    for details.

    For more information about pathwise conditioning, see [wilson2021pathwise]_
    and [Maddox2021bohdo]_. Details about iterative linear system solvers for GPs
    with pathwise conditioning can be found in [lin2023sampling]_.

    NOTE: This model requires iterative methods for efficient posterior inference.
    To enable iterative methods, the `use_iterative_methods` helper function can be
    used as a context manager.

    Example:
        >>> model = LatentKroneckerGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> with model.use_iterative_methods():
        >>>     fit_gpytorch_mll(mll)
        >>>     samples = model.posterior(test_X).rsample()
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Y_valid: Tensor | None = None,
        T: Tensor | None = None,
        likelihood: Likelihood | None = None,
        mean_module_X: Mean | None = None,
        mean_module_T: Mean | None = None,
        covar_module_X: Module | None = None,
        covar_module_T: Module | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x t` tensor of training observations.
            train_Y_valid: A `n x t` boolean tensor of valid values.
                True indicates that the corresponding value is valid.
                False indicates that the corresponding value is missing.
                Does not allow explicit `batch_shape` because
                the mask must be shared across batch dimensions.
            T: A `batch_shape x t` tensor of training time steps.
                If omitted, use [1, ..., t].
            likelihood: A likelihood. If omitted, use a standard
                `GaussianLikelihood` with inferred noise level.
            mean_module_X: The mean function to be used for X.
                If omitted, use a `ConstantMean`.
            mean_module_T: The mean function to be used for T.
                If omitted, use a `ConstantMean`.
            covar_module_X: The module computing the covariance matrix of X.
                If omitted, use a `MaternKernel`.
            covar_module_T: The module computing the covariance matrix of T.
                If omitted, use a `MaternKernel`.
            input_transform: An input transform that is applied to X.
            outcome_transform: An outcome transform that is applied to Y.
        """
        with torch.no_grad():
            # transform inputs here to check resulting shapes
            # actual transforms will be applied in forward() and posterior()
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )

        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        batch_shape, ard_num_dims = transformed_X.shape[:-2], transformed_X.shape[-1]

        self.T = self._init_T(T, batch_shape, train_Y)

        self._num_outputs = self.T.shape[-1]

        if likelihood is None:
            likelihood = GaussianLikelihood(batch_shape=batch_shape)

        if train_Y_valid is not None:
            if train_Y_valid.shape != train_Y.shape[-2:]:
                raise BotorchTensorDimensionError(
                    "Explicit batch_shape not allowed for train_Y_valid, "
                    "because the mask must be shared across batch dimensions. "
                    f"Expected train_Y_valid with shape: {train_Y.shape[-2:]} "
                    f"(got {train_Y_valid.shape})."
                )
            assert train_Y_valid.dtype == torch.bool
            self.mask = train_Y_valid.reshape(-1)
        else:
            mask_len = train_Y.shape[-2] * train_Y.shape[-1]
            self.mask = torch.ones(mask_len, dtype=torch.bool, device=train_Y.device)

        train_Y = train_Y.reshape(*batch_shape, -1)[..., self.mask]

        if outcome_transform == DEFAULT:
            outcome_transform = MinMaxStandardize(batch_shape=batch_shape)
        if outcome_transform is not None:
            # transform outputs once and keep the results
            train_Y = outcome_transform(train_Y.unsqueeze(-1), X=transformed_X)[
                0
            ].squeeze(-1)

        ExactGP.__init__(
            self,
            train_inputs=train_X,
            train_targets=train_Y,
            likelihood=likelihood,
        )

        if mean_module_X is None:
            mean_module_X = ZeroMean(batch_shape=batch_shape)
        self.mean_module_X: Module = mean_module_X

        if mean_module_T is None:
            mean_module_T = ZeroMean(batch_shape=batch_shape)
        self.mean_module_T: Module = mean_module_T

        if covar_module_X is None:
            covar_module_X = MaternKernel(
                ard_num_dims=ard_num_dims, batch_shape=batch_shape
            )

        if covar_module_T is None:
            covar_module_T = ScaleKernel(
                base_kernel=MaternKernel(ard_num_dims=1, batch_shape=batch_shape),
            )

        self.covar_module_X: Module = covar_module_X
        self.covar_module_T: Module = covar_module_T

        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

        self._cached_base_samples = None
        self._cached_L_train_train_X = None
        self._cached_L_T = None
        self._cached_H_inv_v = None

        self.to(train_X)

    def _init_T(
        self, T: Tensor | None, batch_shape: torch.Size, train_Y: Tensor
    ) -> Tensor:
        if T is not None:
            expected_shape = torch.Size([*batch_shape, train_Y.shape[-1]])
            if T.shape != expected_shape:
                raise BotorchTensorDimensionError(
                    f"Expected T with shape: {expected_shape} (got {T.shape})."
                )
            return T
        else:
            T = torch.linspace(
                0, 1, train_Y.shape[-1], dtype=train_Y.dtype, device=train_Y.device
            )
            T = T.expand(*batch_shape, -1)
            return T

    def use_iterative_methods(
        self,
        tol: float = 0.01,
        max_iter: int = 10000,
        covar_root_decomposition: bool = False,
        log_prob: bool = True,
        solves: bool = True,
    ):
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                settings.fast_computations(
                    covar_root_decomposition=covar_root_decomposition,
                    log_prob=log_prob,
                    solves=solves,
                )
            )
            stack.enter_context(settings.cg_tolerance(tol))
            stack.enter_context(settings.max_cg_iterations(max_iter))
            return stack.pop_all()

    def _get_mean(self, X: Tensor, mask: Tensor | None = None) -> Tensor:
        mean_X = self.mean_module_X(X).unsqueeze(-1)
        mean_T = self.mean_module_T(self.T.unsqueeze(-1)).unsqueeze(-1)
        mean = KroneckerProductLinearOperator(mean_X, mean_T).squeeze(-1)
        return mean[..., mask] if mask is not None else mean

    def forward(self, X: Tensor) -> MultivariateNormal:
        if self.training:
            X = self.transform_inputs(X)
            mask = self.mask
        else:
            total_len = X.shape[-2] * self._num_outputs
            mask = torch.ones(total_len, dtype=torch.bool, device=X.device)
            mask[: self.mask.shape[-1]] = self.mask

        mean = self._get_mean(X, mask)

        covar_X = self.covar_module_X(X)
        covar_T = self.covar_module_T(self.T.unsqueeze(-1))
        covar = KroneckerProductLinearOperator(covar_X, covar_T)
        covar = MaskedLinearOperator(covar, row_mask=mask, col_mask=mask)

        return MultivariateNormal(mean, covar)

    def posterior(
        self,
        X: Tensor,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        if posterior_transform is not None:
            raise NotImplementedError(
                "Posterior transforms currently not supported for "
                f"{self.__class__.__name__}"
            )
        if not isinstance(self.likelihood, GaussianLikelihood):
            raise NotImplementedError(
                "Only GaussianLikelihood currently supported for "
                f"{self.__class__.__name__}"
            )
        if observation_noise is not False:
            raise NotImplementedError(
                "Observation noise currently not supported for "
                f"{self.__class__.__name__}"
            )
        return LatentKroneckerGPPosterior(self, X)

    def _rsample_from_base_samples(
        self,
        X: Tensor,
        base_samples: Tensor,
        observation_noise: bool | Tensor = False,
    ) -> Tensor:
        r"""Sample from the posterior distribution at the provided points `X`
        using Matheron's rule, requiring `n + 2 n_train` base samples.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly
            base_samples: A Tensor of `N(0, I)` base samples of shape
                `sample_shape x base_sample_shape`, typically obtained from
                a `Sampler`. This is used for deterministic optimization.
        Returns:
            Samples from the posterior, a tensor of shape
            `self._extended_shape(sample_shape=sample_shape)`.
        """
        # toggle eval mode to switch the behavior of input / outcome transforms
        # this also implicitly applies the input transform to the train_inputs
        self.eval()
        X_train = self.train_inputs[0]
        X_test = self.transform_inputs(X)
        n_train_full = X_train.shape[-2] * self._num_outputs
        n_train = self.train_targets.shape[-1]
        n_test = X_test.shape[-2] * self._num_outputs

        sample_shape = base_samples.shape[: -len(self.batch_shape) - 1]
        w_train, eps_base, w_test = torch.split(
            base_samples, [n_train_full, n_train, n_test], dim=-1
        )
        eps = torch.sqrt(self.likelihood.noise) * eps_base

        K_T = self.covar_module_T(self.T.unsqueeze(-1))

        if self._cached_base_samples is not None and torch.equal(
            base_samples, self._cached_base_samples
        ):
            L_train_train_X = self._cached_L_train_train_X
            L_T = self._cached_L_T
            H_inv_v = self._cached_H_inv_v
        else:
            # Evaluate prior mean at training data
            m_train = self._get_mean(X_train, self.mask)

            # Calculate prior sample
            K_train_train_X = self.covar_module_X(X_train)
            L_train_train_X = K_train_train_X.cholesky(upper=False)
            L_T = K_T.cholesky(upper=False)

            L_train_train = KroneckerProductLinearOperator(L_train_train_X, L_T)

            f_prior_train = L_train_train @ w_train.unsqueeze(-1)
            f_prior_train = m_train + f_prior_train.squeeze(-1)[..., self.mask]

            K_train_train = KroneckerProductLinearOperator(K_train_train_X, K_T)
            K_train_train = MaskedLinearOperator(
                K_train_train, row_mask=self.mask, col_mask=self.mask
            )
            noise_covar = ConstantDiagLinearOperator(
                self.likelihood.noise
                * torch.ones(*self.batch_shape, 1, dtype=X.dtype, device=X.device),
                diag_shape=n_train,
            )
            H = K_train_train + noise_covar

            v = self.train_targets - (f_prior_train + eps)
            # Expand once here to avoid repeated expansion
            # by MaskedLinearOperator later
            H_inv_v = torch.zeros(
                *sample_shape,
                *self.batch_shape,
                n_train_full,
                dtype=X.dtype,
                device=X.device,
            )
            if settings._fast_solves.off():
                warn_msg = (
                    "Iterative methods are disabled. Performing linear solve using "
                    "full joint covariance matrix, which might be slow and require "
                    "a lot of memory. Iterative methods can be enabled using "
                    "'with model.use_iterative_methods():'."
                )
                warnings.warn(
                    warn_msg,
                    PerformanceWarning,
                    stacklevel=2,
                )
            H_inv_v[..., self.mask] = H.solve(v.unsqueeze(-1)).squeeze(-1)

            self._cached_base_samples = base_samples
            self._cached_L_train_train_X = L_train_train_X
            self._cached_L_T = L_T
            self._cached_H_inv_v = H_inv_v

        # Evaluate prior mean at test data
        m_test = self._get_mean(X_test)

        K_train_test_X = self.covar_module_X(X_train, X_test).evaluate_kernel()
        K_test_test_X = self.covar_module_X(X_test).evaluate_kernel()

        L_train_test_X = L_train_train_X.solve_triangular(
            K_train_test_X.tensor, upper=False
        )
        L_test_test_X = (
            K_test_test_X - L_train_test_X.transpose(-2, -1) @ L_train_test_X
        ).cholesky(upper=False)

        L_test_train = KroneckerProductLinearOperator(
            L_train_test_X.transpose(-2, -1), L_T
        )

        L_test_test = KroneckerProductLinearOperator(L_test_test_X, L_T)

        # match dimensions for broadcasting
        broadcast_shape = L_test_train.shape[:-2]
        extra_batch_dims = len(broadcast_shape) - len(self.batch_shape)
        for _ in range(extra_batch_dims):
            w_train = w_train.unsqueeze(len(sample_shape))
            w_test = w_test.unsqueeze(len(sample_shape))
            H_inv_v = H_inv_v.unsqueeze(len(sample_shape))

        f_prior_test = L_test_train @ w_train.unsqueeze(-1)
        f_prior_test = f_prior_test + L_test_test @ w_test.unsqueeze(-1)
        f_prior_test = m_test + f_prior_test.squeeze(-1)

        K_train_test = KroneckerProductLinearOperator(K_train_test_X, K_T)
        # no MaskedLinearOperator here because H_inv_v is already expanded
        samples = K_train_test.transpose(-2, -1) @ H_inv_v.unsqueeze(-1)
        samples = samples + f_prior_test.unsqueeze(-1)
        # reshape samples to separate X and T dimensions
        # samples.shape = (*sample_shape, *broadcast_shape, n_test_x * n_t, 1)
        samples = samples.reshape(
            *samples.shape[:-2], X_test.shape[-2], self._num_outputs
        )
        # samples.shape = (*sample_shape, *broadcast_shape, n_test_x, n_t)
        if hasattr(self, "outcome_transform") and self.outcome_transform is not None:
            samples, _ = self.outcome_transform.untransform(samples, X=X)
        return samples

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, noise: Tensor | None = None, **kwargs: Any
    ) -> Model:
        raise NotImplementedError(
            f"Conditioning currently not supported for {self.__class__.__name__}"
        )
