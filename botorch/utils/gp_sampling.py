#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from copy import deepcopy
from math import pi
from typing import Optional

import torch
from botorch.models.converter import batched_to_model_list
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model, ModelList
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.utils.sampling import manual_seed
from botorch.utils.transforms import is_ensemble
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel
from linear_operator.utils.cholesky import psd_safe_cholesky
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
        warnings.warn(
            "`GPDraw` is deprecated and will be removed in v0.13 release. "
            "For drawing GP sample paths, we recommend using pathwise "
            "sampling code found in `botorch/sampling/pathwise`. We recommend "
            "`get_matheron_path_model` for most use cases.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self._model = deepcopy(model)
        self._num_outputs = self._model.num_outputs
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
        if self._num_outputs == 1:
            # Needed to comply with base sample shape assumptions made here.
            base_sample_shape = base_sample_shape + (1,)
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
        # that could lead to non-deterministic evaluations. We could use SVD- or
        # eigendecomposition-based sampling, but we probably don't want to use this
        # by default for performance reasonse.
        Ys = posterior.rsample_from_base_samples(
            torch.Size(),
            base_samples=(
                base_samples.squeeze(-1) if self._num_outputs == 1 else base_samples
            ),
        )
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
            num_rff_features: The number of Fourier features.
            sample_shape: The shape of a single sample. For a single-element
                `torch.Size` object, this is simply the number of RFF draws.
        """
        if not isinstance(kernel, ScaleKernel):
            base_kernel = kernel
            outputscale = torch.ones(kernel.batch_shape).to(
                dtype=kernel.lengthscale.dtype,
                device=kernel.lengthscale.device,
            )
        else:
            base_kernel = kernel.base_kernel
            outputscale = kernel.outputscale.detach().clone()
        if not isinstance(base_kernel, (MaternKernel, RBFKernel)):
            raise NotImplementedError("Only Matern and RBF kernels are supported.")
        super().__init__()
        self.kernel_batch_shape = base_kernel.batch_shape
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
                *self.kernel_batch_shape,
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
            num_rff_features: The number of Fourier features.
            sample_shape: The sample shape of weights.
        Returns:
            A tensor of weights with shape
            `(*sample_shape, *kernel_batch_shape, input_dim, num_rff_features)`.
        """
        sample_shape = torch.Size() if sample_shape is None else sample_shape
        weights = torch.randn(
            *sample_shape,
            *self.kernel_batch_shape,
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
        """Get Fourier basis features for the provided inputs.

        Note that the right-most subset of the batch shape of `X` should
        be `(sample_shape) x (kernel_batch_shape)` if using either the
        `sample_shape` argument or a batched kernel. In other words,
        `X` should be of shape `(added_batch_shape) x (sample_shape) x
        (kernel_batch_shape) x n x input_dim`, where parantheses denote
        that the given batch shape can be empty. `X` can always be
        a tensor of shape `n x input_dim`, in which case broadcasting
        will take care of the batch shape. This will raise a `ValueError`
        if the batch shapes are not compatible.

        Args:
            X: Input tensor of shape `(batch_shape) x n x input_dim`.

        Returns:
            A Tensor of shape `(batch_shape) x n x rff`. If `X` does not have
            a `batch_shape`, the output `batch_shape` will be
            `(sample_shape) x (kernel_batch_shape)`.
        """
        try:
            self._check_forward_X_shape_compatibility(X)
        except ValueError as e:
            # A workaround to support batched SAAS models.
            # TODO: Support batch evaluation of multi-sample RFFs as well.
            # Multi-sample RFFs have input batch as the 0-th dimension,
            # which is different than other posteriors which would have
            # the sample shape as the 0-th dimension.
            if len(self.kernel_batch_shape) == 1:
                X = X.unsqueeze(-3)
                self._check_forward_X_shape_compatibility(X)
            else:
                raise e

        # X is of shape (additional_batch_shape) x (sample_shape)
        # x (kernel_batch_shape) x n x d.
        # Weights is of shape (sample_shape) x (kernel_batch_shape) x d x num_rff.
        X_scaled = torch.div(X, self.lengthscale)
        batchmatmul = X_scaled @ self.weights
        bias = self.bias
        # Bias is of shape (sample_shape) x (kernel_batch_shape) x num_rff.
        # Batchmatmul is of shape (additional_batch_shape) x (sample_shape)
        # x (kernel_batch_shape) x n x num_rff.
        outputs = torch.cos(batchmatmul + bias.unsqueeze(-2))
        # Make sure we divide at the correct (i.e., kernel's) batch dimension.
        if len(self.kernel_batch_shape) > 0:
            outputscale = self.outputscale.view(*self.kernel_batch_shape, 1, 1)
        else:
            outputscale = self.outputscale
        return torch.sqrt(2.0 * outputscale / self.weights.shape[-1]) * outputs

    def _check_forward_X_shape_compatibility(self, X: Tensor) -> None:
        r"""Check that the `batch_shape` of X, if any, is compatible with the
        `sample_shape` & `kernel_batch_shape`.
        """
        full_batch_shape_X = X.shape[:-2]
        len_full_batch_shape_X = len(full_batch_shape_X)
        if len_full_batch_shape_X == 0:
            # Non-batched X.
            return
        expected_batch_shape = self.sample_shape + self.kernel_batch_shape
        # Check if they're broadcastable.
        for b_idx in range(min(len(expected_batch_shape), len_full_batch_shape_X)):
            neg_idx = -b_idx - 1
            if (
                full_batch_shape_X[neg_idx] != expected_batch_shape[neg_idx]
                and full_batch_shape_X[neg_idx] != 1
            ):
                raise ValueError(
                    "the batch shape of X is expected to follow the pattern: "
                    f"`... x {tuple(expected_batch_shape)}`"
                )


def get_deterministic_model_multi_samples(
    weights: list[Tensor],
    bases: list[RandomFourierFeatures],
) -> GenericDeterministicModel:
    """
    Get a batched deterministic model that batch evaluates `n_samples` function
    samples. This supports multi-output models as well.

    Args:
        weights: A list of weights with `num_outputs` elements. Each weight is of
            shape `(batch_shape_input) x n_samples x num_rff_features`, where
            `(batch_shape_input)` is the batch shape of the inputs used to obtain the
            posterior weights.
        bases: A list of `RandomFourierFeatures` with `num_outputs` elements. Each
            basis has a sample shape of `n_samples`.
        n_samples: The number of function samples.

    Returns:
        A batched `GenericDeterministicModel`s that batch evaluates `n_samples`
        function samples.
    """
    eval_callables = [
        get_eval_gp_sample_callable(w=w, basis=basis)
        for w, basis in zip(weights, bases)
    ]

    def evaluate_gps_X(X):
        return torch.cat([_f(X) for _f in eval_callables], dim=-1)

    return GenericDeterministicModel(
        f=evaluate_gps_X,
        num_outputs=len(weights),
    )


def get_eval_gp_sample_callable(w: Tensor, basis: RandomFourierFeatures) -> Tensor:
    def _f(X):
        return basis(X) @ w.unsqueeze(-1)

    return _f


def get_deterministic_model(
    weights: list[Tensor], bases: list[RandomFourierFeatures]
) -> GenericDeterministicModel:
    """Get a deterministic model using the provided weights and bases for each output.

    Args:
        weights: A list of weights with `m` elements.
        bases: A list of `RandomFourierFeatures` with `m` elements.

    Returns:
        A deterministic model.
    """
    callables = [
        get_eval_gp_sample_callable(w=w, basis=basis)
        for w, basis in zip(weights, bases)
    ]

    def evaluate_gp_sample(X):
        return torch.cat([c(X) for c in callables], dim=-1)

    return GenericDeterministicModel(f=evaluate_gp_sample, num_outputs=len(weights))


def get_deterministic_model_list(
    weights: list[Tensor],
    bases: list[RandomFourierFeatures],
) -> ModelList:
    """Get a deterministic model list using the provided weights and bases
    for each output.

    Args:
        weights: A list of weights with `m` elements.
        bases: A list of `RandomFourierFeatures` with `m` elements.

    Returns:
        A deterministic model.
    """
    samples = []
    for w, basis in zip(weights, bases):
        sample = GenericDeterministicModel(
            f=get_eval_gp_sample_callable(w=w, basis=basis),
            num_outputs=1,
        )
        samples.append(sample)
    return ModelList(*samples)


def get_weights_posterior(X: Tensor, y: Tensor, sigma_sq: Tensor) -> MultivariateNormal:
    r"""Sample bayesian linear regression weights.

    Args:
        X: A tensor of inputs with shape `(*batch_shape, n num_rff_features)`.
        y: A tensor of outcomes with shape `(*batch_shape, n)`.
        sigma_sq: The likelihood noise variance. This should be a tensor with
            shape `kernel_batch_shape, 1, 1` if using a batched kernel.
            Otherwise, it should be a scalar tensor.

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
        u = torch.linalg.solve_triangular(L_A, Iw, upper=False)

        # solve L_A^T @ S = u
        A_inv = torch.linalg.solve_triangular(L_A.transpose(-2, -1), u, upper=True)
        m = (A_inv @ X_trans @ y.unsqueeze(-1)).squeeze(-1)
        L = psd_safe_cholesky(A_inv * sigma_sq)
        return MultivariateNormal(loc=m, scale_tril=L)


def get_gp_samples(
    model: Model, num_outputs: int, n_samples: int, num_rff_features: int = 512
) -> GenericDeterministicModel:
    r"""Sample functions from GP posterior using RFFs. The returned
    `GenericDeterministicModel` effectively wraps `num_outputs` models,
    each of which has a batch shape of `n_samples`. Refer
    `get_deterministic_model_multi_samples` for more details.

    NOTE: If using input / outcome transforms, the gp samples must be accessed via
    the `gp_sample.posterior(X)` call. Otherwise, `gp_sample(X)` will produce bogus
    values that do not agree with the underlying `model`. It is also highly recommended
    to use outcome transforms to standardize the input data, since the gp samples do
    not work well when training outcomes are not zero-mean.

    Args:
        model: The model.
        num_outputs: The number of outputs.
        n_samples: The number of functions to be sampled IID.
        num_rff_features: The number of random Fourier features.

    Returns:
        A `GenericDeterministicModel` that evaluates `n_samples` sampled functions.
        If `n_samples > 1`, this will be a batched model.
    """
    warnings.warn(
        "`get_gp_samples` is deprecated and will be removed in v0.13 release. "
        "For drawing GP sample paths, we recommend using pathwise "
        "sampling code found in `botorch/sampling/pathwise`. We recommend "
        "`get_matheron_path_model` for most use cases.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Get transforms from the model.
    intf = getattr(model, "input_transform", None)
    octf = getattr(model, "outcome_transform", None)
    # Remove the outcome transform - leads to buggy draws.
    if octf is not None:
        del model.outcome_transform
    if intf is not None:
        del model.input_transform

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
    octfs = []
    intfs = []
    for m in range(num_outputs):
        train_X = models[m].train_inputs[0]
        train_targets = models[m].train_targets
        _model = models[m]
        _intf = getattr(_model, "input_transform", None)
        _octf = getattr(_model, "outcome_transform", None)
        # Remove the outcome transform - leads to buggy draws.
        if _octf is not None:
            del _model.outcome_transform

        octfs.append(_octf)
        intfs.append(_intf)
        # Get random Fourier features.
        # sample_shape controls the number of iid functions.
        basis = RandomFourierFeatures(
            kernel=_model.covar_module,
            input_dim=train_X.shape[-1],
            num_rff_features=num_rff_features,
            sample_shape=torch.Size([n_samples] if n_samples > 1 else []),
        )
        bases.append(basis)
        phi_X = basis(train_X)
        # Sample weights from bayesian linear model.
        # weights.sample().shape == (n_samples, batch_shape_input, num_rff_features)
        sigma_sq = _model.likelihood.noise.mean(dim=-1, keepdim=True)
        if len(basis.kernel_batch_shape) > 0:
            sigma_sq = sigma_sq.unsqueeze(-2)
        mvn = get_weights_posterior(
            X=phi_X,
            y=train_targets,
            sigma_sq=sigma_sq,
        )
        weights.append(mvn.sample())

    # TODO: Ideally support RFFs for multi-outputs instead of having to
    # generate a basis for each output serially.
    if any(_octf is not None for _octf in octfs) or any(
        _intf is not None for _intf in intfs
    ):
        base_gp_samples = get_deterministic_model_list(
            weights=weights,
            bases=bases,
        )
        for m in range(len(weights)):
            _octf = octfs[m]
            _intf = intfs[m]
            if _octf is not None:
                base_gp_samples.models[m].outcome_transform = _octf
                models[m].outcome_transform = _octf
            if _intf is not None:
                base_gp_samples.models[m].input_transform = _intf
        base_gp_samples._is_ensemble = is_ensemble(model=model)
        return base_gp_samples
    elif n_samples > 1:
        base_gp_samples = get_deterministic_model_multi_samples(
            weights=weights,
            bases=bases,
        )
    else:
        base_gp_samples = get_deterministic_model(
            weights=weights,
            bases=bases,
        )
    # Load the transforms on the models.
    if intf is not None:
        base_gp_samples.input_transform = intf
        model.input_transform = intf
    if octf is not None:
        base_gp_samples.outcome_transform = octf
        model.outcome_transform = octf
    base_gp_samples._is_ensemble = is_ensemble(model=model)
    return base_gp_samples
