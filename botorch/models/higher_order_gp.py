#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
References

.. [Zhe2019hogp]
    S. Zhe, W. Xing, and R. M. Kirby. Scalable high-order gaussian process regression.
    Proceedings of Machine Learning Research, volume 89, Apr 2019.

.. [Doucet2010sampl]
    A. Doucet. A Note on Efficient Conditional Simulation of Gaussian Distributions.
    http://www.stats.ox.ac.uk/~doucet/doucet_simulationconditionalgaussian.pdf,
    Apr 2010.
"""

from __future__ import annotations

import warnings
from contextlib import ExitStack
from typing import Any, List, Optional, Union, Tuple

import torch
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.models.utils import gpt_posterior_settings
from botorch.posteriors import GPyTorchPosterior, TransformedPosterior
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel
from gpytorch.lazy import (
    BatchRepeatLazyTensor,
    DiagLazyTensor,
    KroneckerProductLazyTensor,
    LazyTensor,
    ZeroLazyTensor,
)
from gpytorch.likelihoods import (
    GaussianLikelihood,
    Likelihood,
)
from gpytorch.models import ExactGP
from gpytorch.priors.torch_priors import GammaPrior, MultivariateNormalPrior
from gpytorch.settings import fast_pred_var, skip_posterior_variances
from torch import Size, Tensor
from torch.nn import ModuleList, Parameter, ParameterList


MIN_INFERRED_NOISE_LEVEL = 1e-4


class FlattenedStandardize(Standardize):
    r"""
    Standardize outcomes in a structured multi-output settings by reshaping the
    batched output dimensions to be a vector. Specifically, an output dimension
    of [a x b x c] will be squeezed to be a vector of [a * b * c].
    """

    def __init__(
        self, output_shape: Size, batch_shape: Size = None, min_stdv: float = 1e-8
    ):
        if batch_shape is None:
            batch_shape = Size()

        super(FlattenedStandardize, self).__init__(
            m=1, outputs=None, batch_shape=batch_shape, min_stdv=min_stdv
        )

        self.output_shape = output_shape
        self.batch_shape = batch_shape

    def _squeeze_to_single_output(self, tsr: Tensor) -> Tensor:
        dim_ct = tsr.ndim - len(self.output_shape) - 1
        return tsr.reshape(*tsr.shape[:dim_ct], -1, 1)

    def _return_to_output_shape(self, tsr: Tensor) -> Tensor:
        out = tsr.reshape(*tsr.shape[:-2], -1, *self.output_shape)
        return out

    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        Y = self._squeeze_to_single_output(Y)
        if Yvar is not None:
            Yvar = self._squeeze_to_single_output(Yvar)

        Y, Yvar = super().forward(Y, Yvar)
        Y_out = self._return_to_output_shape(Y)

        if Yvar is not None:
            Yvar_out = self._return_to_output_shape(Yvar)
        else:
            Yvar_out = None
        return Y_out, Yvar_out

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        Y = self._squeeze_to_single_output(Y)
        if Yvar is not None:
            Yvar = self._squeeze_to_single_output(Yvar)

        Y, Yvar = super().untransform(Y, Yvar)

        Y = self._return_to_output_shape(Y)
        if Yvar is not None:
            Yvar = self._return_to_output_shape(Yvar)
        return Y, Yvar

    def untransform_posterior(
        self, posterior: HigherOrderGPPosterior
    ) -> TransformedPosterior:
        # TODO: return a HigherOrderGPPosterior once rescaling constant
        # muls * LazyTensors won't force a dense decomposition rather than a
        # Kronecker structured one.
        return TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda s: self._return_to_output_shape(
                self.means + self.stdvs * self._squeeze_to_single_output(s)
            ),
            mean_transform=lambda m, v: self._return_to_output_shape(
                self.means + self.stdvs * self._squeeze_to_single_output(m)
            ),
            variance_transform=lambda m, v: self._return_to_output_shape(
                self._stdvs_sq * self._squeeze_to_single_output(v)
            ),
        )


class HigherOrderGPPosterior(GPyTorchPosterior):
    r"""
    Posterior class for a Higher order Gaussian process model [Zhe2019hogp]. Extends the
    standard GPyTorch posterior class by overwriting the rsample method. The posterior
    variance is handled internally by the HigherOrderGP model.
    HOGP is a tensorized GP model so the posterior covariance grows to be extremely
    large, but is highly structured, which means that we can exploit Kronecker
    identities to sample from the posterior using Matheron's rule as described in
    [Doucet2010sampl]_. In general, this posterior should ONLY be used for HOGP models
    that have highly structured covariances. It should also only be used internally when
    called from the HigherOrderGP.posterior(...) method.
    """

    def __init__(
        self,
        mvn: MultivariateNormal,
        joint_covariance_matrix: LazyTensor,
        train_train_covar: LazyTensor,
        test_train_covar: LazyTensor,
        train_targets: Tensor,
        output_shape: Size,
        num_outputs: int,
    ) -> None:
        r"""A Posterior for HigherOrderGP models.

        Args:
            mvn: Posterior multivariate normal distribution
            joint_covariance_matrix: Joint test train covariance matrix over the entire
                tensor
            train_train_covar: covariance matrix of train points in the data space
            test_train_covar: covariance matrix of test x train points in the data space
            train_targets: training responses vectorized
            output_shape: shape output training responses
            num_outputs: batch shaping of model
        """
        super().__init__(mvn)
        self.joint_covariance_matrix = joint_covariance_matrix
        self.train_train_covar = train_train_covar
        self.test_train_covar = test_train_covar
        self.train_targets = train_targets
        self.output_shape = output_shape
        self._is_mt = True
        self.num_outputs = num_outputs

    @property
    def event_shape(self):
        # overwrites the standard event_shape call to inform samplers that
        # n + 2 n_train samples need to be drawn rather than n samples
        # TODO: Expose a sample shape property that is independent of the event shape
        # and handle those transparently in the samplers.
        batch_shape = self.joint_covariance_matrix.shape[:-2]
        sampling_shape = (
            self.joint_covariance_matrix.shape[-2] + self.train_train_covar.shape[-2]
        )
        return batch_shape + torch.Size((sampling_shape,))

    def _prepare_base_samples(
        self, sample_shape: torch.Size, base_samples: Tensor = None
    ) -> Tensor:
        covariance_matrix = self.joint_covariance_matrix
        joint_size = covariance_matrix.shape[-1]
        batch_shape = covariance_matrix.batch_shape

        if base_samples is not None:
            if base_samples.shape[: len(sample_shape)] != sample_shape:
                raise RuntimeError("sample_shape disagrees with shape of base_samples.")

            appended_shape = joint_size + self.train_train_covar.shape[-1]
            if appended_shape != base_samples.shape[-1]:
                # get base_samples to the correct shape by expanding as sample shape,
                # batch shape, then rest of dimensions. We have to add first the sample
                # shape, then the batch shape of the model, and then finally the shape
                # of the test data points squeezed into a single dimension, accessed
                # from the test_train_covar.
                base_sample_shapes = (
                    sample_shape + batch_shape + self.test_train_covar.shape[-2:-1]
                )
                if base_samples.nelement() == base_sample_shapes.numel():
                    base_samples = base_samples.reshape(base_sample_shapes)

                    new_base_samples = torch.randn(
                        *sample_shape,
                        *batch_shape,
                        appended_shape - base_samples.shape[-1],
                        device=base_samples.device,
                        dtype=base_samples.dtype,
                    )
                    base_samples = torch.cat((base_samples, new_base_samples), dim=-1)
                else:
                    # nuke the base samples if we cannot use them.
                    base_samples = None

        if base_samples is None:
            # TODO: Allow qMC sampling
            base_samples = torch.randn(
                *sample_shape,
                *batch_shape,
                joint_size,
                device=covariance_matrix.device,
                dtype=covariance_matrix.dtype,
            )

            noise_base_samples = torch.randn(
                *sample_shape,
                *batch_shape,
                self.train_train_covar.shape[-1],
                device=covariance_matrix.device,
                dtype=covariance_matrix.dtype,
            )
        else:
            # finally split up the base samples
            noise_base_samples = base_samples[..., joint_size:]
            base_samples = base_samples[..., :joint_size]

        perm_list = [*range(1, base_samples.ndim), 0]
        return base_samples.permute(*perm_list), noise_base_samples.permute(*perm_list)

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients).

        As the posterior covariance is difficult to draw from in this model,
        we implement Matheron's rule as described in [Doucet2010sampl]. This may not
        work entirely correctly for deterministic base samples unless base samples
        are provided that are of shape `n + 2 * n_train` because the sampling method
        draws `2 * n_train` samples as well as the standard `n`.
        samples.

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.
            base_samples: An (optional) Tensor of `N(0, I)` base samples of
                appropriate dimension, typically obtained from a `Sampler`.
                This is used for deterministic optimization.

        Returns:
            A `sample_shape x event_shape`-dim Tensor of samples from the posterior.
        """
        if sample_shape is None:
            sample_shape = torch.Size([1])

        base_samples, noise_base_samples = self._prepare_base_samples(
            sample_shape, base_samples
        )

        # base samples now have trailing sample dimension
        covariance_matrix = self.joint_covariance_matrix
        covar_root = covariance_matrix.root_decomposition().root
        samples = covar_root.matmul(base_samples)

        # now pluck out Y_x and X_x
        noiseless_train_marginal_samples = samples[
            ..., : self.train_train_covar.shape[-1], :
        ]
        test_marginal_samples = samples[..., self.train_train_covar.shape[-1] :, :]
        # we need to add noise to the train_joint_samples
        # THIS ASSUMES CONSTANT NOISE
        noise_std = self.train_train_covar.lazy_tensors[1]._diag[..., 0] ** 0.5
        # TODO: cleanup the reshaping here
        # expands the noise to allow broadcasting against the noise base samples
        # reshape_as or view_as don't work here because we need to expand to
        # broadcast against `samples x batch_shape x output_shape` while noise_std
        # is `batch_shape x 1`.
        if self.num_outputs > 1 or noise_std.ndim > 1:
            ntms_dims = [
                i == noise_std.shape[0] for i in noiseless_train_marginal_samples.shape
            ]
            for matched in ntms_dims:
                if not matched:
                    noise_std = noise_std.unsqueeze(-1)

        # we need to add noise into the noiseless samples
        noise_marginal_samples = noise_std * noise_base_samples

        train_marginal_samples = (
            noiseless_train_marginal_samples + noise_marginal_samples
        )

        # compute y - Y_x
        train_rhs = self.train_targets - train_marginal_samples

        # K_{train, train}^{-1} (y - Y_x)
        # internally, this solve is done using Kronecker algebra and is fast.
        kinv_rhs = self.train_train_covar.inv_matmul(train_rhs)
        # multiply by cross-covariance
        test_updated_samples = self.test_train_covar.matmul(kinv_rhs)

        # add samples
        test_cond_samples = test_marginal_samples + test_updated_samples
        test_cond_samples = test_cond_samples.permute(
            test_cond_samples.ndim - 1, *range(0, test_cond_samples.ndim - 1)
        )

        # reshape samples to be the actual size of the train targets
        return test_cond_samples.reshape(*sample_shape, *self.output_shape)


class HigherOrderGP(BatchedMultiOutputGPyTorchModel, ExactGP):
    r"""
    A Higher order Gaussian process model (HOGP) (predictions are matrices/tensors) as
    described in [Zhe2019hogp]_.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: Optional[Likelihood] = None,
        covar_modules: Optional[List[Kernel]] = None,
        num_latent_dims: Optional[List[int]] = None,
        learn_latent_pars: bool = True,
        latent_init: str = "default",
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ):
        r"""A HigherOrderGP model for high-dim output regression.

        Args:
            train_X: A `batch_shape x n x d`-dim tensor of training inputs.
            train_Y: A `batch_shape x n x output_shape`-dim tensor of training targets.
            likelihood: Gaussian likelihood for the model.
            covar_modules: List of kernels for each output structure.
            num_latent_dims: Sizes for the latent dimensions.
            learn_latent_pars: If true, learn the latent parameters.
            latent_init: [default or gp] how to initialize the latent parameters.
        """

        if input_transform is not None:
            input_transform.to(train_X)

        # infer the dimension of `output_shape`.
        num_output_dims = train_Y.dim() - train_X.dim() + 1
        batch_shape = train_X.shape[:-2]
        if len(batch_shape) > 1:
            raise NotImplementedError(
                "HigherOrderGP currently only supports 1-dim `batch_shape`."
            )

        if outcome_transform is not None:
            if isinstance(outcome_transform, Standardize) and not isinstance(
                outcome_transform, FlattenedStandardize
            ):
                warnings.warn(
                    "HigherOrderGP does not support the outcome_transform "
                    "`Standardize`! Using `FlattenedStandardize` with `output_shape="
                    f"{train_Y.shape[- num_output_dims:]} and batch_shape="
                    f"{batch_shape} instead.",
                    RuntimeWarning,
                )
                outcome_transform = FlattenedStandardize(
                    output_shape=train_Y.shape[-num_output_dims:],
                    batch_shape=batch_shape,
                )
            train_Y, _ = outcome_transform(train_Y)

        self._aug_batch_shape = batch_shape
        self._num_dimensions = num_output_dims + 1
        self._num_outputs = train_Y.shape[0] if batch_shape else 1
        self.target_shape = train_Y.shape[-num_output_dims:]
        self._input_batch_shape = batch_shape

        if likelihood is None:

            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                batch_shape=self._aug_batch_shape,
                noise_constraint=GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
        else:
            self._is_custom_likelihood = True

        super().__init__(
            train_X,
            train_Y.view(*self._aug_batch_shape, -1),
            likelihood=likelihood,
        )

        if covar_modules is not None:
            self.covar_modules = ModuleList(covar_modules)
        else:
            self.covar_modules = ModuleList(
                [
                    MaternKernel(
                        nu=2.5,
                        lengthscale_prior=GammaPrior(3.0, 6.0),
                        batch_shape=self._aug_batch_shape,
                        ard_num_dims=1 if dim > 0 else train_X.shape[-1],
                    )
                    for dim in range(self._num_dimensions)
                ]
            )

        if num_latent_dims is None:
            num_latent_dims = [1] * (self._num_dimensions - 1)

        self.to(train_X.device)

        self._initialize_latents(
            latent_init=latent_init,
            num_latent_dims=num_latent_dims,
            learn_latent_pars=learn_latent_pars,
            device=train_Y.device,
            dtype=train_Y.dtype,
        )

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

    def _initialize_latents(
        self,
        latent_init: str,
        num_latent_dims: List[int],
        learn_latent_pars: bool,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.latent_parameters = ParameterList()
        if latent_init == "default":
            for dim_num in range(len(self.covar_modules) - 1):
                self.latent_parameters.append(
                    Parameter(
                        torch.rand(
                            *self._aug_batch_shape,
                            self.target_shape[dim_num],
                            num_latent_dims[dim_num],
                            device=device,
                            dtype=dtype,
                        ),
                        requires_grad=learn_latent_pars,
                    )
                )
        elif latent_init == "gp":
            for dim_num, covar in enumerate(self.covar_modules[1:]):
                latent_covar = covar(
                    torch.linspace(
                        0.0,
                        1.0,
                        self.target_shape[dim_num],
                        device=device,
                        dtype=dtype,
                    )
                ).add_jitter(1e-4)
                latent_dist = MultivariateNormal(
                    torch.zeros(
                        self.target_shape[dim_num],
                        device=device,
                        dtype=dtype,
                    ),
                    latent_covar,
                )
                sample_shape = torch.Size(
                    (
                        *self._aug_batch_shape,
                        num_latent_dims[dim_num],
                    )
                )
                latent_sample = latent_dist.sample(sample_shape=sample_shape)
                latent_sample = latent_sample.reshape(
                    *self._aug_batch_shape,
                    self.target_shape[dim_num],
                    num_latent_dims[dim_num],
                )
                self.latent_parameters.append(
                    Parameter(
                        latent_sample,
                        requires_grad=learn_latent_pars,
                    )
                )
                self.register_prior(
                    "latent_parameters_" + str(dim_num),
                    MultivariateNormalPrior(
                        latent_dist.loc, latent_dist.covariance_matrix.detach().clone()
                    ),
                    lambda module, dim_num=dim_num: self.latent_parameters[dim_num],
                )

    def forward(self, X: Tensor) -> MultivariateNormal:
        X = self.transform_inputs(X)

        covariance_list = []
        covariance_list.append(self.covar_modules[0](X))

        for cm, param in zip(self.covar_modules[1:], self.latent_parameters):
            if not self.training:
                with torch.no_grad():
                    covariance_list.append(cm(param))
            else:
                covariance_list.append(cm(param))

        # check batch_shapes
        if covariance_list[0].batch_shape != covariance_list[1].batch_shape:
            for i in range(1, len(covariance_list)):
                cm = covariance_list[i]
                covariance_list[i] = BatchRepeatLazyTensor(
                    cm, covariance_list[0].batch_shape
                )
        kronecker_covariance = KroneckerProductLazyTensor(*covariance_list)

        # TODO: expand options for the mean module via batch shaping?
        mean = torch.zeros(
            *covariance_list[0].batch_shape,
            kronecker_covariance.shape[-1],
            device=kronecker_covariance.device,
            dtype=kronecker_covariance.dtype,
        )
        return MultivariateNormal(mean, kronecker_covariance)

    def get_fantasy_model(self, inputs, targets, **kwargs):
        # we need to squeeze the targets in order to preserve the shaping
        inputs_batch_dims = len(inputs.shape[:-2])
        target_shape = (*inputs.shape[:-2], -1)
        if (inputs_batch_dims + self._num_dimensions) < targets.ndim:
            target_shape = (targets.shape[0], *target_shape)
        reshaped_targets = targets.view(*target_shape)

        return super().get_fantasy_model(inputs, reshaped_targets, **kwargs)

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, **kwargs: Any
    ) -> HigherOrderGP:
        r"""Condition the model on new observations.
        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
            the feature space, `m` is the number of points per batch, and
            `batch_shape` is the batch shape (must be compatible with the
            batch shape of the model).

            Y: A `batch_shape' x n' x m_d`-dim Tensor, where `m_d` is the shaping
            of the model outputs, `n'` is the number of points per batch, and
            `batch_shape'` is the batch shape of the observations.
            `batch_shape'` must be broadcastable to `batch_shape` using
            standard broadcasting semantics. If `Y` has fewer batch dimensions
            than `X`, its is assumed that the missing batch dimensions are
            the same for all `Y`.
        Returns:
            A `BatchedMultiOutputGPyTorchModel` object of the same type with
            `n + n'` training examples, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).
        """
        noise = kwargs.get("noise")
        if hasattr(self, "outcome_transform"):
            # we need to apply transforms before shifting batch indices around
            Y, noise = self.outcome_transform(Y, noise)
        self._validate_tensor_args(X=X, Y=Y, Yvar=noise, strict=False)

        # we don't need to do un-squeezing because Y already is batched
        # we don't support fixed noise here yet
        # if noise is not None:
        #     kwargs.update({"noise": noise})
        fantasy_model = super(
            BatchedMultiOutputGPyTorchModel, self
        ).condition_on_observations(X=X, Y=Y, **kwargs)
        fantasy_model._input_batch_shape = fantasy_model.train_targets.shape[
            : (-1 if self._num_outputs == 1 else -2)
        ]
        fantasy_model._aug_batch_shape = fantasy_model.train_targets.shape[:-1]
        return fantasy_model

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        self.eval()  # make sure we're calling a posterior

        no_pred_variance = skip_posterior_variances._state

        with ExitStack() as es:
            es.enter_context(gpt_posterior_settings())
            es.enter_context(fast_pred_var(True))

            # we need to skip posterior variances here
            es.enter_context(skip_posterior_variances(True))
            mvn = self(X)
            if observation_noise is not False:
                # TODO: implement Kronecker + diagonal solves so that this is possible.
                # if torch.is_tensor(observation_noise):
                #     # TODO: Validate noise shape
                #     # make observation_noise `batch_shape x q x n`
                #     obs_noise = observation_noise.transpose(-1, -2)
                #     mvn = self.likelihood(mvn, X, noise=obs_noise)
                # elif isinstance(self.likelihood, FixedNoiseGaussianLikelihood):
                #     noise = self.likelihood.noise.mean().expand(X.shape[:-1])
                #     mvn = self.likelihood(mvn, X, noise=noise)
                # else:
                mvn = self.likelihood(mvn, X)

            # lazy covariance matrix includes the interpolated version of the full
            # covariance matrix so we can actually grab that instead.
            if X.ndimension() > self.train_inputs[0].ndimension():
                X_batch_shape = X.shape[:-2]
                train_inputs = self.train_inputs[0].reshape(
                    *[1] * len(X_batch_shape), *self.train_inputs[0].shape
                )
                train_inputs = train_inputs.repeat(
                    *X_batch_shape, *[1] * self.train_inputs[0].ndimension()
                )
            else:
                train_inputs = self.train_inputs[0]
            full_covar = self.covar_modules[0](torch.cat((train_inputs, X), dim=-2))

            if no_pred_variance:
                pred_variance = mvn.variance
            else:
                joint_covar = self._get_joint_covariance([X])
                pred_variance = self.make_posterior_variances(joint_covar)

                full_covar = KroneckerProductLazyTensor(
                    full_covar, *[x.detach() for x in joint_covar.lazy_tensors[1:]]
                )

            joint_covar_list = [self.covar_modules[0](X, train_inputs)]
            batch_shape = joint_covar_list[0].batch_shape
            for cm, param in zip(self.covar_modules[1:], self.latent_parameters):
                covar = cm(param)
                if covar.batch_shape != batch_shape:
                    covar = BatchRepeatLazyTensor(covar, batch_shape)
                joint_covar_list.append(covar.detach())

            test_train_covar = KroneckerProductLazyTensor(*joint_covar_list)

            # mean and variance get reshaped into the target shape
            new_mean = mvn.mean.reshape(*X.shape[:-1], *self.target_shape)
            if not no_pred_variance:
                new_variance = pred_variance.reshape(*X.shape[:-1], *self.target_shape)
                new_variance = DiagLazyTensor(new_variance)
            else:
                new_variance = ZeroLazyTensor(
                    *X.shape[:-1], *self.target_shape, self.target_shape[-1]
                )

            mvn = MultivariateNormal(new_mean, new_variance)

            # return a specialized Posterior to allow for sampling
            posterior = HigherOrderGPPosterior(
                mvn=mvn,
                train_targets=self.train_targets.unsqueeze(-1),
                train_train_covar=self.prediction_strategy.lik_train_train_covar.detach(),
                test_train_covar=test_train_covar,
                joint_covariance_matrix=full_covar.clone(),
                output_shape=Size(
                    (
                        *X.shape[:-1],
                        *self.target_shape,
                    )
                ),
                num_outputs=self._num_outputs,
            )
            if hasattr(self, "outcome_transform"):
                posterior = self.outcome_transform.untransform_posterior(posterior)

            return posterior

    # TODO: remove when this gets exposed in gpytorch
    def _get_joint_covariance(self, inputs):
        """
        Internal method to expose the joint test train covariance.
        """

        from gpytorch.models import ExactGP
        from gpytorch.utils.broadcasting import _mul_broadcast_shape

        train_inputs = self.train_inputs
        # Concatenate the input to the training input
        full_inputs = []
        batch_shape = train_inputs[0].shape[:-2]
        for train_input, input in zip(train_inputs, inputs):
            # Make sure the batch shapes agree for training/test data
            # This seems to be deprecated
            # if batch_shape != train_input.shape[:-2]:
            #     batch_shape = _mul_broadcast_shape(
            #         batch_shape, train_input.shape[:-2]
            #     )
            #     train_input = train_input.expand(
            #         *batch_shape, *train_input.shape[-2:]
            #     )
            if batch_shape != input.shape[:-2]:
                batch_shape = _mul_broadcast_shape(batch_shape, input.shape[:-2])
                train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                input = input.expand(*batch_shape, *input.shape[-2:])
            full_inputs.append(torch.cat([train_input, input], dim=-2))

        # Get the joint distribution for training/test data
        full_output = super(ExactGP, self).__call__(*full_inputs)
        return full_output.lazy_covariance_matrix

    def make_posterior_variances(self, joint_covariance_matrix: LazyTensor) -> Tensor:
        r"""
        Computes the posterior variances given the data points X. As currently
        implemented, it computes another forwards call with the stacked data to get out
        the joint covariance across all data points.
        """
        # TODO: use the exposed joint covariances from the prediction strategy
        data_joint_covariance = joint_covariance_matrix.lazy_tensors[
            0
        ].evaluate_kernel()
        num_train = self.train_inputs[0].shape[-2]
        test_train_covar = data_joint_covariance[..., num_train:, :num_train]
        train_train_covar = data_joint_covariance[..., :num_train, :num_train]
        test_test_covar = data_joint_covariance[..., num_train:, num_train:]

        full_train_train_covar = KroneckerProductLazyTensor(
            train_train_covar, *joint_covariance_matrix.lazy_tensors[1:]
        )
        full_test_test_covar = KroneckerProductLazyTensor(
            test_test_covar, *joint_covariance_matrix.lazy_tensors[1:]
        )
        full_test_train_covar_list = [test_train_covar] + [
            *joint_covariance_matrix.lazy_tensors[1:]
        ]

        train_evals, train_evecs = full_train_train_covar.symeig(eigenvectors=True)
        # (\kron \Lambda_i + \sigma^2 I)^{-1}
        train_inv_evals = DiagLazyTensor(1.0 / (train_evals + self.likelihood.noise))

        # compute K_i S_i \hadamard K_i S_i
        test_train_hadamard = KroneckerProductLazyTensor(
            *[
                lt1.matmul(lt2).evaluate() ** 2
                for lt1, lt2 in zip(
                    full_test_train_covar_list, train_evecs.lazy_tensors
                )
            ]
        )

        # and compute the column sums of
        #  (\kron K_i S_i * K_i S_i) \tilde{\Lambda}^{-1}
        test_train_pred_covar = test_train_hadamard.matmul(train_inv_evals).sum(dim=-1)

        pred_variances = full_test_test_covar.diag() - test_train_pred_covar
        return pred_variances
