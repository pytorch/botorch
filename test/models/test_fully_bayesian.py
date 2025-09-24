#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from unittest import mock
from unittest.mock import patch

import pyro

import torch
from botorch import fit_fully_bayesian_model_nuts, utils
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
    SampleReducingMCAcquisitionFunction,
)
from botorch.acquisition.multi_objective import (
    prune_inferior_points_multi_objective,
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.utils import prune_inferior_points
from botorch.models import ModelList, ModelListGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.fully_bayesian import (
    FullyBayesianLinearSingleTaskGP,
    FullyBayesianSingleTaskGP,
    LinearPyroModel,
    MaternPyroModel,
    MCMC_DIM,
    MIN_INFERRED_NOISE_LEVEL,
    PyroModel,
    SaasFullyBayesianSingleTaskGP,
    SaasPyroModel,
)
from botorch.models.transforms import Normalize, Standardize
from botorch.models.transforms.input import ChainedInputTransform, Warp
from botorch.posteriors.fully_bayesian import (
    batched_bisect,
    FullyBayesianPosterior,
    GaussianMixturePosterior,
)
from botorch.sampling.get_sampler import get_sampler
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.safe_math import logmeanexp
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.linear_kernel import LinearKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean
from linear_operator.operators import to_linear_operator
from pyro.ops.integrator import (
    _EXCEPTION_HANDLERS,
    potential_grad,
    register_exception_handler,
)


class CustomPyroModel(PyroModel):
    def sample(self) -> None:
        pass

    def postprocess_mcmc_samples(self, mcmc_samples, **kwargs) -> None:
        pass

    def load_mcmc_samples(self, mcmc_samples) -> None:
        pass


class TestSaasFullyBayesianSingleTaskGP(BotorchTestCase):
    model_cls: type[FullyBayesianSingleTaskGP] = SaasFullyBayesianSingleTaskGP
    pyro_model_cls: type[PyroModel] = SaasPyroModel
    model_kwargs = {}

    @property
    def expected_keys_warp(self) -> list[str]:
        return [
            "input_transform.concentration1_constraint.upper_bound",
            "input_transform.concentration0",
            "input_transform.concentration1_constraint.lower_bound",
            "input_transform._normalize._coefficient",
            "input_transform.concentration0_constraint.upper_bound",
            "input_transform._normalize.indices",
            "input_transform.concentration0_constraint.lower_bound",
            "input_transform.concentration1",
            "input_transform._normalize._offset",
            "input_transform.indices",
        ]

    @property
    def expected_keys(self) -> list[str]:
        expected_keys = [
            "mean_module.raw_constant",
        ]
        if self.model_cls is SaasFullyBayesianSingleTaskGP:
            expected_keys.extend(
                [
                    "covar_module.raw_outputscale",
                    "covar_module.raw_outputscale_constraint.lower_bound",
                    "covar_module.raw_outputscale_constraint.upper_bound",
                ]
            )
            covar_prefix = "covar_module.base_kernel"
        else:
            covar_prefix = "covar_module"

        expected_keys.extend(
            [
                ".".join([covar_prefix, name])
                for name in (
                    "raw_lengthscale",
                    "raw_lengthscale_constraint.lower_bound",
                    "raw_lengthscale_constraint.upper_bound",
                )
            ]
        )
        if self.model_kwargs.get("use_input_warping", False):
            expected_keys.extend(self.expected_keys_warp)
        return expected_keys

    @property
    def expected_keys_noise(self) -> list[str]:
        return self.expected_keys + [
            "likelihood.noise_covar.raw_noise",
            "likelihood.noise_covar.raw_noise_constraint.lower_bound",
            "likelihood.noise_covar.raw_noise_constraint.upper_bound",
        ]

    def _test_f(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sin(X[:, :1])

    def _get_data_and_model(
        self, infer_noise: bool, **tkwargs
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, FullyBayesianSingleTaskGP
    ]:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 4, **tkwargs)
            train_Y = self._test_f(X=train_X) + 0.1 * torch.randn(
                train_X.shape[0], 1, **tkwargs
            )
            train_Yvar = None if infer_noise else torch.full_like(train_Y, 0.01)
            model = self.model_cls(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                **self.model_kwargs,
            )
        return train_X, train_Y, train_Yvar, model

    def _get_unnormalized_data(
        self, infer_noise: bool, **tkwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = 5 + 5 * torch.rand(10, 4, **tkwargs)
            train_Y = 10 + torch.sin(train_X[:, :1])
            test_X = 5 + 5 * torch.rand(5, 4, **tkwargs)
            train_Yvar = (
                None if infer_noise else 0.1 * torch.arange(10, **tkwargs).unsqueeze(-1)
            )
        return train_X, train_Y, train_Yvar, test_X

    def _get_unnormalized_condition_data(
        self, num_models: int, num_cond: int, infer_noise: bool, **tkwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            cond_X = 5 + 5 * torch.rand(num_models, num_cond, 4, **tkwargs)
            cond_Y = 10 + torch.sin(cond_X[..., :1])
            cond_Yvar = (
                None if infer_noise else 0.1 * torch.ones(cond_Y.shape, **tkwargs)
            )
        return cond_X, cond_Y, cond_Yvar

    def _get_mcmc_samples(
        self, num_samples: int, dim: int, infer_noise: bool, **tkwargs
    ) -> dict[str, torch.Tensor]:
        mcmc_samples = {
            "lengthscale": torch.rand(num_samples, 1, dim, **tkwargs),
            "mean": torch.randn(num_samples, **tkwargs),
        }
        if infer_noise:
            mcmc_samples["noise"] = torch.rand(num_samples, 1, **tkwargs)
        if self.model_cls is SaasFullyBayesianSingleTaskGP:
            mcmc_samples["outputscale"] = torch.rand(num_samples, **tkwargs)
        if self.model_kwargs.get("use_input_warping", False):
            for k in ("c0", "c1"):
                mcmc_samples[k] = torch.rand(num_samples, 1, dim, **tkwargs)
        return mcmc_samples

    def test_raises(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_X to have shape n x d and train_Y to have shape n x 1",
        ):
            self.model_cls(
                train_X=torch.rand(10, 4, **tkwargs),
                train_Y=torch.randn(10, **tkwargs),
                **self.model_kwargs,
            )
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_X to have shape n x d and train_Y to have shape n x 1",
        ):
            self.model_cls(
                train_X=torch.rand(10, 4, **tkwargs),
                train_Y=torch.randn(12, 1, **tkwargs),
                **self.model_kwargs,
            )
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_X to have shape n x d and train_Y to have shape n x 1",
        ):
            self.model_cls(
                train_X=torch.rand(10, **tkwargs),
                train_Y=torch.randn(10, 1, **tkwargs),
                **self.model_kwargs,
            )
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_Yvar to be None or have the same shape as train_Y",
        ):
            self.model_cls(
                train_X=torch.rand(10, 4, **tkwargs),
                train_Y=torch.randn(10, 1, **tkwargs),
                train_Yvar=torch.rand(10, **tkwargs),
                **self.model_kwargs,
            )
        train_X, train_Y, train_Yvar, model = self._get_data_and_model(
            infer_noise=True, **tkwargs
        )
        # Make sure an exception is raised if the model has not been fitted
        not_fitted_error_msg = (
            "Model has not been fitted. You need to call "
            "`fit_fully_bayesian_model_nuts` to fit the model."
        )
        with self.assertRaisesRegex(RuntimeError, not_fitted_error_msg):
            model.num_mcmc_samples
        if issubclass(self.model_cls, FullyBayesianSingleTaskGP):
            with self.assertRaisesRegex(RuntimeError, not_fitted_error_msg):
                model.median_lengthscale
        else:
            with self.assertRaisesRegex(RuntimeError, not_fitted_error_msg):
                model.median_weight_variance
        with self.assertRaisesRegex(RuntimeError, not_fitted_error_msg):
            model.forward(torch.rand(1, 4, **tkwargs))
        with self.assertRaisesRegex(RuntimeError, not_fitted_error_msg):
            model.posterior(torch.rand(1, 4, **tkwargs))

    def test_fit_model(self) -> None:
        torch.manual_seed(16)
        for infer_noise, dtype in itertools.product(
            [True, False], [torch.float, torch.double]
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y, train_Yvar, model = self._get_data_and_model(
                infer_noise=infer_noise, **tkwargs
            )
            n, d = train_X.shape

            # Test init
            self.assertIsNone(model.mean_module)
            self.assertIsNone(model.covar_module)
            self.assertIsNone(model.likelihood)
            self.assertIsInstance(model.pyro_model, self.pyro_model_cls)
            self.assertAllClose(train_X, model.pyro_model.train_X)
            self.assertAllClose(train_Y, model.pyro_model.train_Y)
            if infer_noise:
                self.assertIsNone(model.pyro_model.train_Yvar)
            else:
                self.assertAllClose(
                    train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL),
                    model.pyro_model.train_Yvar,
                )

            # Fit a model and check that the hyperparameters have the correct shape
            fit_fully_bayesian_model_nuts(
                model, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
            )
            self.assertEqual(model.batch_shape, torch.Size([3]))
            self.assertEqual(model._aug_batch_shape, torch.Size([3]))
            # Using mock here since multi-output is currently not supported.
            with mock.patch.object(model, "_num_outputs", 2):
                self.assertEqual(model._aug_batch_shape, torch.Size([3, 2]))
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertEqual(model.mean_module.raw_constant.shape, model.batch_shape)
            if issubclass(self.model_cls, FullyBayesianSingleTaskGP):
                if self.model_cls is SaasFullyBayesianSingleTaskGP:
                    self.assertIsInstance(model.covar_module, ScaleKernel)
                    self.assertEqual(
                        model.covar_module.outputscale.shape, model.batch_shape
                    )
                    covar_module = model.covar_module.base_kernel
                else:
                    covar_module = model.covar_module
                self.assertIsInstance(covar_module, MaternKernel)
                self.assertEqual(
                    covar_module.lengthscale.shape,
                    torch.Size([3, 1, d]),
                )
            else:
                self.assertIsInstance(model.covar_module, LinearKernel)
                self.assertEqual(
                    model.covar_module.variance.shape, torch.Size([3, 1, d])
                )
                if self.model_kwargs.get("use_input_warping"):
                    self.assertIsInstance(model.input_transform, ChainedInputTransform)
                    tfs = list(model.input_transform.values())
                    warp = tfs[0]
                    self.assertIsInstance(warp, Warp)
                    for c in (warp.concentration0, warp.concentration0):
                        self.assertEqual(
                            c.shape,
                            torch.Size([3, 1, d]),
                        )
                    self.assertIsInstance(tfs[1], Normalize)
                else:
                    self.assertIsInstance(model.input_transform, Normalize)
            self.assertIsInstance(
                model.likelihood,
                GaussianLikelihood if infer_noise else FixedNoiseGaussianLikelihood,
            )
            if infer_noise:
                self.assertEqual(model.likelihood.noise.shape, torch.Size([3, 1]))
            else:
                self.assertEqual(model.likelihood.noise.shape, torch.Size([3, n]))
                self.assertAllClose(
                    train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL).squeeze(-1).repeat(3, 1),
                    model.likelihood.noise,
                )

            # Predict on some test points
            for batch_shape in [[5], [6, 5, 2]]:
                test_X = torch.rand(*batch_shape, d, **tkwargs)
                posterior = model.posterior(test_X)
                self.assertIsInstance(posterior, GaussianMixturePosterior)
                # Mean/variance
                expected_shape = (
                    *batch_shape[: MCMC_DIM + 2],
                    *model.batch_shape,
                    *batch_shape[MCMC_DIM + 2 :],
                    1,
                )
                expected_shape = torch.Size(expected_shape)
                mean, var = posterior.mean, posterior.variance
                self.assertEqual(mean.shape, expected_shape)
                self.assertEqual(var.shape, expected_shape)
                # Mixture mean/variance/covariance/median/quantiles
                mixture_mean = posterior.mixture_mean
                mixture_variance = posterior.mixture_variance
                mixture_covariance = posterior.mixture_covariance_matrix
                quantile1 = posterior.quantile(value=torch.tensor(0.01))
                quantile2 = posterior.quantile(value=torch.tensor(0.99))
                self.assertEqual(mixture_mean.shape, torch.Size(batch_shape + [1]))
                self.assertEqual(mixture_variance.shape, torch.Size(batch_shape + [1]))
                self.assertTrue(mixture_variance.min() > 0.0)
                self.assertEqual(
                    mixture_covariance.shape, torch.Size(batch_shape + batch_shape[-1:])
                )
                # Check that it is PSD.
                torch.linalg.cholesky(mixture_covariance.to_dense())
                self.assertEqual(quantile1.shape, torch.Size(batch_shape + [1]))
                self.assertEqual(quantile2.shape, torch.Size(batch_shape + [1]))
                self.assertTrue((quantile2 > quantile1).all())
                quantile12 = posterior.quantile(value=torch.tensor([0.01, 0.99]))
                self.assertAllClose(
                    quantile12, torch.stack([quantile1, quantile2], dim=0)
                )
                dist = torch.distributions.Normal(
                    loc=posterior.mean, scale=posterior.variance.sqrt()
                )
                self.assertAllClose(
                    dist.cdf(quantile1.unsqueeze(MCMC_DIM)).mean(dim=MCMC_DIM),
                    torch.full(batch_shape + [1], 0.01, **tkwargs),
                    atol=1e-6,
                )
                self.assertAllClose(
                    dist.cdf(quantile2.unsqueeze(MCMC_DIM)).mean(dim=MCMC_DIM),
                    torch.full(batch_shape + [1], 0.99, **tkwargs),
                    atol=1e-6,
                )
                # Invalid quantile should raise
                for q in [-1.0, 0.0, 1.0, 1.3333]:
                    with self.assertRaisesRegex(
                        ValueError, "value is expected to be in the range"
                    ):
                        posterior.quantile(value=torch.tensor(q))

                # Test model lists with fully Bayesian models and mixed modeling
                deterministic = GenericDeterministicModel(f=lambda x: x[..., :1])
                for ModelListClass, model2 in zip(
                    [ModelList, ModelListGP], [deterministic, model]
                ):
                    expected_shape = (
                        *batch_shape[: MCMC_DIM + 2],
                        *model.batch_shape,
                        *batch_shape[MCMC_DIM + 2 :],
                        2,
                    )
                    expected_shape = torch.Size(expected_shape)
                    model_list = ModelListClass(model, model2)
                    posterior = model_list.posterior(test_X)
                    mean, var = posterior.mean, posterior.variance
                    self.assertEqual(mean.shape, expected_shape)
                    self.assertEqual(var.shape, expected_shape)
                # This check is only for ModelListGP.
                self.assertEqual(model_list.batch_shape, model.batch_shape)

            # Mixing fully Bayesian models with different batch shapes isn't supported
            _, _, _, model2 = self._get_data_and_model(
                infer_noise=infer_noise, **tkwargs
            )
            fit_fully_bayesian_model_nuts(
                model2, warmup_steps=1, num_samples=1, thinning=1, disable_progbar=True
            )
            with self.assertRaisesRegex(
                NotImplementedError, "All MCMC batch dimensions"
            ):
                ModelList(model, model2).posterior(test_X)._extended_shape()
            with self.assertRaisesRegex(
                NotImplementedError,
                "All MCMC batch dimensions must have the same size, got",
            ):
                ModelList(model, model2).posterior(test_X).mean

            # Check properties
            if issubclass(self.model_cls, FullyBayesianSingleTaskGP):
                median_lengthscale = model.median_lengthscale
                self.assertEqual(median_lengthscale.shape, torch.Size([4]))
            else:
                median_weight_variance = model.median_weight_variance
                self.assertEqual(median_weight_variance.shape, torch.Size([4]))
            self.assertEqual(model.num_mcmc_samples, 3)

            # Check the keys in the state dict
            true_keys = self.expected_keys_noise if infer_noise else self.expected_keys
            self.assertEqual(set(model.state_dict().keys()), set(true_keys))

            for i in range(2):  # Test loading via state dict
                m = model if i == 0 else ModelList(model, deterministic)
                state_dict = m.state_dict()
                _, _, _, m_new = self._get_data_and_model(
                    infer_noise=infer_noise, **tkwargs
                )
                m_new = m_new if i == 0 else ModelList(m_new, deterministic)
                if i == 0:
                    self.assertEqual(m_new.state_dict(), {})
                m_new.load_state_dict(state_dict)
                self.assertEqual(m.state_dict().keys(), m_new.state_dict().keys())
                for k in m.state_dict().keys():
                    self.assertTrue((m.state_dict()[k] == m_new.state_dict()[k]).all())
                preds1, preds2 = m.posterior(test_X), m_new.posterior(test_X)
                self.assertTrue(torch.equal(preds1.mean, preds2.mean))
                self.assertTrue(torch.equal(preds1.variance, preds2.variance))

            # Make sure the model shapes are set correctly
            self.assertEqual(model.pyro_model.train_X.shape, torch.Size([n, d]))
            self.assertAllClose(model.pyro_model.train_X, train_X)
            # Put the model in eval mode with reset=True (reset should be ignored)
            trained_model = model.train(mode=False, reset=True)
            self.assertIs(trained_model, model)
            self.assertAllClose(train_X, model.pyro_model.train_X)
            self.assertIsNotNone(model.mean_module)
            self.assertIsNotNone(model.covar_module)
            self.assertIsNotNone(model.likelihood)
            # Put the model in train mode, without resetting
            trained_model = model.train(reset=False)
            self.assertIs(trained_model, model)
            self.assertAllClose(train_X, model.pyro_model.train_X)
            self.assertIsNotNone(model.mean_module)
            self.assertIsNotNone(model.covar_module)
            self.assertIsNotNone(model.likelihood)
            # Put the model in train mode, with resetting
            trained_model = model.train()
            self.assertIs(trained_model, model)
            self.assertAllClose(train_X, model.pyro_model.train_X)
            self.assertIsNone(model.mean_module)
            self.assertIsNone(model.covar_module)
            self.assertIsNone(model.likelihood)

    def test_empty(self) -> None:
        model = self.model_cls(
            train_X=torch.rand(0, 3),
            train_Y=torch.rand(0, 1),
            **self.model_kwargs,
        )
        fit_fully_bayesian_model_nuts(
            model, warmup_steps=2, num_samples=6, thinning=3, disable_progbar=True
        )
        self.assertEqual(model.num_mcmc_samples, 2)

    def test_transforms(self) -> None:
        for infer_noise in [True, False]:
            tkwargs = {"device": self.device, "dtype": torch.double}
            train_X, train_Y, train_Yvar, test_X = self._get_unnormalized_data(
                infer_noise=infer_noise, **tkwargs
            )

            lb, ub = train_X.min(dim=0).values, train_X.max(dim=0).values
            mu, sigma = train_Y.mean(), train_Y.std()

            # Fit without transforms
            with torch.random.fork_rng():
                torch.manual_seed(0)
                gp1 = self.model_cls(
                    train_X=(train_X - lb) / (ub - lb),
                    train_Y=(train_Y - mu) / sigma,
                    train_Yvar=(
                        train_Yvar / sigma**2 if train_Yvar is not None else train_Yvar
                    ),
                    **self.model_kwargs,
                )
                fit_fully_bayesian_model_nuts(
                    gp1, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
                )
                posterior1 = gp1.posterior((test_X - lb) / (ub - lb))
                pred_mean1 = mu + sigma * posterior1.mean
                pred_var1 = (sigma**2) * posterior1.variance

            # Fit with transforms
            with torch.random.fork_rng():
                torch.manual_seed(0)
                gp2 = self.model_cls(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_Yvar=train_Yvar,
                    input_transform=Normalize(d=train_X.shape[-1]),
                    outcome_transform=Standardize(m=1),
                    **self.model_kwargs,
                )
                fit_fully_bayesian_model_nuts(
                    gp2, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
                )
                posterior2 = gp2.posterior(test_X)
                pred_mean2, pred_var2 = posterior2.mean, posterior2.variance

            self.assertAllClose(pred_mean1, pred_mean2)
            self.assertAllClose(pred_var1, pred_var2)

            # check the transforms
            use_input_warping = self.model_kwargs.get("use_input_warping", False)
            if use_input_warping or (self.model_cls is FullyBayesianLinearSingleTaskGP):
                self.assertIsInstance(gp2.input_transform, ChainedInputTransform)
                tf_iter = iter(gp2.input_transform.values())
                tf = next(tf_iter)
                self.assertIsInstance(tf, Normalize)
                if use_input_warping:
                    tf = next(tf_iter)
                    self.assertIsInstance(tf, Warp)
                if self.model_cls is FullyBayesianLinearSingleTaskGP:
                    tf = next(tf_iter)
                    self.assertIsInstance(tf, Normalize)
                    self.assertEqual(tf.center, 0.0)
            else:
                self.assertIsInstance(gp2.input_transform, Normalize)

    def test_acquisition_functions(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X, train_Y, train_Yvar, model = self._get_data_and_model(
            infer_noise=True, **tkwargs
        )
        fit_fully_bayesian_model_nuts(
            model, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
        )
        deterministic = GenericDeterministicModel(f=lambda x: x[..., :1])
        # due to ModelList type, setting cache_root=False for all noisy EI variants
        list_gp = ModelListGP(model, model)
        mixed_list = ModelList(deterministic, model)
        simple_sampler = get_sampler(
            posterior=model.posterior(train_X), sample_shape=torch.Size([2])
        )
        list_gp_sampler = get_sampler(
            posterior=list_gp.posterior(train_X), sample_shape=torch.Size([2])
        )
        mixed_list_sampler = get_sampler(
            posterior=mixed_list.posterior(train_X), sample_shape=torch.Size([2])
        )
        acquisition_functions = [
            ExpectedImprovement(model=model, best_f=train_Y.max()),
            ProbabilityOfImprovement(model=model, best_f=train_Y.max()),
            PosteriorMean(model=model),
            UpperConfidenceBound(model=model, beta=4),
            qLogExpectedImprovement(
                model=model, best_f=train_Y.max(), sampler=simple_sampler
            ),
            qExpectedImprovement(
                model=model, best_f=train_Y.max(), sampler=simple_sampler
            ),
            qNoisyExpectedImprovement(
                model=model,
                X_baseline=train_X,
                sampler=simple_sampler,
                cache_root=False,
            ),
            qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=train_X,
                sampler=simple_sampler,
                cache_root=False,
            ),
            qProbabilityOfImprovement(
                model=model, best_f=train_Y.max(), sampler=simple_sampler
            ),
            qSimpleRegret(model=model, sampler=simple_sampler),
            qUpperConfidenceBound(model=model, beta=4, sampler=simple_sampler),
            qNoisyExpectedHypervolumeImprovement(
                model=list_gp,
                X_baseline=train_X,
                ref_point=torch.zeros(2, **tkwargs),
                sampler=list_gp_sampler,
                cache_root=False,
            ),
            qLogNoisyExpectedHypervolumeImprovement(
                model=list_gp,
                X_baseline=train_X,
                ref_point=torch.zeros(2, **tkwargs),
                sampler=list_gp_sampler,
                cache_root=False,
            ),
            qExpectedHypervolumeImprovement(
                model=list_gp,
                ref_point=torch.zeros(2, **tkwargs),
                sampler=list_gp_sampler,
                partitioning=NondominatedPartitioning(
                    ref_point=torch.zeros(2, **tkwargs), Y=train_Y.repeat([1, 2])
                ),
            ),
            qLogExpectedHypervolumeImprovement(
                model=list_gp,
                ref_point=torch.zeros(2, **tkwargs),
                sampler=list_gp_sampler,
                partitioning=NondominatedPartitioning(
                    ref_point=torch.zeros(2, **tkwargs), Y=train_Y.repeat([1, 2])
                ),
            ),
            # qEHVI/qNEHVI with mixed models
            qNoisyExpectedHypervolumeImprovement(
                model=mixed_list,
                X_baseline=train_X,
                ref_point=torch.zeros(2, **tkwargs),
                sampler=mixed_list_sampler,
                cache_root=False,
            ),
            qLogNoisyExpectedHypervolumeImprovement(
                model=mixed_list,
                X_baseline=train_X,
                ref_point=torch.zeros(2, **tkwargs),
                sampler=mixed_list_sampler,
                cache_root=False,
            ),
            qExpectedHypervolumeImprovement(
                model=mixed_list,
                ref_point=torch.zeros(2, **tkwargs),
                sampler=mixed_list_sampler,
                partitioning=NondominatedPartitioning(
                    ref_point=torch.zeros(2, **tkwargs), Y=train_Y.repeat([1, 2])
                ),
            ),
            qLogExpectedHypervolumeImprovement(
                model=mixed_list,
                ref_point=torch.zeros(2, **tkwargs),
                sampler=mixed_list_sampler,
                partitioning=NondominatedPartitioning(
                    ref_point=torch.zeros(2, **tkwargs), Y=train_Y.repeat([1, 2])
                ),
            ),
        ]

        for acqf, batch_shape in itertools.product(
            acquisition_functions, [[5], [6, 5, 2]]
        ):
            test_X = torch.rand(*batch_shape, 1, 4, **tkwargs)
            # Testing that the `average_over_ensemble_models` decorator works
            # correctly for fully Bayesian models with log-space acquisition
            # functions.
            with patch.object(utils.transforms, "logmeanexp", wraps=logmeanexp) as mock:
                self.assertEqual(acqf(test_X).shape, torch.Size(batch_shape))
                # The sample-reducing acquisition functions are using the
                # `sample_reduction` to average over the ensembles.
                if acqf._log and not isinstance(
                    acqf, SampleReducingMCAcquisitionFunction
                ):
                    mock.assert_called_once()
                else:
                    mock.assert_not_called()

        # Test prune_inferior_points
        X_pruned = prune_inferior_points(model=model, X=train_X)
        self.assertTrue(X_pruned.ndim == 2 and X_pruned.shape[-1] == 4)

        # Test prune_inferior_points_multi_objective
        for model_list in [ModelListGP(model, model), ModelList(deterministic, model)]:
            X_pruned = prune_inferior_points_multi_objective(
                model=model_list,
                X=train_X,
                ref_point=torch.zeros(2, **tkwargs),
            )
            self.assertTrue(X_pruned.ndim == 2 and X_pruned.shape[-1] == 4)

    def test_load_samples(self) -> None:
        for infer_noise, dtype in itertools.product(
            [True, False], [torch.float, torch.double]
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, _, train_Yvar, model = self._get_data_and_model(
                infer_noise=infer_noise, **tkwargs
            )
            mcmc_samples = self._get_mcmc_samples(
                num_samples=3, dim=train_X.shape[-1], infer_noise=infer_noise, **tkwargs
            )
            model.load_mcmc_samples(mcmc_samples)

            self.assertAllClose(
                model.mean_module.raw_constant.data, mcmc_samples["mean"]
            )
            if infer_noise:
                self.assertAllClose(
                    model.likelihood.noise_covar.noise, mcmc_samples["noise"]
                )
            else:
                self.assertAllClose(
                    model.likelihood.noise_covar.noise,
                    train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL).squeeze(-1).repeat(3, 1),
                )

            if issubclass(self.model_cls, FullyBayesianSingleTaskGP):
                if self.model_cls is SaasFullyBayesianSingleTaskGP:
                    self.assertAllClose(
                        model.covar_module.outputscale, mcmc_samples["outputscale"]
                    )
                    covar_module = model.covar_module.base_kernel
                else:
                    covar_module = model.covar_module
                self.assertAllClose(
                    covar_module.lengthscale, mcmc_samples["lengthscale"]
                )

            else:
                self.assertAllClose(
                    model.covar_module.variance, mcmc_samples["weight_variance"]
                )
                if self.model_kwargs.get("use_input_warping", False):
                    warp = list(model.input_transform.values())[0]
                    self.assertAllClose(warp.concentration0, mcmc_samples["c0"])
                    self.assertAllClose(warp.concentration1, mcmc_samples["c1"])

    def test_construct_inputs(self) -> None:
        for infer_noise, dtype in itertools.product(
            (True, False), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            X, Y, Yvar, model = self._get_data_and_model(
                infer_noise=infer_noise, **tkwargs
            )
            training_data = SupervisedDataset(
                X, Y, Yvar=Yvar, feature_names=["1", "2", "3", "4"], outcome_names=["1"]
            )

            data_dict = model.construct_inputs(training_data)
            self.assertTrue(X.equal(data_dict["train_X"]))
            self.assertTrue(Y.equal(data_dict["train_Y"]))
            if infer_noise:
                self.assertTrue("train_Yvar" not in data_dict)
            else:
                self.assertTrue(Yvar.equal(data_dict["train_Yvar"]))

    def test_condition_on_observations(self) -> None:
        # The following conditioned data shapes should work (output describes):
        # training data shape after cond(batch shape in output is req. in gpytorch)
        # X: num_models x n x d, Y: num_models x n x d --> num_models x n x d
        # X: n x d, Y: n x d --> num_models x n x d
        # X: n x d, Y: num_models x n x d --> num_models x n x d
        num_models = 3
        num_cond = 2
        for infer_noise, dtype in itertools.product(
            (True, False), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y, train_Yvar, test_X = self._get_unnormalized_data(
                infer_noise=infer_noise, **tkwargs
            )
            num_train, num_dims = train_X.shape
            # condition on different observations per model to obtain num_models sets
            # of training data
            cond_X, cond_Y, cond_Yvar = self._get_unnormalized_condition_data(
                num_models=num_models,
                num_cond=num_cond,
                infer_noise=infer_noise,
                **tkwargs,
            )
            model = self.model_cls(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                **self.model_kwargs,
            )
            mcmc_samples = self._get_mcmc_samples(
                num_samples=num_models,
                dim=train_X.shape[-1],
                infer_noise=infer_noise,
                **tkwargs,
            )
            model.load_mcmc_samples(mcmc_samples)

            # need to forward pass before conditioning
            model.posterior(train_X)
            cond_model = model.condition_on_observations(
                cond_X, cond_Y, noise=cond_Yvar
            )
            posterior = cond_model.posterior(test_X)
            self.assertEqual(
                posterior.mean.shape, torch.Size([num_models, len(test_X), 1])
            )

            # since the data is not equal for the conditioned points, a batch size
            # is added to the training data
            self.assertEqual(
                cond_model.train_inputs[0].shape,
                torch.Size([num_models, num_train + num_cond, num_dims]),
            )

            # the batch shape of the condition model is added during conditioning
            self.assertEqual(cond_model.batch_shape, torch.Size([num_models]))

            # condition on identical sets of data (i.e. one set) for all models
            # i.e, with no batch shape. This infers the batch shape.
            cond_X_nobatch, cond_Y_nobatch = cond_X[0], cond_Y[0]
            model = self.model_cls(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                **self.model_kwargs,
            )
            mcmc_samples = self._get_mcmc_samples(
                num_samples=num_models,
                dim=train_X.shape[-1],
                infer_noise=infer_noise,
                **tkwargs,
            )
            model.load_mcmc_samples(mcmc_samples)

            # conditioning without a batch size - the resulting conditioned model
            # will still have a batch size
            model.posterior(train_X)
            cond_model = model.condition_on_observations(
                cond_X_nobatch, cond_Y_nobatch, noise=cond_Yvar
            )
            self.assertEqual(
                cond_model.train_inputs[0].shape,
                torch.Size([num_models, num_train + num_cond, num_dims]),
            )

            # With batch size only on Y.
            cond_model = model.condition_on_observations(
                cond_X_nobatch, cond_Y, noise=cond_Yvar
            )
            self.assertEqual(
                cond_model.train_inputs[0].shape,
                torch.Size([num_models, num_train + num_cond, num_dims]),
            )

            # test repeated conditioning
            repeat_cond_X = cond_X + 5
            repeat_cond_model = cond_model.condition_on_observations(
                repeat_cond_X, cond_Y, noise=cond_Yvar
            )
            self.assertEqual(
                repeat_cond_model.train_inputs[0].shape,
                torch.Size([num_models, num_train + 2 * num_cond, num_dims]),
            )

            # test repeated conditioning without a batch size
            repeat_cond_X_nobatch = cond_X_nobatch + 10
            repeat_cond_model2 = repeat_cond_model.condition_on_observations(
                repeat_cond_X_nobatch, cond_Y_nobatch, noise=cond_Yvar
            )
            self.assertEqual(
                repeat_cond_model2.train_inputs[0].shape,
                torch.Size([num_models, num_train + 3 * num_cond, num_dims]),
            )

    def test_bisect(self) -> None:
        def f(x):
            return 1 + x

        for dtype, batch_shape in itertools.product(
            (torch.float, torch.double), ([5], [6, 5, 2])
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.stack(
                (
                    torch.zeros(batch_shape, **tkwargs),
                    torch.ones(batch_shape, **tkwargs),
                )
            )
            for target, tol in itertools.product([1.01, 1.5, 1.99], [1e-3, 1e-6]):
                x = batched_bisect(f=f, target=target, bounds=bounds, tol=tol)
                self.assertAllClose(
                    f(x), torch.full(batch_shape, target, **tkwargs), atol=tol
                )
            # Do one step and make sure we didn't converge in this case
            x = batched_bisect(f=f, target=1.71, bounds=bounds, max_steps=1)
            self.assertAllClose(x, torch.full(batch_shape, 0.75, **tkwargs), atol=tol)
            # Target outside the bounds should raise
            with self.assertRaisesRegex(
                ValueError,
                "The target is not contained in the interval specified by the bounds",
            ):
                batched_bisect(f=f, target=2.1, bounds=bounds)
            # Test analytic solution when there is only one MCMC sample
            mean = torch.randn(1, 5, **tkwargs)
            variance = torch.rand(1, 5, **tkwargs)
            covar = torch.diag_embed(variance)
            mvn = MultivariateNormal(mean, to_linear_operator(covar))
            posterior = GaussianMixturePosterior(distribution=mvn)
            dist = torch.distributions.Normal(
                loc=mean.unsqueeze(-1), scale=variance.unsqueeze(-1).sqrt()
            )
            for q in [0.1, 0.5, 0.9]:
                x = posterior.quantile(value=torch.tensor(q))
                self.assertAllClose(
                    dist.cdf(x), q * torch.ones(1, 5, 1, **tkwargs), atol=1e-4
                )

    def test_deprecated_posterior(self) -> None:
        mean = torch.randn(1, 5)
        variance = torch.rand(1, 5)
        covar = torch.diag_embed(variance)
        mvn = MultivariateNormal(mean, to_linear_operator(covar))
        with self.assertWarnsRegex(
            DeprecationWarning, "`FullyBayesianPosterior` is marked for deprecation"
        ):
            posterior = FullyBayesianPosterior(distribution=mvn)
        self.assertIsInstance(posterior, GaussianMixturePosterior)

    def test_predict_in_train_mode(self) -> None:
        torch.manual_seed(16)
        for infer_noise, dtype in itertools.product(
            [True, False], [torch.float, torch.double]
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y, train_Yvar, _ = self._get_data_and_model(
                infer_noise=infer_noise, **tkwargs
            )
            # Fit a model and check that the hyperparameters have the correct shape
            model = self.model_cls(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                input_transform=Normalize(d=train_X.shape[-1]),
                outcome_transform=Standardize(m=1),
                **self.model_kwargs,
            )
            fit_fully_bayesian_model_nuts(
                model, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
            )
            # check that input transforms are called when calling forward in train mode
            model.train(reset=False)
            with mock.patch.object(
                model.input_transform, "forward", wraps=model.input_transform.forward
            ) as mock_input_tf:
                with torch.no_grad():
                    model(*model.train_inputs)
                mock_input_tf.assert_called_once()


class TestFullyBayesianSingleTaskGP(TestSaasFullyBayesianSingleTaskGP):
    model_cls: type[FullyBayesianSingleTaskGP] = FullyBayesianSingleTaskGP
    pyro_model_cls: type[PyroModel] = MaternPyroModel


class TestSaasFullyBayesianSingleTaskGPWarped(TestSaasFullyBayesianSingleTaskGP):
    model_kwargs = {"use_input_warping": True}


class TestFullyBayesianSingleTaskGPWarped(TestFullyBayesianSingleTaskGP):
    model_kwargs = {"use_input_warping": True}


class TestPyroCatchNumericalErrors(BotorchTestCase):
    def tearDown(self) -> None:
        super().tearDown()
        # Remove exception handler so they don't affect the tests on rerun
        # TODO: Add functionality to pyro to clear the handlers so this
        # does not require touching the internals.
        del _EXCEPTION_HANDLERS["foo_runtime"]

    def test_pyro_catch_error(self) -> None:
        def potential_fn(z):
            mvn = pyro.distributions.MultivariateNormal(
                loc=torch.zeros(2),
                covariance_matrix=z["K"],
            )
            return mvn.log_prob(torch.zeros(2))

        # Test base case where everything is fine
        z = {"K": torch.eye(2)}
        grads, val = potential_grad(potential_fn, z)
        self.assertAllClose(grads["K"], -0.5 * torch.eye(2))
        norm_mvn = torch.distributions.Normal(0, 1)
        self.assertAllClose(val, 2 * norm_mvn.log_prob(torch.tensor(0.0)))

        # Default behavior should catch the ValueError when trying to instantiate
        # the MVN and return NaN instead
        z = {"K": torch.ones(2, 2)}
        _, val = potential_grad(potential_fn, z)
        self.assertTrue(torch.isnan(val))

        # Default behavior should catch the LinAlgError when peforming a
        # Cholesky decomposition and return NaN instead
        def potential_fn_chol(z) -> torch.Tensor:
            return torch.linalg.cholesky(z["K"])

        _, val = potential_grad(potential_fn_chol, z)
        self.assertTrue(torch.isnan(val))

        # Default behavior should not catch other errors
        def potential_fn_rterr_foo(z):
            raise RuntimeError("foo")

        with self.assertRaisesRegex(RuntimeError, "foo"):
            potential_grad(potential_fn_rterr_foo, z)

        # But once we register this specific error then it should
        def catch_runtime_error(e) -> bool:
            return type(e) is RuntimeError and "foo" in str(e)

        register_exception_handler("foo_runtime", catch_runtime_error)
        _, val = potential_grad(potential_fn_rterr_foo, z)
        self.assertTrue(torch.isnan(val))

        # Unless the error message is different
        def potential_fn_rterr_bar(z):
            raise RuntimeError("bar")

        with self.assertRaisesRegex(RuntimeError, "bar"):
            potential_grad(potential_fn_rterr_bar, z)


class TestFullyBayesianLinearSingleTaskGP(TestSaasFullyBayesianSingleTaskGP):
    model_cls = FullyBayesianLinearSingleTaskGP
    pyro_model_cls = LinearPyroModel
    model_kwargs = {"use_input_warping": False}

    def _test_f(self, X):
        return X.sum(dim=-1, keepdim=True)

    @property
    def expected_keys_warp(self) -> list[str]:
        return [
            "input_transform.warp.concentration1_constraint.upper_bound",
            "input_transform.warp.concentration0",
            "input_transform.warp.concentration1_constraint.lower_bound",
            "input_transform.normalize._coefficient",
            "input_transform.warp._normalize._coefficient",
            "input_transform.warp.concentration0_constraint.upper_bound",
            "input_transform.normalize._offset",
            "input_transform.warp._normalize.indices",
            "input_transform.warp.concentration0_constraint.lower_bound",
            "input_transform.warp.concentration1",
            "input_transform.warp._normalize._offset",
            "input_transform.warp.indices",
        ]

    @property
    def expected_keys(self) -> list[str]:
        expected_keys = [
            "mean_module.raw_constant",
            "covar_module.raw_variance",
            "covar_module.raw_variance_constraint.lower_bound",
            "covar_module.raw_variance_constraint.upper_bound",
        ]
        if self.model_kwargs["use_input_warping"]:
            expected_keys.extend(self.expected_keys_warp)
        else:
            expected_keys.extend(
                ["input_transform._offset", "input_transform._coefficient"]
            )
        return expected_keys

    def _get_mcmc_samples(
        self,
        num_samples: int,
        dim: int,
        infer_noise: bool,
        **tkwargs,
    ) -> dict[str, torch.Tensor]:
        mcmc_samples = {
            "weight_variance": torch.rand(num_samples, 1, dim, **tkwargs),
            "mean": torch.randn(num_samples, **tkwargs),
        }
        if infer_noise:
            mcmc_samples["noise"] = torch.rand(num_samples, 1, **tkwargs)
        if self.model_kwargs["use_input_warping"]:
            for k in ("c0", "c1"):
                mcmc_samples[k] = torch.rand(num_samples, 1, dim, **tkwargs)
        return mcmc_samples

    def test_custom_pyro_model(self) -> None:
        # custom pyro models are not supported by FullyBayesianLinearSingleTaskGP
        pass

    def test_empty(self) -> None:
        # TODO: support empty models with LinearKernels
        pass


class TestFullyBayesianLinearWarpingSingleTaskGP(TestFullyBayesianLinearSingleTaskGP):
    model_kwargs = {"use_input_warping": True}
