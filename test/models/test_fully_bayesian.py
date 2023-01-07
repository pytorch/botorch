#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from unittest import mock

import pyro

import torch
from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective import (
    prune_inferior_points_multi_objective,
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.utils import prune_inferior_points
from botorch.models import ModelList, ModelListGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.fully_bayesian import (
    MCMC_DIM,
    MIN_INFERRED_NOISE_LEVEL,
    PyroModel,
    SaasFullyBayesianSingleTaskGP,
    SaasPyroModel,
)
from botorch.models.transforms import Normalize, Standardize
from botorch.posteriors.fully_bayesian import batched_bisect, FullyBayesianPosterior
from botorch.sampling.get_sampler import get_sampler
from botorch.utils.datasets import FixedNoiseDataset, SupervisedDataset
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean
from linear_operator.operators import to_linear_operator
from pyro.ops.integrator import potential_grad, register_exception_handler


EXPECTED_KEYS = [
    "mean_module.raw_constant",
    "covar_module.raw_outputscale",
    "covar_module.base_kernel.raw_lengthscale",
    "covar_module.base_kernel.raw_lengthscale_constraint.lower_bound",
    "covar_module.base_kernel.raw_lengthscale_constraint.upper_bound",
    "covar_module.raw_outputscale_constraint.lower_bound",
    "covar_module.raw_outputscale_constraint.upper_bound",
]
EXPECTED_KEYS_NOISE = EXPECTED_KEYS + [
    "likelihood.noise_covar.raw_noise",
    "likelihood.noise_covar.raw_noise_constraint.lower_bound",
    "likelihood.noise_covar.raw_noise_constraint.upper_bound",
]


class CustomPyroModel(PyroModel):
    def sample(self) -> None:
        pass

    def postprocess_mcmc_samples(self, mcmc_samples, **kwargs):
        pass

    def load_mcmc_samples(self, mcmc_samples):
        pass


class TestFullyBayesianSingleTaskGP(BotorchTestCase):
    def _get_data_and_model(self, infer_noise: bool, **tkwargs):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 4, **tkwargs)
            train_Y = torch.sin(train_X[:, :1])
            train_Yvar = (
                None
                if infer_noise
                else torch.arange(0.1, 1.1, 0.1, **tkwargs).unsqueeze(-1)
            )
            model = SaasFullyBayesianSingleTaskGP(
                train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar
            )
        return train_X, train_Y, train_Yvar, model

    def _get_unnormalized_data(self, infer_noise: bool, **tkwargs):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = 5 + 5 * torch.rand(10, 4, **tkwargs)
            train_Y = 10 + torch.sin(train_X[:, :1])
            test_X = 5 + 5 * torch.rand(5, 4, **tkwargs)
            train_Yvar = (
                None if infer_noise else 0.1 * torch.arange(10, **tkwargs).unsqueeze(-1)
            )
        return train_X, train_Y, train_Yvar, test_X

    def _get_mcmc_samples(
        self, num_samples: int, dim: int, infer_noise: bool, **tkwargs
    ):
        mcmc_samples = {
            "lengthscale": torch.rand(num_samples, 1, dim, **tkwargs),
            "outputscale": torch.rand(num_samples, **tkwargs),
            "mean": torch.randn(num_samples, **tkwargs),
        }
        if infer_noise:
            mcmc_samples["noise"] = torch.rand(num_samples, 1, **tkwargs)
        return mcmc_samples

    def test_raises(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_X to have shape n x d and train_Y to have shape n x 1",
        ):
            SaasFullyBayesianSingleTaskGP(
                train_X=torch.rand(10, 4, **tkwargs), train_Y=torch.randn(10, **tkwargs)
            )
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_X to have shape n x d and train_Y to have shape n x 1",
        ):
            SaasFullyBayesianSingleTaskGP(
                train_X=torch.rand(10, 4, **tkwargs),
                train_Y=torch.randn(12, 1, **tkwargs),
            )
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_X to have shape n x d and train_Y to have shape n x 1",
        ):
            SaasFullyBayesianSingleTaskGP(
                train_X=torch.rand(10, **tkwargs),
                train_Y=torch.randn(10, 1, **tkwargs),
            )
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_Yvar to be None or have the same shape as train_Y",
        ):
            SaasFullyBayesianSingleTaskGP(
                train_X=torch.rand(10, 4, **tkwargs),
                train_Y=torch.randn(10, 1, **tkwargs),
                train_Yvar=torch.rand(10, **tkwargs),
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
        with self.assertRaisesRegex(RuntimeError, not_fitted_error_msg):
            model.median_lengthscale
        with self.assertRaisesRegex(RuntimeError, not_fitted_error_msg):
            model.forward(torch.rand(1, 4, **tkwargs))
        with self.assertRaisesRegex(RuntimeError, not_fitted_error_msg):
            model.posterior(torch.rand(1, 4, **tkwargs))

    def test_fit_model(self):
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
            self.assertIsInstance(model.pyro_model, SaasPyroModel)
            self.assertAllClose(train_X, model.pyro_model.train_X)
            self.assertAllClose(train_Y, model.pyro_model.train_Y)
            if infer_noise:
                self.assertIsNone(model.pyro_model.train_Yvar)
            else:
                self.assertTrue(
                    torch.allclose(
                        train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL),
                        model.pyro_model.train_Yvar,
                    )
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
            self.assertIsInstance(model.covar_module, ScaleKernel)
            self.assertEqual(model.covar_module.outputscale.shape, model.batch_shape)
            self.assertIsInstance(model.covar_module.base_kernel, MaternKernel)
            self.assertEqual(
                model.covar_module.base_kernel.lengthscale.shape, torch.Size([3, 1, d])
            )
            self.assertIsInstance(
                model.likelihood,
                GaussianLikelihood if infer_noise else FixedNoiseGaussianLikelihood,
            )
            if infer_noise:
                self.assertEqual(model.likelihood.noise.shape, torch.Size([3, 1]))
            else:
                self.assertEqual(model.likelihood.noise.shape, torch.Size([3, n]))
                self.assertTrue(
                    torch.allclose(
                        train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL)
                        .squeeze(-1)
                        .repeat(3, 1),
                        model.likelihood.noise,
                    )
                )

            # Predict on some test points
            for batch_shape in [[5], [6, 5, 2]]:
                test_X = torch.rand(*batch_shape, d, **tkwargs)
                posterior = model.posterior(test_X)
                self.assertIsInstance(posterior, FullyBayesianPosterior)
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
                # Mixture mean/variance/median/quantiles
                mixture_mean = posterior.mixture_mean
                mixture_variance = posterior.mixture_variance
                quantile1 = posterior.quantile(value=torch.tensor(0.01))
                quantile2 = posterior.quantile(value=torch.tensor(0.99))
                self.assertEqual(mixture_mean.shape, torch.Size(batch_shape + [1]))
                self.assertEqual(mixture_variance.shape, torch.Size(batch_shape + [1]))
                self.assertTrue(mixture_variance.min() > 0.0)
                self.assertEqual(quantile1.shape, torch.Size(batch_shape + [1]))
                self.assertEqual(quantile2.shape, torch.Size(batch_shape + [1]))
                self.assertTrue((quantile2 > quantile1).all())
                quantile12 = posterior.quantile(value=torch.tensor([0.01, 0.99]))
                self.assertTrue(
                    torch.allclose(
                        quantile12, torch.stack([quantile1, quantile2], dim=0)
                    )
                )
                dist = torch.distributions.Normal(
                    loc=posterior.mean, scale=posterior.variance.sqrt()
                )
                torch.allclose(
                    dist.cdf(quantile1.unsqueeze(MCMC_DIM)).mean(dim=MCMC_DIM),
                    0.05 * torch.ones(batch_shape + [1], **tkwargs),
                )
                torch.allclose(
                    dist.cdf(quantile2.unsqueeze(MCMC_DIM)).mean(dim=MCMC_DIM),
                    0.95 * torch.ones(batch_shape + [1], **tkwargs),
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
            median_lengthscale = model.median_lengthscale
            self.assertEqual(median_lengthscale.shape, torch.Size([4]))
            self.assertEqual(model.num_mcmc_samples, 3)

            # Check the keys in the state dict
            true_keys = EXPECTED_KEYS_NOISE if infer_noise else EXPECTED_KEYS
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
            model.train()  # Put the model in train mode
            self.assertAllClose(train_X, model.pyro_model.train_X)
            self.assertIsNone(model.mean_module)
            self.assertIsNone(model.covar_module)
            self.assertIsNone(model.likelihood)

    def test_transforms(self):
        for infer_noise in [True, False]:
            tkwargs = {"device": self.device, "dtype": torch.double}
            train_X, train_Y, train_Yvar, test_X = self._get_unnormalized_data(
                infer_noise=infer_noise, **tkwargs
            )
            n, d = train_X.shape

            lb, ub = train_X.min(dim=0).values, train_X.max(dim=0).values
            mu, sigma = train_Y.mean(), train_Y.std()

            # Fit without transforms
            with torch.random.fork_rng():
                torch.manual_seed(0)
                gp1 = SaasFullyBayesianSingleTaskGP(
                    train_X=(train_X - lb) / (ub - lb),
                    train_Y=(train_Y - mu) / sigma,
                    train_Yvar=train_Yvar / sigma**2
                    if train_Yvar is not None
                    else train_Yvar,
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
                gp2 = SaasFullyBayesianSingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_Yvar=train_Yvar,
                    input_transform=Normalize(d=train_X.shape[-1]),
                    outcome_transform=Standardize(m=1),
                )
                fit_fully_bayesian_model_nuts(
                    gp2, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
                )
                posterior2 = gp2.posterior(test_X)
                pred_mean2, pred_var2 = posterior2.mean, posterior2.variance

            self.assertAllClose(pred_mean1, pred_mean2)
            self.assertAllClose(pred_var1, pred_var2)

    def test_acquisition_functions(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X, train_Y, train_Yvar, model = self._get_data_and_model(
            infer_noise=True, **tkwargs
        )
        fit_fully_bayesian_model_nuts(
            model, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
        )
        deterministic = GenericDeterministicModel(f=lambda x: x[..., :1])
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
            qExpectedImprovement(
                model=model, best_f=train_Y.max(), sampler=simple_sampler
            ),
            qNoisyExpectedImprovement(
                model=model, X_baseline=train_X, sampler=simple_sampler
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
            ),
            qExpectedHypervolumeImprovement(
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
            ),
            qExpectedHypervolumeImprovement(
                model=mixed_list,
                ref_point=torch.zeros(2, **tkwargs),
                sampler=mixed_list_sampler,
                partitioning=NondominatedPartitioning(
                    ref_point=torch.zeros(2, **tkwargs), Y=train_Y.repeat([1, 2])
                ),
            ),
        ]

        for acqf in acquisition_functions:
            for batch_shape in [[5], [6, 5, 2]]:
                test_X = torch.rand(*batch_shape, 1, 4, **tkwargs)
                self.assertEqual(acqf(test_X).shape, torch.Size(batch_shape))

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

    def test_load_samples(self):
        for infer_noise, dtype in itertools.product(
            [True, False], [torch.float, torch.double]
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y, train_Yvar, model = self._get_data_and_model(
                infer_noise=infer_noise, **tkwargs
            )
            n, d = train_X.shape
            mcmc_samples = self._get_mcmc_samples(
                num_samples=3, dim=train_X.shape[-1], infer_noise=infer_noise, **tkwargs
            )
            model.load_mcmc_samples(mcmc_samples)

            self.assertTrue(
                torch.allclose(
                    model.covar_module.base_kernel.lengthscale,
                    mcmc_samples["lengthscale"],
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.covar_module.outputscale,
                    mcmc_samples["outputscale"],
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.mean_module.raw_constant.data,
                    mcmc_samples["mean"],
                )
            )
            if infer_noise:
                self.assertTrue(
                    torch.allclose(
                        model.likelihood.noise_covar.noise, mcmc_samples["noise"]
                    )
                )
            else:
                self.assertTrue(
                    torch.allclose(
                        model.likelihood.noise_covar.noise,
                        train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL)
                        .squeeze(-1)
                        .repeat(3, 1),
                    )
                )

    def test_construct_inputs(self):
        for infer_noise, dtype in itertools.product(
            (True, False), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            X, Y, Yvar, model = self._get_data_and_model(
                infer_noise=infer_noise, **tkwargs
            )
            if infer_noise:
                training_data = SupervisedDataset(X, Y)
            else:
                training_data = FixedNoiseDataset(X, Y, Yvar)

            data_dict = model.construct_inputs(training_data)
            self.assertTrue(X.equal(data_dict["train_X"]))
            self.assertTrue(Y.equal(data_dict["train_Y"]))
            if infer_noise:
                self.assertTrue("train_Yvar" not in data_dict)
            else:
                self.assertTrue(Yvar.equal(data_dict["train_Yvar"]))

    def test_custom_pyro_model(self):
        for infer_noise, dtype in itertools.product(
            (True, False), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y, train_Yvar, _ = self._get_unnormalized_data(
                infer_noise=infer_noise, **tkwargs
            )
            model = SaasFullyBayesianSingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                pyro_model=CustomPyroModel(),
            )
            with self.assertRaisesRegex(
                NotImplementedError, "load_state_dict only works for SaasPyroModel"
            ):
                model.load_state_dict({})
            self.assertIsInstance(model.pyro_model, CustomPyroModel)
            self.assertAllClose(model.pyro_model.train_X, train_X)
            self.assertAllClose(model.pyro_model.train_Y, train_Y)
            if infer_noise:
                self.assertIsNone(model.pyro_model.train_Yvar)
            else:
                self.assertTrue(
                    torch.allclose(
                        model.pyro_model.train_Yvar,
                        train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL),
                    )
                )
            # Use transforms
            model = SaasFullyBayesianSingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                input_transform=Normalize(d=train_X.shape[-1]),
                outcome_transform=Standardize(m=1),
                pyro_model=CustomPyroModel(),
            )
            self.assertIsInstance(model.pyro_model, CustomPyroModel)
            lb, ub = train_X.min(dim=0).values, train_X.max(dim=0).values
            self.assertTrue(
                torch.allclose(model.pyro_model.train_X, (train_X - lb) / (ub - lb))
            )
            mu, sigma = train_Y.mean(dim=0), train_Y.std(dim=0)
            self.assertTrue(
                torch.allclose(model.pyro_model.train_Y, (train_Y - mu) / sigma)
            )
            if not infer_noise:
                self.assertTrue(
                    torch.allclose(
                        model.pyro_model.train_Yvar,
                        train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL) / (sigma**2),
                        atol=1e-4,
                    )
                )

    def test_bisect(self):
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
                self.assertTrue(
                    torch.allclose(
                        f(x), target * torch.ones(batch_shape, **tkwargs), atol=tol
                    )
                )
            # Do one step and make sure we didn't converge in this case
            x = batched_bisect(f=f, target=1.71, bounds=bounds, max_steps=1)
            self.assertTrue(
                torch.allclose(x, 0.75 * torch.ones(batch_shape, **tkwargs), atol=tol)
            )
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
            posterior = FullyBayesianPosterior(distribution=mvn)
            dist = torch.distributions.Normal(
                loc=mean.unsqueeze(-1), scale=variance.unsqueeze(-1).sqrt()
            )
            for q in [0.1, 0.5, 0.9]:
                x = posterior.quantile(value=torch.tensor(q))
                self.assertTrue(
                    torch.allclose(
                        dist.cdf(x), q * torch.ones(1, 5, **tkwargs), atol=1e-4
                    )
                )


class TestPyroCatchNumericalErrors(BotorchTestCase):
    def test_pyro_catch_error(self):
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
        def potential_fn_chol(z):
            return torch.linalg.cholesky(z["K"])

        _, val = potential_grad(potential_fn_chol, z)
        self.assertTrue(torch.isnan(val))

        # Default behavior should not catch other errors
        def potential_fn_rterr_foo(z):
            raise RuntimeError("foo")

        with self.assertRaisesRegex(RuntimeError, "foo"):
            potential_grad(potential_fn_rterr_foo, z)

        # But once we register this specific error then it should
        def catch_runtime_error(e):
            return type(e) == RuntimeError and "foo" in str(e)

        register_exception_handler("foo_runtime", catch_runtime_error)
        _, val = potential_grad(potential_fn_rterr_foo, z)
        self.assertTrue(torch.isnan(val))

        # Unless the error message is different
        def potential_fn_rterr_bar(z):
            raise RuntimeError("bar")

        with self.assertRaisesRegex(RuntimeError, "bar"):
            potential_grad(potential_fn_rterr_bar, z)
