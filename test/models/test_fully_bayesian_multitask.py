#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools

import torch
from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition.analytic import (
    LogExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.monte_carlo import (
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)

from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models import ModelList, ModelListGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.fully_bayesian import MCMC_DIM, MIN_INFERRED_NOISE_LEVEL
from botorch.models.fully_bayesian_multitask import (
    MultitaskSaasPyroModel,
    SaasFullyBayesianMultiTaskGP,
)
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import GaussianMixturePosterior
from botorch.sampling.get_sampler import get_sampler
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.test_helpers import gen_multi_task_dataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.means import ConstantMean

EXPECTED_KEYS = [
    "latent_features",
    "mean_module.raw_constant",
    "covar_module.raw_outputscale",
    "covar_module.base_kernel.raw_lengthscale",
    "covar_module.base_kernel.raw_lengthscale_constraint.lower_bound",
    "covar_module.base_kernel.raw_lengthscale_constraint.upper_bound",
    "covar_module.raw_outputscale_constraint.lower_bound",
    "covar_module.raw_outputscale_constraint.upper_bound",
    "task_covar_module.raw_lengthscale",
    "task_covar_module.raw_lengthscale_constraint.lower_bound",
    "task_covar_module.raw_lengthscale_constraint.upper_bound",
]
EXPECTED_KEYS_NOISE = EXPECTED_KEYS + [
    "likelihood.noise_covar.raw_noise",
    "likelihood.noise_covar.raw_noise_constraint.lower_bound",
    "likelihood.noise_covar.raw_noise_constraint.upper_bound",
]


class TestFullyBayesianMultiTaskGP(BotorchTestCase):
    def _get_data_and_model(
        self,
        task_rank: int | None = None,
        output_tasks: list[int] | None = None,
        infer_noise: bool = False,
        use_outcome_transform: bool = True,
        **tkwargs,
    ):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 4, **tkwargs)
        task_indices = torch.cat(
            [torch.zeros(5, 1, **tkwargs), torch.ones(5, 1, **tkwargs)], dim=0
        )
        self.num_tasks = 2
        train_X = torch.cat([train_X, task_indices], dim=1)
        train_Y = torch.sin(train_X[:, :1])
        train_Yvar = 0.5 * torch.arange(10, **tkwargs).unsqueeze(-1)
        model = SaasFullyBayesianMultiTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=None if infer_noise else train_Yvar,
            task_feature=4,
            output_tasks=output_tasks,
            rank=task_rank,
            outcome_transform=(
                Standardize(m=1, batch_shape=train_X.shape[:-2])
                if use_outcome_transform
                else None
            ),
        )
        return train_X, train_Y, train_Yvar, model

    def _get_unnormalized_data(self, **tkwargs):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 4, **tkwargs)
            train_Y = torch.sin(train_X[:, :1])
            task_indices = torch.cat(
                [torch.zeros(5, 1, **tkwargs), torch.ones(5, 1, **tkwargs)], dim=0
            )
            train_X = torch.cat([5 + 5 * train_X, task_indices], dim=1)
            test_X = 5 + 5 * torch.rand(5, 4, **tkwargs)
            train_Yvar = 0.1 * torch.arange(10, **tkwargs).unsqueeze(-1)
        return train_X, train_Y, train_Yvar, test_X

    def _get_mcmc_samples(self, num_samples: int, dim: int, task_rank: int, **tkwargs):
        mcmc_samples = {
            "lengthscale": torch.rand(num_samples, 1, dim, **tkwargs),
            "outputscale": torch.rand(num_samples, **tkwargs),
            "mean": torch.randn(num_samples, **tkwargs),
            "noise": torch.rand(num_samples, 1, **tkwargs),
            "task_lengthscale": torch.rand(num_samples, 1, task_rank, **tkwargs),
            "latent_features": torch.rand(
                num_samples, self.num_tasks, task_rank, **tkwargs
            ),
        }
        return mcmc_samples

    def test_raises(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_X to have shape n x d and train_Y to have shape n x 1",
        ):
            SaasFullyBayesianMultiTaskGP(
                train_X=torch.rand(10, 4, **tkwargs),
                train_Y=torch.randn(10, **tkwargs),
                train_Yvar=torch.rand(10, 1, **tkwargs),
                task_feature=4,
            )
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_X to have shape n x d and train_Y to have shape n x 1",
        ):
            SaasFullyBayesianMultiTaskGP(
                train_X=torch.rand(10, 4, **tkwargs),
                train_Y=torch.randn(12, 1, **tkwargs),
                train_Yvar=torch.rand(12, 1, **tkwargs),
                task_feature=4,
            )
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_X to have shape n x d and train_Y to have shape n x 1",
        ):
            SaasFullyBayesianMultiTaskGP(
                train_X=torch.rand(10, **tkwargs),
                train_Y=torch.randn(10, 1, **tkwargs),
                train_Yvar=torch.rand(10, 1, **tkwargs),
                task_feature=4,
            )
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_Yvar to be None or have the same shape as train_Y",
        ):
            SaasFullyBayesianMultiTaskGP(
                train_X=torch.rand(10, 4, **tkwargs),
                train_Y=torch.randn(10, 1, **tkwargs),
                train_Yvar=torch.rand(10, **tkwargs),
                task_feature=4,
            )
        train_X, train_Y, train_Yvar, model = self._get_data_and_model(**tkwargs)
        with self.assertRaisesRegex(
            NotImplementedError, "`all_tasks` argument is not supported"
        ):
            SaasFullyBayesianMultiTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                task_feature=-1,
                all_tasks=[0, 1, 2, 3],
            )
        sampler = IIDNormalSampler(sample_shape=torch.Size([2]))
        with self.assertRaisesRegex(
            NotImplementedError, "Fantasize is not implemented!"
        ):
            model.fantasize(
                X=torch.cat(
                    [torch.rand(5, 4, **tkwargs), torch.ones(5, 1, **tkwargs)], dim=1
                ),
                sampler=sampler,
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

    def test_fit_model(
        self,
        dtype: torch.dtype = torch.double,
        infer_noise: bool = False,
        task_rank: int = 1,
        use_outcome_transform: bool = False,
    ):
        tkwargs = {"device": self.device, "dtype": dtype}
        train_X, train_Y, train_Yvar, model = self._get_data_and_model(
            infer_noise=infer_noise,
            task_rank=task_rank,
            use_outcome_transform=use_outcome_transform,
            **tkwargs,
        )
        n = train_X.shape[0]
        d = train_X.shape[1] - 1

        # Handle outcome transforms (if used)
        train_Y_tf, train_Yvar_tf = train_Y, train_Yvar
        if use_outcome_transform:
            train_Y_tf, train_Yvar_tf = model.outcome_transform(
                Y=train_Y, Yvar=train_Yvar
            )

        # Test init
        self.assertIsNone(model.mean_module)
        self.assertIsNone(model.covar_module)
        self.assertIsNone(model.likelihood)
        self.assertIsInstance(model.pyro_model, MultitaskSaasPyroModel)
        self.assertAllClose(train_X, model.pyro_model.train_X)
        self.assertAllClose(train_Y_tf, model.pyro_model.train_Y)
        if infer_noise:
            self.assertIsNone(model.pyro_model.train_Yvar)
        else:
            self.assertAllClose(
                train_Yvar_tf.clamp(MIN_INFERRED_NOISE_LEVEL),
                model.pyro_model.train_Yvar,
            )

        # Fit a model and check that the hyperparameters have the correct shape
        fit_fully_bayesian_model_nuts(
            model, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
        )
        self.assertEqual(model.batch_shape, torch.Size([3]))
        self.assertIsInstance(model.mean_module, ConstantMean)
        self.assertEqual(model.mean_module.raw_constant.shape, model.batch_shape)
        self.assertIsInstance(model.covar_module, ScaleKernel)
        self.assertEqual(model.covar_module.outputscale.shape, model.batch_shape)
        self.assertIsInstance(model.covar_module.base_kernel, MaternKernel)
        self.assertEqual(
            model.covar_module.base_kernel.lengthscale.shape, torch.Size([3, 1, d])
        )
        if infer_noise:
            self.assertIsInstance(model.likelihood, GaussianLikelihood)
            self.assertEqual(
                model.likelihood.noise_covar.noise.shape, torch.Size([3, 1])
            )
        else:
            self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)
        self.assertIsInstance(model.task_covar_module, MaternKernel)
        self.assertEqual(
            model.task_covar_module.lengthscale.shape, torch.Size([3, 1, task_rank])
        )
        self.assertEqual(
            model.latent_features.shape, torch.Size([3, self.num_tasks, task_rank])
        )

        # Predict on some test points
        for batch_shape in [[5], [5, 2], [5, 2, 6]]:
            test_X = torch.rand(*batch_shape, d, **tkwargs)
            posterior = model.posterior(test_X)
            self.assertIsInstance(posterior, GaussianMixturePosterior)
            self.assertIsInstance(posterior, GaussianMixturePosterior)

            test_X = torch.rand(*batch_shape, d, **tkwargs)
            posterior = model.posterior(test_X)
            self.assertIsInstance(posterior, GaussianMixturePosterior)
            # Mean/variance
            expected_shape = (
                *batch_shape[: MCMC_DIM + 2],
                *model.batch_shape,
                *batch_shape[MCMC_DIM + 2 :],
                self.num_tasks,
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

            # Marginalized mean/variance
            self.assertEqual(
                mixture_mean.shape, torch.Size(batch_shape + [self.num_tasks])
            )
            self.assertEqual(
                mixture_variance.shape, torch.Size(batch_shape + [self.num_tasks])
            )
            self.assertTrue(mixture_variance.min() > 0.0)
            self.assertEqual(
                quantile1.shape, torch.Size(batch_shape + [self.num_tasks])
            )
            self.assertEqual(
                quantile2.shape, torch.Size(batch_shape + [self.num_tasks])
            )
            self.assertTrue((quantile2 > quantile1).all())

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
            for ModelListClass, models, expected_outputs in zip(
                [ModelList, ModelListGP],
                [[deterministic, model], [model, model]],
                [3, 4],
            ):
                expected_shape = (
                    *batch_shape[: MCMC_DIM + 2],
                    *model.batch_shape,
                    *batch_shape[MCMC_DIM + 2 :],
                    expected_outputs,
                )
                expected_shape = torch.Size(expected_shape)
                model_list = ModelListClass(*models)
                posterior = model_list.posterior(test_X)
                mean, var = posterior.mean, posterior.variance
                self.assertEqual(mean.shape, expected_shape)
                self.assertEqual(var.shape, expected_shape)

        # Check properties
        median_lengthscale = model.median_lengthscale
        self.assertEqual(median_lengthscale.shape, torch.Size([d]))
        self.assertEqual(model.num_mcmc_samples, 3)

        # Check the keys in the state dict
        true_keys = EXPECTED_KEYS_NOISE if infer_noise else EXPECTED_KEYS
        if use_outcome_transform:
            true_keys = true_keys + [
                "outcome_transform.stdvs",
                "outcome_transform._is_trained",
                "outcome_transform._stdvs_sq",
                "outcome_transform.means",
            ]
        self.assertEqual(set(model.state_dict().keys()), set(true_keys))

        # Check that we can load the state dict.
        state_dict = model.state_dict()
        _, _, _, m_new = self._get_data_and_model(
            infer_noise=infer_noise,
            task_rank=task_rank,
            use_outcome_transform=use_outcome_transform,
            **tkwargs,
        )
        expected_state_dict = {}
        if use_outcome_transform:
            expected_state_dict.update(
                {
                    "outcome_transform." + k: v
                    for k, v in model.outcome_transform.state_dict().items()
                }
            )
        self.assertEqual(m_new.state_dict(), expected_state_dict)
        m_new.load_state_dict(state_dict)
        self.assertEqual(model.state_dict().keys(), m_new.state_dict().keys())
        for k in model.state_dict().keys():
            self.assertTrue((model.state_dict()[k] == m_new.state_dict()[k]).all())
        preds1, preds2 = model.posterior(test_X), m_new.posterior(test_X)
        self.assertTrue(torch.equal(preds1.mean, preds2.mean))
        self.assertTrue(torch.equal(preds1.variance, preds2.variance))

        # Make sure the model shapes are set correctly
        self.assertEqual(model.pyro_model.train_X.shape, torch.Size([n, d + 1]))
        self.assertAllClose(model.pyro_model.train_X, train_X)
        model.train()  # Put the model in train mode
        self.assertAllClose(train_X, model.pyro_model.train_X)
        self.assertIsNone(model.mean_module)
        self.assertIsNone(model.covar_module)
        self.assertIsNone(model.likelihood)
        self.assertIsNone(model.task_covar_module)

    def test_fit_model_float(self):
        self.test_fit_model(dtype=torch.float)

    def test_fit_model_infer_noise(self):
        self.test_fit_model(infer_noise=True, task_rank=2)

    def test_fit_model_with_outcome_transform(self):
        self.test_fit_model(use_outcome_transform=True)

    def test_transforms(self, infer_noise: bool = False):
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X, train_Y, train_Yvar, test_X = self._get_unnormalized_data(**tkwargs)
        n, d = train_X.shape
        normalize_indices = torch.tensor(
            list(range(train_X.shape[-1] - 1)), device=self.device
        )

        lb, ub = (
            train_X[:, normalize_indices].min(dim=0).values,
            train_X[:, normalize_indices].max(dim=0).values,
        )
        train_X_new = train_X.clone()
        train_X_new[..., normalize_indices] = (train_X[..., normalize_indices] - lb) / (
            ub - lb
        )
        # TODO: add testing of stratified standardization
        mu, sigma = train_Y.mean(), train_Y.std()

        # Fit without transforms
        with torch.random.fork_rng():
            torch.manual_seed(0)
            gp1 = SaasFullyBayesianMultiTaskGP(
                train_X=train_X_new,
                train_Y=(train_Y - mu) / sigma,
                train_Yvar=None if infer_noise else train_Yvar / sigma**2,
                task_feature=d - 1,
            )
            fit_fully_bayesian_model_nuts(
                gp1, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
            )
        posterior1 = gp1.posterior((test_X - lb) / (ub - lb), output_indices=[0])
        pred_mean1 = mu + sigma * posterior1.mean
        pred_var1 = (sigma**2) * posterior1.variance

        # Fit with transforms
        with torch.random.fork_rng():
            torch.manual_seed(0)
            gp2 = SaasFullyBayesianMultiTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=None if infer_noise else train_Yvar,
                task_feature=d - 1,
                input_transform=Normalize(
                    d=train_X.shape[-1], indices=normalize_indices
                ),
                outcome_transform=Standardize(m=1),
            )
            fit_fully_bayesian_model_nuts(
                gp2, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
            )
            posterior2 = gp2.posterior(X=test_X, output_indices=[0])
            pred_mean2, pred_var2 = posterior2.mean, posterior2.variance

        self.assertAllClose(pred_mean1, pred_mean2)
        self.assertAllClose(pred_var1, pred_var2)

    def test_transforms_infer_noise(self):
        self.test_transforms(infer_noise=True)

    def test_acquisition_functions(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        # Using a single output model here since we test with single objective acqfs.
        train_X, train_Y, train_Yvar, model = self._get_data_and_model(
            task_rank=1, output_tasks=[0], **tkwargs
        )
        fit_fully_bayesian_model_nuts(
            model, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
        )
        for include_task_feature in [True, False]:
            if not include_task_feature:
                test_X = train_X[..., :-1]
            else:
                test_X = train_X

            deterministic = GenericDeterministicModel(f=lambda x: x[..., :1])
            list_gp = ModelListGP(model, model)
            mixed_list = ModelList(deterministic, model)
            simple_sampler = get_sampler(
                posterior=model.posterior(test_X),
                sample_shape=torch.Size([2]),
            )
            list_gp_sampler = get_sampler(
                posterior=list_gp.posterior(test_X), sample_shape=torch.Size([2])
            )
            mixed_list_sampler = get_sampler(
                posterior=mixed_list.posterior(test_X), sample_shape=torch.Size([2])
            )
            acquisition_functions = [
                LogExpectedImprovement(model=model, best_f=train_Y.max()),
                ProbabilityOfImprovement(model=model, best_f=train_Y.max()),
                PosteriorMean(model=model),
                UpperConfidenceBound(model=model, beta=4),
                qLogExpectedImprovement(
                    model=model, best_f=train_Y.max(), sampler=simple_sampler
                ),
                qLogNoisyExpectedImprovement(
                    model=model, X_baseline=test_X, sampler=simple_sampler
                ),
                qProbabilityOfImprovement(
                    model=model, best_f=train_Y.max(), sampler=simple_sampler
                ),
                qSimpleRegret(model=model, sampler=simple_sampler),
                qUpperConfidenceBound(model=model, beta=4, sampler=simple_sampler),
                qLogNoisyExpectedHypervolumeImprovement(
                    model=list_gp,
                    X_baseline=test_X,
                    ref_point=torch.zeros(2, **tkwargs),
                    sampler=list_gp_sampler,
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
                qLogNoisyExpectedHypervolumeImprovement(
                    model=mixed_list,
                    X_baseline=test_X,
                    ref_point=torch.zeros(2, **tkwargs),
                    sampler=mixed_list_sampler,
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

            for acqf in acquisition_functions:
                for batch_shape in [[2], [6, 2], [5, 6, 2]]:
                    test_X = torch.rand(*batch_shape, 1, 4, **tkwargs)
                    if include_task_feature:
                        test_X = torch.cat(
                            [test_X, torch.zeros_like(test_X[..., :1])], dim=-1
                        )
                    self.assertEqual(acqf(test_X).shape, torch.Size(batch_shape))

    def test_load_samples(self):
        for task_rank, dtype, use_outcome_transform in itertools.product(
            [1, 2], [torch.float, torch.double], (False, True)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y, train_Yvar, model = self._get_data_and_model(
                task_rank=task_rank,
                use_outcome_transform=use_outcome_transform,
                **tkwargs,
            )

            d = train_X.shape[1] - 1
            mcmc_samples = self._get_mcmc_samples(
                num_samples=3,
                dim=d,
                task_rank=task_rank,
                **tkwargs,
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

            # Handle outcome transforms (if used)
            train_Y_tf, train_Yvar_tf = train_Y, train_Yvar
            if use_outcome_transform:
                train_Y_tf, train_Yvar_tf = model.outcome_transform(
                    Y=train_Y, Yvar=train_Yvar
                )

            self.assertTrue(
                torch.allclose(
                    model.pyro_model.train_Y,
                    train_Y_tf,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.pyro_model.train_Yvar,
                    train_Yvar_tf.clamp(MIN_INFERRED_NOISE_LEVEL),
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.task_covar_module.lengthscale,
                    mcmc_samples["task_lengthscale"],
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.latent_features,
                    mcmc_samples["latent_features"],
                )
            )

    def test_construct_inputs(self):
        for dtype, infer_noise in [(torch.float, False), (torch.double, True)]:
            tkwargs = {"device": self.device, "dtype": dtype}
            task_feature = 0
            datasets, (train_X, train_Y, train_Yvar) = gen_multi_task_dataset(
                yvar=None if infer_noise else 0.05, **tkwargs
            )

            model = SaasFullyBayesianMultiTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                task_feature=task_feature,
            )

            data_dict = model.construct_inputs(
                datasets,
                task_feature=task_feature,
                rank=1,
            )
            self.assertTrue(torch.equal(data_dict["train_X"], train_X))
            self.assertTrue(torch.equal(data_dict["train_Y"], train_Y))
            if train_Yvar is not None:
                self.assertAllClose(data_dict["train_Yvar"], train_Yvar)
            else:
                self.assertNotIn("train_Yvar", data_dict)
            self.assertEqual(data_dict["task_feature"], task_feature)
            self.assertEqual(data_dict["rank"], 1)
            self.assertTrue("task_covar_prior" not in data_dict)
