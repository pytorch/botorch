#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools

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
    qNoisyExpectedHypervolumeImprovement,
    qExpectedHypervolumeImprovement,
)
from botorch.models import ModelListGP
from botorch.models.fully_bayesian import (
    SaasFullyBayesianSingleTaskGP,
    SaasPyroModel,
    MIN_INFERRED_NOISE_LEVEL,
    PyroModel,
)
from botorch.models.transforms import Normalize, Standardize
from botorch.posteriors import FullyBayesianPosterior
from botorch.sampling.samplers import IIDNormalSampler
from botorch.utils.containers import TrainingData
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.means import ConstantMean


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
                None if infer_noise else 0.1 * torch.arange(10, **tkwargs).unsqueeze(-1)
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
            "mean": torch.randn(num_samples, 1, **tkwargs),
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
        sampler = IIDNormalSampler(num_samples=2)
        with self.assertRaisesRegex(
            NotImplementedError, "Fantasize is not implemented!"
        ):
            model.fantasize(X=torch.rand(5, 4, **tkwargs), sampler=sampler)
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
            self.assertTrue(torch.allclose(train_X, model.pyro_model.train_X))
            self.assertTrue(torch.allclose(train_Y, model.pyro_model.train_Y))
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
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertEqual(model.mean_module.constant.shape, torch.Size([3, 1]))
            self.assertIsInstance(model.covar_module, ScaleKernel)
            self.assertEqual(model.covar_module.outputscale.shape, torch.Size([3]))
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
                self.assertEqual(model.likelihood.noise.shape, torch.Size([n, 1]))
                self.assertTrue(
                    torch.allclose(
                        train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL),
                        model.likelihood.noise,
                    )
                )

            # Predict on some test points
            for batch_shape, marginalize in itertools.product(
                [[5], [6, 5, 2]], [True, False]
            ):
                test_X = torch.rand(*batch_shape, d, **tkwargs)
                posterior = model.posterior(
                    test_X, marginalize_over_mcmc_samples=marginalize
                )
                self.assertIsInstance(posterior, FullyBayesianPosterior)
                expected_shape = (
                    batch_shape + [1]
                    if marginalize
                    else batch_shape[:-1] + [3] + batch_shape[-1:] + [1]
                )
                mean, var = posterior.mean, posterior.variance
                self.assertEqual(mean.shape, torch.Size(expected_shape))
                self.assertEqual(var.shape, torch.Size(expected_shape))

            # Check properties
            median_lengthscale = model.median_lengthscale
            self.assertEqual(median_lengthscale.shape, torch.Size([4]))
            self.assertEqual(model.num_mcmc_samples, 3)

            # Make sure the model shapes are set correctly
            self.assertEqual(model.pyro_model.train_X.shape, torch.Size([n, d]))
            self.assertTrue(torch.allclose(model.pyro_model.train_X, train_X))
            model.train()  # Put the model in train mode
            self.assertTrue(torch.allclose(train_X, model.pyro_model.train_X))
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
                    train_Yvar=train_Yvar / sigma ** 2
                    if train_Yvar is not None
                    else train_Yvar,
                )
                fit_fully_bayesian_model_nuts(
                    gp1, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
                )
                posterior1 = gp1.posterior(
                    (test_X - lb) / (ub - lb), marginalize_over_mcmc_samples=True
                )
                pred_mean1 = mu + sigma * posterior1.mean
                pred_var1 = (sigma ** 2) * posterior1.variance

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
                posterior2 = gp2.posterior(test_X, marginalize_over_mcmc_samples=True)
                pred_mean2, pred_var2 = posterior2.mean, posterior2.variance

            self.assertTrue(torch.allclose(pred_mean1, pred_mean2))
            self.assertTrue(torch.allclose(pred_var1, pred_var2))

    def test_acquisition_functions(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X, train_Y, train_Yvar, model = self._get_data_and_model(
            infer_noise=True, **tkwargs
        )
        fit_fully_bayesian_model_nuts(
            model, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
        )
        sampler = IIDNormalSampler(num_samples=2)
        acquisition_functions = [
            ExpectedImprovement(model=model, best_f=train_Y.max()),
            ProbabilityOfImprovement(model=model, best_f=train_Y.max()),
            PosteriorMean(model=model),
            UpperConfidenceBound(model=model, beta=4),
            qExpectedImprovement(model=model, best_f=train_Y.max(), sampler=sampler),
            qNoisyExpectedImprovement(model=model, X_baseline=train_X, sampler=sampler),
            qProbabilityOfImprovement(
                model=model, best_f=train_Y.max(), sampler=sampler
            ),
            qSimpleRegret(model=model, sampler=sampler),
            qUpperConfidenceBound(model=model, beta=4, sampler=sampler),
            qNoisyExpectedHypervolumeImprovement(
                model=ModelListGP(model, model),
                X_baseline=train_X,
                ref_point=torch.zeros(2, **tkwargs),
                sampler=sampler,
            ),
            qExpectedHypervolumeImprovement(
                model=ModelListGP(model, model),
                ref_point=torch.zeros(2, **tkwargs),
                sampler=sampler,
                partitioning=NondominatedPartitioning(
                    ref_point=torch.zeros(2, **tkwargs), Y=train_Y.repeat([1, 2])
                ),
            ),
        ]

        for acqf in acquisition_functions:
            for batch_shape in [[5], [6, 5, 2]]:
                test_X = torch.rand(*batch_shape, 1, 4, **tkwargs)
                self.assertEqual(acqf(test_X).shape, torch.Size(batch_shape))

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
                    model.mean_module.constant.data,
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
                        train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL),
                    )
                )

    def test_construct_inputs(self):
        for infer_noise, dtype in itertools.product(
            (True, False), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y, train_Yvar, model = self._get_data_and_model(
                infer_noise=infer_noise, **tkwargs
            )
            training_data = TrainingData.from_block_design(
                X=train_X,
                Y=train_Y,
                Yvar=train_Yvar,
            )
            data_dict = model.construct_inputs(training_data)
            if infer_noise:
                self.assertTrue("train_Yvar" not in data_dict)
            else:
                self.assertTrue(torch.equal(data_dict["train_Yvar"], train_Yvar))
            self.assertTrue(torch.equal(data_dict["train_X"], train_X))
            self.assertTrue(torch.equal(data_dict["train_Y"], train_Y))

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
            self.assertIsInstance(model.pyro_model, CustomPyroModel)
            self.assertTrue(torch.allclose(model.pyro_model.train_X, train_X))
            self.assertTrue(torch.allclose(model.pyro_model.train_Y, train_Y))
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
                        train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL) / (sigma ** 2),
                        atol=1e-4,
                    )
                )
