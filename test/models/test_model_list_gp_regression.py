#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import ChainedOutcomeTransform, Log, Standardize
from botorch.posteriors import GPyTorchPosterior, PosteriorList, TransformedPosterior
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.testing import _get_random_data, BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import LikelihoodList
from gpytorch.means import ConstantMean
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior


def _get_model(
    fixed_noise=False, outcome_transform: str = "None", use_intf=False, **tkwargs
) -> ModelListGP:
    train_x1, train_y1 = _get_random_data(
        batch_shape=torch.Size(), m=1, n=10, **tkwargs
    )
    train_y1 = torch.exp(train_y1)
    train_x2, train_y2 = _get_random_data(
        batch_shape=torch.Size(), m=1, n=11, **tkwargs
    )
    if outcome_transform == "Standardize":
        octfs = [Standardize(m=1), Standardize(m=1)]
    elif outcome_transform == "Log":
        octfs = [Log(), Standardize(m=1)]
    elif outcome_transform == "Chained":
        octfs = [
            ChainedOutcomeTransform(
                chained=ChainedOutcomeTransform(log=Log(), standardize=Standardize(m=1))
            ),
            Standardize(m=1),
        ]
    elif outcome_transform == "None":
        octfs = [None, None]
    else:
        raise KeyError(  # pragma: no cover
            "outcome_transform must be one of 'Standardize', 'Log', 'Chained', or "
            "'None'."
        )
    intfs = [Normalize(d=1), Normalize(d=1)] if use_intf else [None, None]
    if fixed_noise:
        train_y1_var = 0.1 + 0.1 * torch.rand_like(train_y1, **tkwargs)
        train_y2_var = 0.1 + 0.1 * torch.rand_like(train_y2, **tkwargs)
        model1 = FixedNoiseGP(
            train_X=train_x1,
            train_Y=train_y1,
            train_Yvar=train_y1_var,
            outcome_transform=octfs[0],
            input_transform=intfs[0],
        )
        model2 = FixedNoiseGP(
            train_X=train_x2,
            train_Y=train_y2,
            train_Yvar=train_y2_var,
            outcome_transform=octfs[1],
            input_transform=intfs[1],
        )
    else:
        model1 = SingleTaskGP(
            train_X=train_x1,
            train_Y=train_y1,
            outcome_transform=octfs[0],
            input_transform=intfs[0],
        )
        model2 = SingleTaskGP(
            train_X=train_x2,
            train_Y=train_y2,
            outcome_transform=octfs[1],
            input_transform=intfs[1],
        )
    model = ModelListGP(model1, model2)
    return model.to(**tkwargs)


class TestModelListGP(BotorchTestCase):
    def _base_test_ModelListGP(
        self, fixed_noise: bool, dtype, outcome_transform: str
    ) -> ModelListGP:
        tkwargs = {"device": self.device, "dtype": dtype}
        model = _get_model(
            fixed_noise=fixed_noise, outcome_transform=outcome_transform, **tkwargs
        )
        self.assertIsInstance(model, ModelListGP)
        self.assertIsInstance(model.likelihood, LikelihoodList)
        for m in model.models:
            self.assertIsInstance(m.mean_module, ConstantMean)
            self.assertIsInstance(m.covar_module, ScaleKernel)
            matern_kernel = m.covar_module.base_kernel
            self.assertIsInstance(matern_kernel, MaternKernel)
            self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)
            if outcome_transform != "None":
                self.assertIsInstance(
                    m.outcome_transform, (Log, Standardize, ChainedOutcomeTransform)
                )
            else:
                assert not hasattr(m, "outcome_transform")

        # test constructing likelihood wrapper
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        for mll_ in mll.mlls:
            self.assertIsInstance(mll_, ExactMarginalLogLikelihood)

        # test model fitting (sequential)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            mll = fit_gpytorch_mll(
                mll, optimizer_kwargs={"options": {"maxiter": 1}}, max_attempts=1
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            # test model fitting (joint)
            mll = fit_gpytorch_mll(
                mll,
                optimizer_kwargs={"options": {"maxiter": 1}},
                max_attempts=1,
                sequential=False,
            )

        # test subset outputs
        subset_model = model.subset_output([1])
        self.assertIsInstance(subset_model, ModelListGP)
        self.assertEqual(len(subset_model.models), 1)
        sd_subset = subset_model.models[0].state_dict()
        sd = model.models[1].state_dict()
        self.assertTrue(set(sd_subset.keys()) == set(sd.keys()))
        self.assertTrue(all(torch.equal(v, sd[k]) for k, v in sd_subset.items()))

        # test posterior
        test_x = torch.tensor([[0.25], [0.75]], **tkwargs)
        posterior = model.posterior(test_x)
        gpytorch_posterior_expected = outcome_transform in ("None", "Standardize")
        expected_type = (
            GPyTorchPosterior if gpytorch_posterior_expected else PosteriorList
        )
        self.assertIsInstance(posterior, expected_type)
        submodel = model.models[0]
        p0 = submodel.posterior(test_x)
        self.assertAllClose(posterior.mean[:, [0]], p0.mean)
        self.assertAllClose(posterior.variance[:, [0]], p0.variance)

        if gpytorch_posterior_expected:
            self.assertIsInstance(posterior.distribution, MultitaskMultivariateNormal)
        if outcome_transform != "None":
            # ensure un-transformation is applied
            submodel = model.models[0]
            p0 = submodel.posterior(test_x)
            tmp_tf = submodel.outcome_transform
            del submodel.outcome_transform
            p0_tf = submodel.posterior(test_x)
            submodel.outcome_transform = tmp_tf
            expected_var = tmp_tf.untransform_posterior(p0_tf).variance
            self.assertAllClose(p0.variance, expected_var)

        # test output_indices
        posterior = model.posterior(test_x, output_indices=[0], observation_noise=True)
        self.assertIsInstance(posterior, expected_type)
        if gpytorch_posterior_expected:
            self.assertIsInstance(posterior.distribution, MultivariateNormal)

        # test condition_on_observations
        f_x = [torch.rand(2, 1, **tkwargs) for _ in range(2)]
        f_y = torch.rand(2, 2, **tkwargs)
        if fixed_noise:
            noise = 0.1 + 0.1 * torch.rand_like(f_y)
            cond_kwargs = {"noise": noise}
        else:
            cond_kwargs = {}
        cm = model.condition_on_observations(f_x, f_y, **cond_kwargs)
        self.assertIsInstance(cm, ModelListGP)

        # test condition_on_observations batched
        f_x = [torch.rand(3, 2, 1, **tkwargs) for _ in range(2)]
        f_y = torch.rand(3, 2, 2, **tkwargs)
        cm = model.condition_on_observations(f_x, f_y, **cond_kwargs)
        self.assertIsInstance(cm, ModelListGP)

        # test condition_on_observations batched (fast fantasies)
        f_x = [torch.rand(2, 1, **tkwargs) for _ in range(2)]
        f_y = torch.rand(3, 2, 2, **tkwargs)
        cm = model.condition_on_observations(f_x, f_y, **cond_kwargs)
        self.assertIsInstance(cm, ModelListGP)

        # test condition_on_observations (incorrect input shape error)
        with self.assertRaises(BotorchTensorDimensionError):
            model.condition_on_observations(
                f_x, torch.rand(3, 2, 3, **tkwargs), **cond_kwargs
            )

        # test X having wrong size
        with self.assertRaises(AssertionError):
            model.condition_on_observations(f_x[:1], f_y)

        # test posterior transform
        X = torch.rand(3, 1, **tkwargs)
        weights = torch.tensor([1, 2], **tkwargs)
        post_tf = ScalarizedPosteriorTransform(weights=weights)
        if gpytorch_posterior_expected:
            posterior_tf = model.posterior(X, posterior_transform=post_tf)
            self.assertTrue(
                torch.allclose(
                    posterior_tf.mean,
                    model.posterior(X).mean @ weights.unsqueeze(-1),
                )
            )

        return model

    def test_ModelListGP(self) -> None:
        for dtype, outcome_transform in itertools.product(
            (torch.float, torch.double), ("None", "Standardize", "Log", "Chained")
        ):

            model = self._base_test_ModelListGP(
                fixed_noise=False, dtype=dtype, outcome_transform=outcome_transform
            )
            tkwargs = {"device": self.device, "dtype": dtype}

            # test observation_noise
            test_x = torch.tensor([[0.25], [0.75]], **tkwargs)
            posterior = model.posterior(test_x, observation_noise=True)

            gpytorch_posterior_expected = outcome_transform in ("None", "Standardize")
            expected_type = (
                GPyTorchPosterior if gpytorch_posterior_expected else PosteriorList
            )
            self.assertIsInstance(posterior, expected_type)
            if gpytorch_posterior_expected:
                self.assertIsInstance(
                    posterior.distribution, MultitaskMultivariateNormal
                )
            else:
                self.assertIsInstance(posterior.posteriors[0], TransformedPosterior)

    def test_ModelListGP_fixed_noise(self) -> None:

        for dtype, outcome_transform in itertools.product(
            (torch.float, torch.double), ("None", "Standardize")
        ):
            model = self._base_test_ModelListGP(
                fixed_noise=True, dtype=dtype, outcome_transform=outcome_transform
            )
            tkwargs = {"device": self.device, "dtype": dtype}
            f_x = [torch.rand(2, 1, **tkwargs) for _ in range(2)]
            f_y = torch.rand(2, 2, **tkwargs)

            # test condition_on_observations (incorrect noise shape error)
            with self.assertRaises(BotorchTensorDimensionError):
                model.condition_on_observations(
                    f_x, f_y, noise=torch.rand(2, 3, **tkwargs)
                )

    def test_ModelListGP_single(self):
        tkwargs = {"device": self.device, "dtype": torch.float}
        train_x1, train_y1 = _get_random_data(
            batch_shape=torch.Size(), m=1, n=10, **tkwargs
        )
        model1 = SingleTaskGP(train_X=train_x1, train_Y=train_y1)
        model = ModelListGP(model1)
        model.to(**tkwargs)
        test_x = torch.tensor([[0.25], [0.75]], **tkwargs)
        posterior = model.posterior(test_x)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertIsInstance(posterior.distribution, MultivariateNormal)

    def test_transform_revert_train_inputs(self):
        tkwargs = {"device": self.device, "dtype": torch.float}
        model_list = _get_model(use_intf=True, **tkwargs)
        org_inputs = [m.train_inputs[0] for m in model_list.models]
        model_list.eval()
        for i, m in enumerate(model_list.models):
            self.assertTrue(
                torch.allclose(
                    m.train_inputs[0],
                    m.input_transform.preprocess_transform(org_inputs[i]),
                )
            )
            self.assertTrue(m._has_transformed_inputs)
            self.assertTrue(torch.equal(m._original_train_inputs, org_inputs[i]))
        model_list.train(mode=True)
        for i, m in enumerate(model_list.models):
            self.assertTrue(torch.equal(m.train_inputs[0], org_inputs[i]))
            self.assertFalse(m._has_transformed_inputs)
        model_list.train(mode=False)
        for i, m in enumerate(model_list.models):
            self.assertTrue(
                torch.allclose(
                    m.train_inputs[0],
                    m.input_transform.preprocess_transform(org_inputs[i]),
                )
            )
            self.assertTrue(m._has_transformed_inputs)
            self.assertTrue(torch.equal(m._original_train_inputs, org_inputs[i]))

    def test_fantasize(self):
        m1 = SingleTaskGP(torch.rand(5, 2), torch.rand(5, 1)).eval()
        m2 = SingleTaskGP(torch.rand(5, 2), torch.rand(5, 1)).eval()
        modellist = ModelListGP(m1, m2)
        fm = modellist.fantasize(torch.rand(3, 2), sampler=IIDNormalSampler(2))
        self.assertIsInstance(fm, ModelListGP)
        for i in range(2):
            fm_i = fm.models[i]
            self.assertIsInstance(fm_i, SingleTaskGP)
            self.assertEqual(fm_i.train_inputs[0].shape, torch.Size([2, 8, 2]))
            self.assertEqual(fm_i.train_targets.shape, torch.Size([2, 8]))

    def test_fantasize_with_outcome_transform(self) -> None:
        """
        Check that fantasized posteriors from a `ModelListGP` with transforms
        relate in a predictable way to posteriors from a `ModelListGP` when the
        outputs have been manually transformed.

        We are essentially fitting "Y = 10 * X" with Y standardized.
        - In the original space, we should predict a mean of ~5 at 0.5
        - In the standardized space, we should predict ~0.
        - If we untransform the result in the standardized space, we should recover
        the prediction of ~5 we would have gotten in the original space.
        """

        for dtype in [torch.float, torch.double]:
            with self.subTest(dtype=dtype):
                tkwargs = {"device": self.device, "dtype": dtype}
                X = torch.linspace(0, 1, 20, **tkwargs)[:, None]
                Y = 10 * torch.linspace(0, 1, 20, **tkwargs)[:, None]
                target_x = torch.tensor([[0.5]], **tkwargs)

                model_with_transform = ModelListGP(
                    SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
                )
                y_standardized, _ = Standardize(m=1).forward(Y)
                model_manually_transformed = ModelListGP(
                    SingleTaskGP(X, y_standardized)
                )

                def _get_fant_mean(model: ModelListGP) -> float:
                    fant = model.fantasize(
                        target_x, sampler=IIDNormalSampler(10, seed=0)
                    )
                    return fant.posterior(target_x).mean.mean().item()

                outcome_transform = model_with_transform.models[0].outcome_transform
                # ~0
                fant_mean_with_manual_transform = _get_fant_mean(
                    model_manually_transformed
                )
                # Inexact since this is an MC test and we don't want it flaky
                self.assertAlmostEqual(fant_mean_with_manual_transform, 0.0, delta=0.1)
                manually_rescaled_mean, _ = outcome_transform.untransform(
                    fant_mean_with_manual_transform
                )
                fant_mean_with_native_transform = _get_fant_mean(model_with_transform)
                # Inexact since this is an MC test and we don't want it flaky
                self.assertAlmostEqual(fant_mean_with_native_transform, 5.0, delta=0.5)

                # tighter tolerance here since the models should use the same samples
                self.assertAlmostEqual(
                    manually_rescaled_mean.item(),
                    fant_mean_with_native_transform,
                    delta=1e-6,
                )

    def test_fantasize_with_outcome_transform_fixed_noise(self) -> None:
        """
        Test that 'fantasize' on average recovers the true mean fn.

        Loose tolerance to protect against flakiness. The true mean function is
        100 at x=0. If transforms are not properly applied, we'll get answers
        on the order of ~1. Answers between 99 and 101 are acceptable.
        """
        n_fants = 20
        y_at_low_x = 100.0
        y_at_high_x = -40.0

        for dtype in [torch.float, torch.double]:
            with self.subTest(dtype=dtype):
                tkwargs = {"device": self.device, "dtype": dtype}
                X = torch.tensor([[0.0], [1.0]], **tkwargs)
                Y = torch.tensor([[y_at_low_x], [y_at_high_x]], **tkwargs)
                yvar = torch.full_like(Y, 1e-4)
                model = ModelListGP(
                    FixedNoiseGP(X, Y, yvar, outcome_transform=Standardize(m=1))
                )

                model.posterior(torch.zeros((1, 1), **tkwargs))

                fant = model.fantasize(
                    X, sampler=IIDNormalSampler(n_fants, seed=0), noise=yvar
                )

                fant_mean = fant.posterior(X).mean.mean(0).flatten().tolist()
                self.assertAlmostEqual(fant_mean[0], y_at_low_x, delta=1)
                # delta=1 is a 1% error (since y_at_low_x = 100)
                self.assertAlmostEqual(fant_mean[1], y_at_high_x, delta=1)
