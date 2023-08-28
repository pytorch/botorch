#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings
from typing import Optional

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import ChainedOutcomeTransform, Log, Standardize
from botorch.posteriors import GPyTorchPosterior, PosteriorList, TransformedPosterior
from botorch.sampling.base import MCSampler
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.testing import _get_random_data, BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import LikelihoodList
from gpytorch.means import ConstantMean
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor


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
        self.assertEqual(model.num_outputs, 2)
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

            # Test tensor valued observation noise.
            observation_noise = torch.rand(2, 2, **tkwargs)
            with torch.no_grad():
                noise_free_variance = model.posterior(test_x).variance
                noisy_variance = model.posterior(
                    test_x, observation_noise=observation_noise
                ).variance
            self.assertEqual(noise_free_variance.shape, noisy_variance.shape)
            if outcome_transform == "None":
                self.assertAllClose(
                    noise_free_variance + observation_noise, noisy_variance
                )

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

    def test_ModelListGP_multi_task(self):
        tkwargs = {"device": self.device, "dtype": torch.float}
        train_x_raw, train_y = _get_random_data(
            batch_shape=torch.Size(), m=1, n=10, **tkwargs
        )
        task_idx = torch.cat(
            [torch.ones(5, 1, **tkwargs), torch.zeros(5, 1, **tkwargs)], dim=0
        )
        train_x = torch.cat([train_x_raw, task_idx], dim=-1)
        model = MultiTaskGP(
            train_X=train_x,
            train_Y=train_y,
            task_feature=-1,
            output_tasks=[0],
        )
        # Wrap a single single-output MTGP.
        model_list_gp = ModelListGP(model)
        self.assertEqual(model_list_gp.num_outputs, 1)
        with torch.no_grad():
            model_mean = model.posterior(train_x_raw).mean
            model_list_gp_mean = model_list_gp.posterior(train_x_raw).mean
        self.assertAllClose(model_mean, model_list_gp_mean)
        # Wrap two single-output MTGPs.
        model_list_gp = ModelListGP(model, model)
        self.assertEqual(model_list_gp.num_outputs, 2)
        with torch.no_grad():
            model_list_gp_mean = model_list_gp.posterior(train_x_raw).mean
        expected_mean = torch.cat([model_mean, model_mean], dim=-1)
        self.assertAllClose(expected_mean, model_list_gp_mean)
        # Wrap a multi-output MTGP.
        model2 = MultiTaskGP(
            train_X=train_x,
            train_Y=train_y,
            task_feature=-1,
        )
        model_list_gp = ModelListGP(model2)
        self.assertEqual(model_list_gp.num_outputs, 2)
        with torch.no_grad():
            model2_mean = model2.posterior(train_x_raw).mean
            model_list_gp_mean = model_list_gp.posterior(train_x_raw).mean
        self.assertAllClose(model2_mean, model_list_gp_mean)
        # Mix of multi-output and single-output MTGPs.
        model_list_gp = ModelListGP(model, model2)
        self.assertEqual(model_list_gp.num_outputs, 3)
        with torch.no_grad():
            model_list_gp_mean = model_list_gp.posterior(train_x_raw).mean
        expected_mean = torch.cat([model_mean, model2_mean], dim=-1)
        self.assertAllClose(expected_mean, model_list_gp_mean)

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
        fm = modellist.fantasize(
            torch.rand(3, 2), sampler=IIDNormalSampler(sample_shape=torch.Size([2]))
        )
        self.assertIsInstance(fm, ModelListGP)
        for i in range(2):
            fm_i = fm.models[i]
            self.assertIsInstance(fm_i, SingleTaskGP)
            self.assertEqual(fm_i.train_inputs[0].shape, torch.Size([2, 8, 2]))
            self.assertEqual(fm_i.train_targets.shape, torch.Size([2, 8]))

        # test decoupled
        sampler1 = IIDNormalSampler(sample_shape=torch.Size([2]))
        sampler2 = IIDNormalSampler(sample_shape=torch.Size([2]))
        eval_mask = torch.tensor(
            [[1, 0], [0, 1], [1, 0]],
            dtype=torch.bool,
        )
        fm = modellist.fantasize(
            torch.rand(3, 2),
            sampler=ListSampler(sampler1, sampler2),
            evaluation_mask=eval_mask,
        )
        self.assertIsInstance(fm, ModelListGP)
        for i in range(2):
            fm_i = fm.models[i]
            self.assertIsInstance(fm_i, SingleTaskGP)
            num_points = 7 - i
            self.assertEqual(fm_i.train_inputs[0].shape, torch.Size([2, num_points, 2]))
            self.assertEqual(fm_i.train_targets.shape, torch.Size([2, num_points]))

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
                Y1 = 10 * torch.linspace(0, 1, 20, **tkwargs)[:, None]
                Y2 = 2 * Y1
                Y = torch.cat([Y1, Y2], dim=-1)
                target_x = torch.tensor([[0.5]], **tkwargs)

                model_with_transform = ModelListGP(
                    SingleTaskGP(X, Y1, outcome_transform=Standardize(m=1)),
                    SingleTaskGP(X, Y2, outcome_transform=Standardize(m=1)),
                )
                outcome_transform = Standardize(m=2)
                y_standardized, _ = outcome_transform(Y)
                outcome_transform.eval()
                model_manually_transformed = ModelListGP(
                    SingleTaskGP(X, y_standardized[:, :1]),
                    SingleTaskGP(X, y_standardized[:, 1:]),
                )

                def _get_fant_mean(
                    model: ModelListGP,
                    sampler: MCSampler,
                    eval_mask: Optional[Tensor] = None,
                ) -> float:
                    fant = model.fantasize(
                        target_x,
                        sampler=sampler,
                        evaluation_mask=eval_mask,
                    )
                    return fant.posterior(target_x).mean.mean(dim=(-2, -3))

                # ~0
                sampler = IIDNormalSampler(sample_shape=torch.Size([10]), seed=0)
                fant_mean_with_manual_transform = _get_fant_mean(
                    model_manually_transformed, sampler=sampler
                )
                # Inexact since this is an MC test and we don't want it flaky
                self.assertLessEqual(
                    (fant_mean_with_manual_transform - 0.0).abs().max().item(), 0.1
                )
                manually_rescaled_mean = outcome_transform.untransform(
                    fant_mean_with_manual_transform
                )[0].view(-1)
                fant_mean_with_native_transform = _get_fant_mean(
                    model_with_transform, sampler=sampler
                )
                # Inexact since this is an MC test and we don't want it flaky
                self.assertLessEqual(
                    (
                        fant_mean_with_native_transform
                        - torch.tensor([5.0, 10.0], **tkwargs)
                    )
                    .abs()
                    .max()
                    .item(),
                    0.5,
                )

                # tighter tolerance here since the models should use the same samples
                self.assertAllClose(
                    manually_rescaled_mean,
                    fant_mean_with_native_transform,
                )
                # test decoupled
                sampler = ListSampler(
                    IIDNormalSampler(sample_shape=torch.Size([10]), seed=0),
                    IIDNormalSampler(sample_shape=torch.Size([10]), seed=0),
                )
                fant_mean_with_manual_transform = _get_fant_mean(
                    model_manually_transformed,
                    sampler=sampler,
                    eval_mask=torch.tensor(
                        [[0, 1]], dtype=torch.bool, device=tkwargs["device"]
                    ),
                )
                # Inexact since this is an MC test and we don't want it flaky
                self.assertLessEqual(
                    (fant_mean_with_manual_transform - 0.0).abs().max().item(), 0.1
                )
                manually_rescaled_mean = outcome_transform.untransform(
                    fant_mean_with_manual_transform
                )[0].view(-1)
                fant_mean_with_native_transform = _get_fant_mean(
                    model_with_transform,
                    sampler=sampler,
                    eval_mask=torch.tensor(
                        [[0, 1]], dtype=torch.bool, device=tkwargs["device"]
                    ),
                )
                # Inexact since this is an MC test and we don't want it flaky
                self.assertLessEqual(
                    (
                        fant_mean_with_native_transform
                        - torch.tensor([5.0, 10.0], **tkwargs)
                    )
                    .abs()
                    .max()
                    .item(),
                    0.5,
                )
                # tighter tolerance here since the models should use the same samples
                self.assertAllClose(
                    manually_rescaled_mean,
                    fant_mean_with_native_transform,
                )

    def test_fantasize_with_outcome_transform_fixed_noise(self) -> None:
        """
        Test that 'fantasize' on average recovers the true mean fn.

        Loose tolerance to protect against flakiness. The true mean function is
        100 at x=0. If transforms are not properly applied, we'll get answers
        on the order of ~1. Answers between 99 and 101 are acceptable.
        """
        n_fants = torch.Size([20])
        y_at_low_x = 100.0
        y_at_high_x = -40.0

        for dtype in [torch.float, torch.double]:
            with self.subTest(dtype=dtype):
                tkwargs = {"device": self.device, "dtype": dtype}
                X = torch.tensor([[0.0], [1.0]], **tkwargs)
                Y = torch.tensor([[y_at_low_x], [y_at_high_x]], **tkwargs)
                Y2 = 2 * Y
                yvar = torch.full_like(Y, 1e-4)
                yvar2 = 2 * yvar
                model = ModelListGP(
                    FixedNoiseGP(X, Y, yvar, outcome_transform=Standardize(m=1)),
                    FixedNoiseGP(X, Y2, yvar2, outcome_transform=Standardize(m=1)),
                )
                # test exceptions
                eval_mask = torch.zeros(
                    3, 2, 2, dtype=torch.bool, device=tkwargs["device"]
                )
                msg = (
                    f"Expected evaluation_mask of shape `{X.shape[0]} x "
                    f"{model.num_outputs}`, but got `"
                    f"{' x '.join(str(i) for i in eval_mask.shape)}`."
                )
                with self.assertRaisesRegex(BotorchTensorDimensionError, msg):

                    model.fantasize(
                        X,
                        evaluation_mask=eval_mask,
                        sampler=ListSampler(
                            IIDNormalSampler(n_fants, seed=0),
                            IIDNormalSampler(n_fants, seed=0),
                        ),
                    )
                msg = "Decoupled fantasization requires a list of samplers."
                with self.assertRaisesRegex(ValueError, msg):
                    model.fantasize(
                        X,
                        evaluation_mask=eval_mask[0],
                        sampler=IIDNormalSampler(n_fants, seed=0),
                    )
                model.posterior(torch.zeros((1, 1), **tkwargs))
                for decoupled in (False, True):
                    if decoupled:
                        kwargs = {
                            "sampler": ListSampler(
                                IIDNormalSampler(n_fants, seed=0),
                                IIDNormalSampler(n_fants, seed=0),
                            ),
                            "evaluation_mask": torch.tensor(
                                [[0, 1], [1, 0]],
                                dtype=torch.bool,
                                device=tkwargs["device"],
                            ),
                        }
                    else:
                        kwargs = {
                            "sampler": IIDNormalSampler(n_fants, seed=0),
                        }
                    fant = model.fantasize(X, **kwargs)

                    fant_mean = fant.posterior(X).mean.mean(0)
                    self.assertAlmostEqual(fant_mean[0, 0].item(), y_at_low_x, delta=1)
                    self.assertAlmostEqual(
                        fant_mean[0, 1].item(), 2 * y_at_low_x, delta=1
                    )
                    # delta=1 is a 1% error (since y_at_low_x = 100)
                    self.assertAlmostEqual(fant_mean[1, 0].item(), y_at_high_x, delta=1)
                    self.assertAlmostEqual(
                        fant_mean[1, 1].item(), 2 * y_at_high_x, delta=1
                    )
                    for i, fm_i in enumerate(fant.models):
                        n_points = 3 if decoupled else 4
                        self.assertEqual(
                            fm_i.train_inputs[0].shape, torch.Size([20, n_points, 1])
                        )
                        self.assertEqual(
                            fm_i.train_targets.shape, torch.Size([20, n_points])
                        )
                        if decoupled:
                            self.assertTrue(
                                torch.equal(fm_i.train_inputs[0][0][-1], X[1 - i])
                            )
