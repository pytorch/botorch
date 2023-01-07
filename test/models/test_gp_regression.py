#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import (
    FixedNoiseGP,
    HeteroskedasticSingleTaskGP,
    SingleTaskGP,
)
from botorch.models.transforms import Normalize, Standardize
from botorch.models.transforms.input import InputStandardize
from botorch.models.utils import add_output_dim
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.datasets import FixedNoiseDataset, SupervisedDataset
from botorch.utils.sampling import manual_seed
from botorch.utils.testing import _get_random_data, BotorchTestCase
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import (
    _GaussianLikelihoodBase,
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    HeteroskedasticNoise,
)
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.noise_model_added_loss_term import NoiseModelAddedLossTerm
from gpytorch.priors import GammaPrior


class TestSingleTaskGP(BotorchTestCase):
    def _get_model_and_data(
        self,
        batch_shape,
        m,
        outcome_transform=None,
        input_transform=None,
        extra_model_kwargs=None,
        **tkwargs,
    ):
        extra_model_kwargs = extra_model_kwargs or {}
        train_X, train_Y = _get_random_data(batch_shape=batch_shape, m=m, **tkwargs)
        model_kwargs = {
            "train_X": train_X,
            "train_Y": train_Y,
            "outcome_transform": outcome_transform,
            "input_transform": input_transform,
        }
        model = SingleTaskGP(**model_kwargs, **extra_model_kwargs)
        return model, model_kwargs

    def _get_extra_model_kwargs(self):
        return {
            "mean_module": ZeroMean(),
            "covar_module": RBFKernel(use_ard=False),
            "likelihood": GaussianLikelihood(),
        }

    def test_gp(self, double_only: bool = False):
        bounds = torch.tensor([[-1.0], [1.0]])
        for batch_shape, m, dtype, use_octf, use_intf in itertools.product(
            (torch.Size(), torch.Size([2])),
            (1, 2),
            (torch.double,) if double_only else (torch.float, torch.double),
            (False, True),
            (False, True),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=m, batch_shape=batch_shape) if use_octf else None
            intf = (
                Normalize(d=1, bounds=bounds.to(**tkwargs), transform_on_train=True)
                if use_intf
                else None
            )
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape,
                m=m,
                outcome_transform=octf,
                input_transform=intf,
                **tkwargs,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model).to(**tkwargs)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                fit_gpytorch_mll(
                    mll, optimizer_kwargs={"options": {"maxiter": 1}}, max_attempts=1
                )

            # test init
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(model.covar_module, ScaleKernel)
            matern_kernel = model.covar_module.base_kernel
            self.assertIsInstance(matern_kernel, MaternKernel)
            self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)
            if use_octf:
                self.assertIsInstance(model.outcome_transform, Standardize)
            if use_intf:
                self.assertIsInstance(model.input_transform, Normalize)
                # permute output dim
                train_X, train_Y, _ = model._transform_tensor_args(
                    X=model_kwargs["train_X"], Y=model_kwargs["train_Y"]
                )
                # check that the train inputs have been transformed and set on the model
                self.assertTrue(torch.equal(model.train_inputs[0], intf(train_X)))

            # test param sizes
            params = dict(model.named_parameters())
            for p in params:
                self.assertEqual(
                    params[p].numel(), m * torch.tensor(batch_shape).prod().item()
                )

            # test posterior
            # test non batch evaluation
            X = torch.rand(batch_shape + torch.Size([3, 1]), **tkwargs)
            expected_shape = batch_shape + torch.Size([3, m])
            posterior = model.posterior(X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, expected_shape)
            self.assertEqual(posterior.variance.shape, expected_shape)

            # test adding observation noise
            posterior_pred = model.posterior(X, observation_noise=True)
            self.assertIsInstance(posterior_pred, GPyTorchPosterior)
            self.assertEqual(posterior_pred.mean.shape, expected_shape)
            self.assertEqual(posterior_pred.variance.shape, expected_shape)
            if use_octf:
                # ensure un-transformation is applied
                tmp_tf = model.outcome_transform
                del model.outcome_transform
                pp_tf = model.posterior(X, observation_noise=True)
                model.outcome_transform = tmp_tf
                expected_var = tmp_tf.untransform_posterior(pp_tf).variance
                self.assertAllClose(posterior_pred.variance, expected_var)
            else:
                pvar = posterior_pred.variance
                pvar_exp = _get_pvar_expected(posterior, model, X, m)
                self.assertAllClose(pvar, pvar_exp, rtol=1e-4, atol=1e-5)

            # test batch evaluation
            X = torch.rand(2, *batch_shape, 3, 1, **tkwargs)
            expected_shape = torch.Size([2]) + batch_shape + torch.Size([3, m])

            posterior = model.posterior(X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, expected_shape)
            # test adding observation noise in batch mode
            posterior_pred = model.posterior(X, observation_noise=True)
            self.assertIsInstance(posterior_pred, GPyTorchPosterior)
            self.assertEqual(posterior_pred.mean.shape, expected_shape)
            if use_octf:
                # ensure un-transformation is applied
                tmp_tf = model.outcome_transform
                del model.outcome_transform
                pp_tf = model.posterior(X, observation_noise=True)
                model.outcome_transform = tmp_tf
                expected_var = tmp_tf.untransform_posterior(pp_tf).variance
                self.assertAllClose(posterior_pred.variance, expected_var)
            else:
                pvar = posterior_pred.variance
                pvar_exp = _get_pvar_expected(posterior, model, X, m)
                self.assertAllClose(pvar, pvar_exp, rtol=1e-4, atol=1e-5)

    def test_custom_init(self):
        extra_model_kwargs = self._get_extra_model_kwargs()
        for batch_shape, m, dtype in itertools.product(
            (torch.Size(), torch.Size([2])),
            (1, 2),
            (torch.float, torch.double),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape,
                m=m,
                extra_model_kwargs=extra_model_kwargs,
                **tkwargs,
            )
            self.assertEqual(model.mean_module, extra_model_kwargs["mean_module"])
            self.assertEqual(model.covar_module, extra_model_kwargs["covar_module"])
            if "likelihood" in extra_model_kwargs:
                self.assertEqual(model.likelihood, extra_model_kwargs["likelihood"])

    def test_condition_on_observations(self):
        for batch_shape, m, dtype, use_octf in itertools.product(
            (torch.Size(), torch.Size([2])),
            (1, 2),
            (torch.float, torch.double),
            (False, True),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=m, batch_shape=batch_shape) if use_octf else None
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape, m=m, outcome_transform=octf, **tkwargs
            )
            # evaluate model
            model.posterior(torch.rand(torch.Size([4, 1]), **tkwargs))
            # test condition_on_observations
            fant_shape = torch.Size([2])
            # fantasize at different input points
            X_fant, Y_fant = _get_random_data(
                batch_shape=fant_shape + batch_shape, m=m, n=3, **tkwargs
            )
            c_kwargs = (
                {"noise": torch.full_like(Y_fant, 0.01)}
                if isinstance(model, FixedNoiseGP)
                else {}
            )
            cm = model.condition_on_observations(X_fant, Y_fant, **c_kwargs)
            # fantasize at same input points (check proper broadcasting)
            c_kwargs_same_inputs = (
                {"noise": torch.full_like(Y_fant[0], 0.01)}
                if isinstance(model, FixedNoiseGP)
                else {}
            )
            cm_same_inputs = model.condition_on_observations(
                X_fant[0], Y_fant, **c_kwargs_same_inputs
            )

            test_Xs = [
                # test broadcasting single input across fantasy and model batches
                torch.rand(4, 1, **tkwargs),
                # separate input for each model batch and broadcast across
                # fantasy batches
                torch.rand(batch_shape + torch.Size([4, 1]), **tkwargs),
                # separate input for each model and fantasy batch
                torch.rand(fant_shape + batch_shape + torch.Size([4, 1]), **tkwargs),
            ]
            for test_X in test_Xs:
                posterior = cm.posterior(test_X)
                self.assertEqual(
                    posterior.mean.shape, fant_shape + batch_shape + torch.Size([4, m])
                )
                posterior_same_inputs = cm_same_inputs.posterior(test_X)
                self.assertEqual(
                    posterior_same_inputs.mean.shape,
                    fant_shape + batch_shape + torch.Size([4, m]),
                )

                # check that fantasies of batched model are correct
                if len(batch_shape) > 0 and test_X.dim() == 2:
                    state_dict_non_batch = {
                        key: (val[0] if val.numel() > 1 else val)
                        for key, val in model.state_dict().items()
                    }
                    model_kwargs_non_batch = {
                        "train_X": model_kwargs["train_X"][0],
                        "train_Y": model_kwargs["train_Y"][0],
                    }
                    if "train_Yvar" in model_kwargs:
                        model_kwargs_non_batch["train_Yvar"] = model_kwargs[
                            "train_Yvar"
                        ][0]
                    if model_kwargs["outcome_transform"] is not None:
                        model_kwargs_non_batch["outcome_transform"] = Standardize(m=m)
                    model_non_batch = type(model)(**model_kwargs_non_batch)
                    model_non_batch.load_state_dict(state_dict_non_batch)
                    model_non_batch.eval()
                    model_non_batch.likelihood.eval()
                    model_non_batch.posterior(torch.rand(torch.Size([4, 1]), **tkwargs))
                    c_kwargs = (
                        {"noise": torch.full_like(Y_fant[0, 0, :], 0.01)}
                        if isinstance(model, FixedNoiseGP)
                        else {}
                    )
                    cm_non_batch = model_non_batch.condition_on_observations(
                        X_fant[0][0], Y_fant[:, 0, :], **c_kwargs
                    )
                    non_batch_posterior = cm_non_batch.posterior(test_X)
                    self.assertTrue(
                        torch.allclose(
                            posterior_same_inputs.mean[:, 0, ...],
                            non_batch_posterior.mean,
                            atol=1e-3,
                        )
                    )
                    self.assertTrue(
                        torch.allclose(
                            posterior_same_inputs.distribution.covariance_matrix[
                                :, 0, :, :
                            ],
                            non_batch_posterior.distribution.covariance_matrix,
                            atol=1e-3,
                        )
                    )

    def test_fantasize(self):
        for batch_shape, m, dtype, use_octf in itertools.product(
            (torch.Size(), torch.Size([2])),
            (1, 2),
            (torch.float, torch.double),
            (False, True),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=m, batch_shape=batch_shape) if use_octf else None
            model, _ = self._get_model_and_data(
                batch_shape=batch_shape, m=m, outcome_transform=octf, **tkwargs
            )
            # fantasize
            X_f = torch.rand(torch.Size(batch_shape + torch.Size([4, 1])), **tkwargs)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([3]))
            fm = model.fantasize(X=X_f, sampler=sampler)
            self.assertIsInstance(fm, model.__class__)
            fm = model.fantasize(X=X_f, sampler=sampler, observation_noise=False)
            self.assertIsInstance(fm, model.__class__)

        # check that input transforms are applied to X.
        tkwargs = {"device": self.device, "dtype": torch.float}
        intf = Normalize(d=1, bounds=torch.tensor([[0], [10]], **tkwargs))
        model, _ = self._get_model_and_data(
            batch_shape=torch.Size(),
            m=1,
            input_transform=intf,
            **tkwargs,
        )
        X_f = torch.rand(4, 1, **tkwargs)
        fm = model.fantasize(
            X_f, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([3]))
        )
        self.assertTrue(
            torch.allclose(fm.train_inputs[0][:, -4:], intf(X_f).expand(3, -1, -1))
        )

    def test_subset_model(self):
        for batch_shape, dtype, use_octf in itertools.product(
            (torch.Size(), torch.Size([2])), (torch.float, torch.double), (True, False)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=2, batch_shape=batch_shape) if use_octf else None
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape, m=2, outcome_transform=octf, **tkwargs
            )
            subset_model = model.subset_output([0])
            X = torch.rand(torch.Size(batch_shape + torch.Size([3, 1])), **tkwargs)
            p = model.posterior(X)
            p_sub = subset_model.posterior(X)
            self.assertTrue(
                torch.allclose(p_sub.mean, p.mean[..., [0]], atol=1e-4, rtol=1e-4)
            )
            self.assertTrue(
                torch.allclose(
                    p_sub.variance, p.variance[..., [0]], atol=1e-4, rtol=1e-4
                )
            )
            # test subsetting each of the outputs (follows a different code branch)
            subset_all_model = model.subset_output([0, 1])
            p_sub_all = subset_all_model.posterior(X)
            self.assertAllClose(p_sub_all.mean, p.mean)
            # subsetting should still return a copy
            self.assertNotEqual(model, subset_all_model)

    def test_construct_inputs(self):
        for batch_shape, dtype in itertools.product(
            (torch.Size(), torch.Size([2])), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape, m=1, **tkwargs
            )
            X = model_kwargs["train_X"]
            Y = model_kwargs["train_Y"]
            training_data = SupervisedDataset(X, Y)
            data_dict = model.construct_inputs(training_data)
            self.assertTrue(X.equal(data_dict["train_X"]))
            self.assertTrue(Y.equal(data_dict["train_Y"]))

    def test_set_transformed_inputs(self):
        # This intended to catch https://github.com/pytorch/botorch/issues/1078.
        # More general testing of _set_transformed_inputs is done under ModelListGP.
        X = torch.rand(5, 2)
        Y = X**2
        for tf_class in [Normalize, InputStandardize]:
            intf = tf_class(d=2)
            model = SingleTaskGP(X, Y, input_transform=intf)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 2}})
            tf_X = intf(X)
            self.assertEqual(X.shape, tf_X.shape)


class TestFixedNoiseGP(TestSingleTaskGP):
    def _get_model_and_data(
        self,
        batch_shape,
        m,
        outcome_transform=None,
        input_transform=None,
        extra_model_kwargs=None,
        **tkwargs,
    ):
        extra_model_kwargs = extra_model_kwargs or {}
        train_X, train_Y = _get_random_data(batch_shape=batch_shape, m=m, **tkwargs)
        model_kwargs = {
            "train_X": train_X,
            "train_Y": train_Y,
            "train_Yvar": torch.full_like(train_Y, 0.01),
            "input_transform": input_transform,
            "outcome_transform": outcome_transform,
        }
        model = FixedNoiseGP(**model_kwargs, **extra_model_kwargs)
        return model, model_kwargs

    def _get_extra_model_kwargs(self):
        return {
            "mean_module": ZeroMean(),
            "covar_module": RBFKernel(use_ard=False),
        }

    def test_fixed_noise_likelihood(self):
        for batch_shape, m, dtype in itertools.product(
            (torch.Size(), torch.Size([2])), (1, 2), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape, m=m, **tkwargs
            )
            self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)
            self.assertTrue(
                torch.equal(
                    model.likelihood.noise.contiguous().view(-1),
                    model_kwargs["train_Yvar"].contiguous().view(-1),
                )
            )

    def test_construct_inputs(self):
        for batch_shape, dtype in itertools.product(
            (torch.Size(), torch.Size([2])), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape, m=1, **tkwargs
            )
            X = model_kwargs["train_X"]
            Y = model_kwargs["train_Y"]
            Yvar = model_kwargs["train_Yvar"]
            training_data = FixedNoiseDataset(X, Y, Yvar)
            data_dict = model.construct_inputs(training_data)
            self.assertTrue(X.equal(data_dict["train_X"]))
            self.assertTrue(Y.equal(data_dict["train_Y"]))
            self.assertTrue(Yvar.equal(data_dict["train_Yvar"]))


class TestHeteroskedasticSingleTaskGP(TestSingleTaskGP):
    def _get_model_and_data(
        self, batch_shape, m, outcome_transform=None, input_transform=None, **tkwargs
    ):
        with manual_seed(0):
            train_X, train_Y = _get_random_data(batch_shape=batch_shape, m=m, **tkwargs)
            train_Yvar = (0.1 + 0.1 * torch.rand_like(train_Y)) ** 2
        model_kwargs = {
            "train_X": train_X,
            "train_Y": train_Y,
            "train_Yvar": train_Yvar,
            "input_transform": input_transform,
            "outcome_transform": outcome_transform,
        }
        model = HeteroskedasticSingleTaskGP(**model_kwargs)
        return model, model_kwargs

    def test_custom_init(self) -> None:
        """
        This test exists because `TestHeteroskedasticSingleTaskGP` inherits from
        `TestSingleTaskGP`, which has a `test_custom_init` method that isn't relevant
        for `TestHeteroskedasticSingleTaskGP`.
        """

    def test_gp(self):
        super().test_gp(double_only=True)

    def test_fantasize(self) -> None:
        """
        This test exists because `TestHeteroskedasticSingleTaskGP` inherits from
        `TestSingleTaskGP`, which has a `fantasize` method that isn't relevant
        for `TestHeteroskedasticSingleTaskGP`.
        """

    def test_heteroskedastic_likelihood(self):
        for batch_shape, m, dtype in itertools.product(
            (torch.Size(), torch.Size([2])), (1, 2), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, _ = self._get_model_and_data(batch_shape=batch_shape, m=m, **tkwargs)
            self.assertIsInstance(model.likelihood, _GaussianLikelihoodBase)
            self.assertFalse(isinstance(model.likelihood, GaussianLikelihood))
            self.assertIsInstance(model.likelihood.noise_covar, HeteroskedasticNoise)
            self.assertIsInstance(
                model.likelihood.noise_covar.noise_model, SingleTaskGP
            )
            self.assertIsInstance(
                model._added_loss_terms["noise_added_loss"], NoiseModelAddedLossTerm
            )

    def test_condition_on_observations(self):
        with self.assertRaises(NotImplementedError):
            super().test_condition_on_observations()

    def test_subset_model(self):
        with self.assertRaises(NotImplementedError):
            super().test_subset_model()


def _get_pvar_expected(posterior, model, X, m):
    X = model.transform_inputs(X)
    lh_kwargs = {}
    if isinstance(model.likelihood, FixedNoiseGaussianLikelihood):
        lh_kwargs["noise"] = model.likelihood.noise.mean().expand(X.shape[:-1])
    if m == 1:
        return model.likelihood(
            posterior.distribution, X, **lh_kwargs
        ).variance.unsqueeze(-1)
    X_, odi = add_output_dim(X=X, original_batch_shape=model._input_batch_shape)
    pvar_exp = model.likelihood(model(X_), X_, **lh_kwargs).variance
    return torch.stack([pvar_exp.select(dim=odi, index=i) for i in range(m)], dim=-1)
