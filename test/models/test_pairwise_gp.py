#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch import fit_gpytorch_model
from botorch.exceptions.warnings import OptimizationWarning
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling.pairwise_samplers import PairwiseSobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import RBFKernel
from gpytorch.kernels.linear_kernel import LinearKernel
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior


class TestPairwiseGP(BotorchTestCase):
    def _make_rand_mini_data(self, batch_shape, X_dim=2, **tkwargs):
        train_X = torch.rand(*batch_shape, 2, X_dim, **tkwargs)
        train_Y = train_X.sum(dim=-1, keepdim=True)
        train_comp = torch.topk(train_Y, k=2, dim=-2).indices.transpose(-1, -2)

        return train_X, train_Y, train_comp

    def _get_model_and_data(self, batch_shape, X_dim=2, **tkwargs):
        train_X, train_Y, train_comp = self._make_rand_mini_data(
            batch_shape=batch_shape, X_dim=X_dim, **tkwargs
        )

        model_kwargs = {"datapoints": train_X, "comparisons": train_comp}
        model = PairwiseGP(**model_kwargs)
        return model, model_kwargs

    def test_pairwise_gp(self):
        for batch_shape, dtype in itertools.product(
            (torch.Size(), torch.Size([2])), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            X_dim = 2

            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape, X_dim=X_dim, **tkwargs
            )
            train_X = model_kwargs["datapoints"]
            train_comp = model_kwargs["comparisons"]

            # test training
            # regular training
            mll = PairwiseLaplaceMarginalLogLikelihood(model).to(**tkwargs)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                fit_gpytorch_model(mll, options={"maxiter": 2}, max_retries=1)
            # prior training
            prior_m = PairwiseGP(None, None)
            with self.assertRaises(RuntimeError):
                prior_m(train_X)
            # forward in training mode with non-training data
            custom_m = PairwiseGP(**model_kwargs)
            other_X = torch.rand(batch_shape + torch.Size([3, X_dim]), **tkwargs)
            other_comp = train_comp.clone()
            with self.assertRaises(RuntimeError):
                custom_m(other_X)
            custom_mll = PairwiseLaplaceMarginalLogLikelihood(custom_m).to(**tkwargs)
            post = custom_m(train_X)
            with self.assertRaises(RuntimeError):
                custom_mll(post, other_comp)

            # setting jitter = 0 with a singular covar will raise error
            sing_train_X = torch.ones(batch_shape + torch.Size([10, X_dim]), **tkwargs)
            with self.assertRaises(RuntimeError):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    custom_m = PairwiseGP(sing_train_X, train_comp, jitter=0)
                    custom_m.posterior(sing_train_X)

            # test init
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(model.covar_module, RBFKernel)
            self.assertIsInstance(model.covar_module.lengthscale_prior, GammaPrior)
            self.assertEqual(model.num_outputs, 1)

            # test custom models
            custom_m = PairwiseGP(**model_kwargs, covar_module=LinearKernel())
            self.assertIsInstance(custom_m.covar_module, LinearKernel)
            # std_noise setter
            custom_m.std_noise = 123
            self.assertTrue(torch.all(custom_m.std_noise == 123))
            # prior prediction
            prior_m = PairwiseGP(None, None)
            prior_m.eval()
            post = prior_m.posterior(train_X)
            self.assertIsInstance(post, GPyTorchPosterior)

            # test methods that are not commonly or explicitly used
            # _calc_covar with observation noise
            no_noise_cov = model._calc_covar(train_X, train_X, observation_noise=False)
            noise_cov = model._calc_covar(train_X, train_X, observation_noise=True)
            diag_diff = (noise_cov - no_noise_cov).diagonal(dim1=-2, dim2=-1)
            self.assertTrue(
                torch.allclose(
                    diag_diff,
                    model.std_noise.expand(diag_diff.shape),
                    rtol=1e-4,
                    atol=1e-5,
                )
            )
            # test trying adding jitter
            pd_mat = torch.eye(2, 2)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                jittered_pd_mat = model._add_jitter(pd_mat)
            diag_diff = (jittered_pd_mat - pd_mat).diagonal(dim1=-2, dim2=-1)
            self.assertTrue(
                torch.allclose(
                    diag_diff,
                    torch.full_like(diag_diff, model._jitter),
                    atol=model._jitter / 10,
                )
            )

            # test initial utility val
            util_comp = torch.topk(model.utility, k=2, dim=-1).indices.unsqueeze(-2)
            self.assertTrue(torch.all(util_comp == train_comp))

            # test posterior
            # test non batch evaluation
            X = torch.rand(batch_shape + torch.Size([3, X_dim]), **tkwargs)
            expected_shape = batch_shape + torch.Size([3, 1])
            posterior = model.posterior(X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, expected_shape)
            self.assertEqual(posterior.variance.shape, expected_shape)

            # expect to raise error when output_indices is not None
            with self.assertRaises(RuntimeError):
                model.posterior(X, output_indices=[0])

            # test re-evaluating utility when it's None
            model.utility = None
            posterior = model.posterior(X)
            self.assertIsInstance(posterior, GPyTorchPosterior)

            # test adding observation noise
            posterior_pred = model.posterior(X, observation_noise=True)
            self.assertIsInstance(posterior_pred, GPyTorchPosterior)
            self.assertEqual(posterior_pred.mean.shape, expected_shape)
            self.assertEqual(posterior_pred.variance.shape, expected_shape)
            pvar = posterior_pred.variance
            reshaped_noise = model.std_noise.unsqueeze(-2).expand(
                posterior.variance.shape
            )
            pvar_exp = posterior.variance + reshaped_noise
            self.assertTrue(torch.allclose(pvar, pvar_exp, rtol=1e-4, atol=1e-5))

            # test batch evaluation
            X = torch.rand(2, *batch_shape, 3, X_dim, **tkwargs)
            expected_shape = torch.Size([2]) + batch_shape + torch.Size([3, 1])

            posterior = model.posterior(X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, expected_shape)
            # test adding observation noise in batch mode
            posterior_pred = model.posterior(X, observation_noise=True)
            self.assertIsInstance(posterior_pred, GPyTorchPosterior)
            self.assertEqual(posterior_pred.mean.shape, expected_shape)
            pvar = posterior_pred.variance
            reshaped_noise = model.std_noise.unsqueeze(-2).expand(
                posterior.variance.shape
            )
            pvar_exp = posterior.variance + reshaped_noise
            self.assertTrue(torch.allclose(pvar, pvar_exp, rtol=1e-4, atol=1e-5))

    def test_condition_on_observations(self):
        for batch_shape, dtype in itertools.product(
            (torch.Size(), torch.Size([2])), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            X_dim = 2

            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape, X_dim=X_dim, **tkwargs
            )
            train_X = model_kwargs["datapoints"]
            train_comp = model_kwargs["comparisons"]

            # evaluate model
            model.posterior(torch.rand(torch.Size([4, X_dim]), **tkwargs))
            # test condition_on_observations

            # test condition_on_observations with prior mode
            prior_m = PairwiseGP(None, None)
            cond_m = prior_m.condition_on_observations(train_X, train_comp)
            self.assertTrue(cond_m.datapoints is train_X)
            self.assertTrue(cond_m.comparisons is train_comp)

            # fantasize at different input points
            fant_shape = torch.Size([2])
            X_fant, Y_fant, comp_fant = self._make_rand_mini_data(
                batch_shape=fant_shape + batch_shape, X_dim=X_dim, **tkwargs
            )

            # cannot condition on non-pairwise Ys
            with self.assertRaises(RuntimeError):
                model.condition_on_observations(X_fant, comp_fant[..., 0])
            cm = model.condition_on_observations(X_fant, comp_fant)
            # make sure it's a deep copy
            self.assertTrue(model is not cm)

            # fantasize at same input points (check proper broadcasting)
            cm_same_inputs = model.condition_on_observations(X_fant[0], comp_fant)

            test_Xs = [
                # test broadcasting single input across fantasy and model batches
                torch.rand(4, X_dim, **tkwargs),
                # separate input for each model batch and broadcast across
                # fantasy batches
                torch.rand(batch_shape + torch.Size([4, X_dim]), **tkwargs),
                # separate input for each model and fantasy batch
                torch.rand(
                    fant_shape + batch_shape + torch.Size([4, X_dim]), **tkwargs
                ),
            ]
            for test_X in test_Xs:
                posterior = cm.posterior(test_X)
                self.assertEqual(
                    posterior.mean.shape, fant_shape + batch_shape + torch.Size([4, 1])
                )
                posterior_same_inputs = cm_same_inputs.posterior(test_X)
                self.assertEqual(
                    posterior_same_inputs.mean.shape,
                    fant_shape + batch_shape + torch.Size([4, 1]),
                )

                # check that fantasies of batched model are correct
                if len(batch_shape) > 0 and test_X.dim() == 2:
                    state_dict_non_batch = {
                        key: (val[0] if val.numel() > 1 else val)
                        for key, val in model.state_dict().items()
                    }
                    model_kwargs_non_batch = {
                        "datapoints": model_kwargs["datapoints"][0],
                        "comparisons": model_kwargs["comparisons"][0],
                    }
                    model_non_batch = model.__class__(**model_kwargs_non_batch)
                    model_non_batch.load_state_dict(state_dict_non_batch)
                    model_non_batch.eval()
                    model_non_batch.posterior(
                        torch.rand(torch.Size([4, X_dim]), **tkwargs)
                    )
                    cm_non_batch = model_non_batch.condition_on_observations(
                        X_fant[0][0], comp_fant[:, 0, :]
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
                            posterior_same_inputs.mvn.covariance_matrix[:, 0, :, :],
                            non_batch_posterior.mvn.covariance_matrix,
                            atol=1e-3,
                        )
                    )

    def test_fantasize(self):
        for batch_shape, dtype in itertools.product(
            (torch.Size(), torch.Size([2])), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            X_dim = 2

            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape, X_dim=X_dim, **tkwargs
            )

            # fantasize
            X_f = torch.rand(
                torch.Size(batch_shape + torch.Size([4, X_dim])), **tkwargs
            )
            sampler = PairwiseSobolQMCNormalSampler(num_samples=3)
            fm = model.fantasize(X=X_f, sampler=sampler)
            self.assertIsInstance(fm, model.__class__)
            fm = model.fantasize(X=X_f, sampler=sampler, observation_noise=False)
            self.assertIsInstance(fm, model.__class__)
