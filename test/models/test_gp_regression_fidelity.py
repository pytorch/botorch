#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch import fit_gpytorch_model
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase, _get_random_data
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


def _get_random_data_with_fidelity(
    batch_shape, num_outputs, n_fidelity, n=10, **tkwargs
):
    r"""Construct test data.
    For this test, by convention the trailing dimesions are the fidelity dimensions
    """
    train_x, train_y = _get_random_data(batch_shape, num_outputs, n, **tkwargs)
    s = torch.rand(n, n_fidelity, **tkwargs).repeat(batch_shape + torch.Size([1, 1]))
    train_x = torch.cat((train_x, s), dim=-1)
    train_y = train_y + (1 - s).pow(2).sum(dim=-1).unsqueeze(-1)
    return train_x, train_y


def _get_model_and_data(
    iteration_fidelity,
    data_fidelity,
    batch_shape,
    num_outputs,
    lin_truncated,
    **tkwargs,
):
    n_fidelity = (iteration_fidelity is not None) + (data_fidelity is not None)
    train_X, train_Y = _get_random_data_with_fidelity(
        batch_shape=batch_shape,
        num_outputs=num_outputs,
        n_fidelity=n_fidelity,
        **tkwargs,
    )
    model_kwargs = {
        "train_X": train_X,
        "train_Y": train_Y,
        "iteration_fidelity": iteration_fidelity,
        "data_fidelity": data_fidelity,
        "linear_truncated": lin_truncated,
    }
    model = SingleTaskMultiFidelityGP(**model_kwargs)
    return model, model_kwargs


class TestSingleTaskMultiFidelityGP(BotorchTestCase):

    FIDELITY_TEST_PAIRS = ((None, 1), (1, None), (None, -1), (-1, None), (1, 2))

    def test_init_error(self):
        train_X = torch.rand(2, 2, device=self.device)
        train_Y = torch.rand(2, 1)
        for lin_truncated in (True, False):
            with self.assertRaises(UnsupportedError):
                SingleTaskMultiFidelityGP(
                    train_X, train_Y, linear_truncated=lin_truncated
                )

    def test_gp(self):
        for (iteration_fidelity, data_fidelity) in self.FIDELITY_TEST_PAIRS:
            num_dim = 1 + (iteration_fidelity is not None) + (data_fidelity is not None)
            for batch_shape, num_outputs, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (1, 2),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, _ = _get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelity=data_fidelity,
                    batch_shape=batch_shape,
                    num_outputs=num_outputs,
                    lin_truncated=lin_trunc,
                    **tkwargs,
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                mll.to(**tkwargs)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=OptimizationWarning)
                    fit_gpytorch_model(mll, sequential=False, options={"maxiter": 1})

                # test init
                self.assertIsInstance(model.mean_module, ConstantMean)
                self.assertIsInstance(model.covar_module, ScaleKernel)

                # test param sizes
                params = dict(model.named_parameters())
                for p in params:
                    self.assertEqual(
                        params[p].numel(),
                        num_outputs * torch.tensor(batch_shape).prod().item(),
                    )

                # test posterior
                # test non batch evaluation
                X = torch.rand(batch_shape + torch.Size([3, num_dim]), **tkwargs)
                posterior = model.posterior(X)
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertEqual(
                    posterior.mean.shape, batch_shape + torch.Size([3, num_outputs])
                )
                # test batch evaluation
                X = torch.rand(
                    torch.Size([2]) + batch_shape + torch.Size([3, num_dim]), **tkwargs
                )
                posterior = model.posterior(X)
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertEqual(
                    posterior.mean.shape,
                    torch.Size([2]) + batch_shape + torch.Size([3, num_outputs]),
                )

    def test_condition_on_observations(self):
        for (iteration_fidelity, data_fidelity) in self.FIDELITY_TEST_PAIRS:
            n_fidelity = (iteration_fidelity is not None) + (data_fidelity is not None)
            num_dim = 1 + n_fidelity
            for batch_shape, num_outputs, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (1, 2),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, model_kwargs = _get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelity=data_fidelity,
                    batch_shape=batch_shape,
                    num_outputs=num_outputs,
                    lin_truncated=lin_trunc,
                    **tkwargs,
                )
                # evaluate model
                model.posterior(torch.rand(torch.Size([4, num_dim]), **tkwargs))
                # test condition_on_observations
                fant_shape = torch.Size([2])
                # fantasize at different input points
                X_fant, Y_fant = _get_random_data_with_fidelity(
                    fant_shape + batch_shape,
                    num_outputs,
                    n_fidelity=n_fidelity,
                    n=3,
                    **tkwargs,
                )
                c_kwargs = (
                    {"noise": torch.full_like(Y_fant, 0.01)}
                    if isinstance(model, FixedNoiseGP)
                    else {}
                )
                cm = model.condition_on_observations(X_fant, Y_fant, **c_kwargs)
                # fantasize at different same input points
                c_kwargs_same_inputs = (
                    {"noise": torch.full_like(Y_fant[0], 0.01)}
                    if isinstance(model, FixedNoiseGP)
                    else {}
                )
                cm_same_inputs = model.condition_on_observations(
                    X_fant[0], Y_fant, **c_kwargs_same_inputs
                )

                test_Xs = [
                    # test broadcasting single input across fantasy and
                    # model batches
                    torch.rand(4, num_dim, **tkwargs),
                    # separate input for each model batch and broadcast across
                    # fantasy batches
                    torch.rand(batch_shape + torch.Size([4, num_dim]), **tkwargs),
                    # separate input for each model and fantasy batch
                    torch.rand(
                        fant_shape + batch_shape + torch.Size([4, num_dim]), **tkwargs
                    ),
                ]
                for test_X in test_Xs:
                    posterior = cm.posterior(test_X)
                    self.assertEqual(
                        posterior.mean.shape,
                        fant_shape + batch_shape + torch.Size([4, num_outputs]),
                    )
                    posterior_same_inputs = cm_same_inputs.posterior(test_X)
                    self.assertEqual(
                        posterior_same_inputs.mean.shape,
                        fant_shape + batch_shape + torch.Size([4, num_outputs]),
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
                            "iteration_fidelity": model_kwargs["iteration_fidelity"],
                            "data_fidelity": model_kwargs["data_fidelity"],
                            "linear_truncated": model_kwargs["linear_truncated"],
                        }
                        model_non_batch = type(model)(**model_kwargs_non_batch)
                        model_non_batch.load_state_dict(state_dict_non_batch)
                        model_non_batch.eval()
                        model_non_batch.likelihood.eval()
                        model_non_batch.posterior(
                            torch.rand(torch.Size([4, num_dim]), **tkwargs)
                        )
                        c_kwargs = (
                            {"noise": torch.full_like(Y_fant[0, 0, :], 0.01)}
                            if isinstance(model, FixedNoiseGP)
                            else {}
                        )
                        mnb = model_non_batch
                        cm_non_batch = mnb.condition_on_observations(
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
                                posterior_same_inputs.mvn.covariance_matrix[:, 0, :, :],
                                non_batch_posterior.mvn.covariance_matrix,
                                atol=1e-3,
                            )
                        )

    def test_fantasize(self):
        for (iteration_fidelity, data_fidelity) in self.FIDELITY_TEST_PAIRS:
            n_fidelity = (iteration_fidelity is not None) + (data_fidelity is not None)
            num_dim = 1 + n_fidelity
            for batch_shape, num_outputs, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (1, 2),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, model_kwargs = _get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelity=data_fidelity,
                    batch_shape=batch_shape,
                    num_outputs=num_outputs,
                    lin_truncated=lin_trunc,
                    **tkwargs,
                )
                # fantasize
                X_f = torch.rand(
                    torch.Size(batch_shape + torch.Size([4, num_dim])), **tkwargs
                )
                sampler = SobolQMCNormalSampler(num_samples=3)
                fm = model.fantasize(X=X_f, sampler=sampler)
                self.assertIsInstance(fm, model.__class__)
                fm = model.fantasize(X=X_f, sampler=sampler, observation_noise=False)
                self.assertIsInstance(fm, model.__class__)
