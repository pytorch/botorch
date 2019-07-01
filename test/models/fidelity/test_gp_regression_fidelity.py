#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import unittest

import torch
from botorch import fit_gpytorch_model
from botorch.exceptions import UnsupportedError
from botorch.models.fidelity.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gp_regression import FixedNoiseGP
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


def _get_random_data_with_fidelity(
    batch_shape,
    num_outputs,
    n=10,
    train_iteration_fidelity=True,
    train_data_fidelity=True,
    **tkwargs,
):
    m = train_iteration_fidelity + train_data_fidelity
    train_x = torch.linspace(0, 0.95, n, **tkwargs).unsqueeze(-1) + 0.05 * torch.rand(
        n, 1, **tkwargs
    ).repeat(batch_shape + torch.Size([1, 1]))
    s = torch.rand(n, m, **tkwargs).repeat(batch_shape + torch.Size([1, 1]))
    train_X = torch.cat((train_x, s), dim=-1)
    train_y = (
        torch.sin(train_x * (2 * math.pi))
        + 0.2
        * torch.randn(n, num_outputs, **tkwargs).repeat(
            batch_shape + torch.Size([1, 1])
        )
        + (1 - s).pow(2).sum(dim=-1).unsqueeze(-1)
    )
    if num_outputs == 1:
        train_y = train_y.squeeze(-1)
    return train_X, train_y


class TestSingleTaskGPFidelity(unittest.TestCase):
    def _get_model_and_data(
        self,
        train_iteration_fidelity,
        train_data_fidelity,
        batch_shape,
        num_outputs,
        **tkwargs,
    ):
        train_X, train_Y = _get_random_data_with_fidelity(
            batch_shape=batch_shape,
            num_outputs=num_outputs,
            train_iteration_fidelity=train_iteration_fidelity,
            train_data_fidelity=train_data_fidelity,
            **tkwargs,
        )
        model_kwargs = {
            "train_X": train_X,
            "train_Y": train_Y,
            "train_iteration_fidelity": train_iteration_fidelity,
            "train_data_fidelity": train_data_fidelity,
        }
        model = SingleTaskMultiFidelityGP(**model_kwargs)
        return model, model_kwargs

    def test_exception_message(self, cuda=False):
        train_X = torch.rand(20, 4, device=torch.device("cuda" if cuda else "cpu"))
        train_Y = train_X.pow(2).sum(dim=-1)
        with self.assertRaises(UnsupportedError):
            SingleTaskMultiFidelityGP(
                train_X,
                train_Y,
                train_iteration_fidelity=False,
                train_data_fidelity=False,
            )

    def test_exception_message_cuda(self):
        if torch.cuda.is_available():
            self.test_exception_message(cuda=True)

    def test_gp(self, cuda=False):
        for (train_iteration_fidelity, train_data_fidelity) in [
            (False, True),
            (True, False),
            (True, True),
        ]:
            for batch_shape in (torch.Size(), torch.Size([2])):
                for num_outputs in (1, 2):
                    for double in (False, True):
                        num_dim = 1 + train_iteration_fidelity + train_data_fidelity
                        tkwargs = {
                            "device": torch.device("cuda")
                            if cuda
                            else torch.device("cpu"),
                            "dtype": torch.double if double else torch.float,
                        }
                        model, _ = self._get_model_and_data(
                            batch_shape=batch_shape,
                            num_outputs=num_outputs,
                            train_iteration_fidelity=train_iteration_fidelity,
                            train_data_fidelity=train_data_fidelity,
                            **tkwargs,
                        )
                        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(
                            **tkwargs
                        )
                        fit_gpytorch_model(
                            mll, sequential=False, options={"maxiter": 1}
                        )

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
                        X = torch.rand(
                            batch_shape + torch.Size([3, num_dim]), **tkwargs
                        )
                        posterior = model.posterior(X)
                        self.assertIsInstance(posterior, GPyTorchPosterior)
                        self.assertEqual(
                            posterior.mean.shape,
                            batch_shape + torch.Size([3, num_outputs]),
                        )
                        # test batch evaluation
                        X = torch.rand(
                            torch.Size([2]) + batch_shape + torch.Size([3, num_dim]),
                            **tkwargs,
                        )
                        posterior = model.posterior(X)
                        self.assertIsInstance(posterior, GPyTorchPosterior)
                        self.assertEqual(
                            posterior.mean.shape,
                            torch.Size([2])
                            + batch_shape
                            + torch.Size([3, num_outputs]),
                        )

    def test_gp_cuda(self):
        if torch.cuda.is_available():
            self.test_gp(cuda=True)

    def test_condition_on_observations(self, cuda=False):
        for (train_iteration_fidelity, train_data_fidelity) in [
            (False, True),
            (True, False),
            (True, True),
        ]:
            for batch_shape in (torch.Size(), torch.Size([2])):
                for num_outputs in (1, 2):
                    for double in (False, True):
                        num_dim = 1 + train_iteration_fidelity + train_data_fidelity
                        tkwargs = {
                            "device": torch.device("cuda")
                            if cuda
                            else torch.device("cpu"),
                            "dtype": torch.double if double else torch.float,
                        }
                        model, model_kwargs = self._get_model_and_data(
                            batch_shape=batch_shape,
                            num_outputs=num_outputs,
                            train_iteration_fidelity=train_iteration_fidelity,
                            train_data_fidelity=train_data_fidelity,
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
                            n=3,
                            train_iteration_fidelity=train_iteration_fidelity,
                            train_data_fidelity=train_data_fidelity,
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
                            torch.rand(
                                batch_shape + torch.Size([4, num_dim]), **tkwargs
                            ),
                            # separate input for each model and fantasy batch
                            torch.rand(
                                fant_shape + batch_shape + torch.Size([4, num_dim]),
                                **tkwargs,
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
                                    "train_iteration_fidelity": model_kwargs[
                                        "train_iteration_fidelity"
                                    ],
                                    "train_data_fidelity": model_kwargs[
                                        "train_data_fidelity"
                                    ],
                                }
                                if "train_Yvar" in model_kwargs:
                                    model_kwargs_non_batch["train_Yvar"] = model_kwargs[
                                        "train_Yvar"
                                    ][0]
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
                                        posterior_same_inputs.mvn.covariance_matrix[
                                            :, 0, :, :
                                        ],
                                        non_batch_posterior.mvn.covariance_matrix,
                                        atol=1e-3,
                                    )
                                )

    def test_condition_on_observations_cuda(self):
        if torch.cuda.is_available():
            self.test_condition_on_observations(cuda=True)

    def test_fantasize(self, cuda=False):
        for (train_iteration_fidelity, train_data_fidelity) in [
            (False, True),
            (True, False),
            (True, True),
        ]:
            num_dim = 1 + train_iteration_fidelity + train_data_fidelity
            for batch_shape in (torch.Size(), torch.Size([2])):
                for num_outputs in (1, 2):
                    for double in (False, True):
                        tkwargs = {
                            "device": torch.device("cuda")
                            if cuda
                            else torch.device("cpu"),
                            "dtype": torch.double if double else torch.float,
                        }
                        model, model_kwargs = self._get_model_and_data(
                            batch_shape=batch_shape,
                            num_outputs=num_outputs,
                            train_iteration_fidelity=train_iteration_fidelity,
                            train_data_fidelity=train_data_fidelity,
                            **tkwargs,
                        )
                        # fantasize
                        X_f = torch.rand(
                            torch.Size(batch_shape + torch.Size([4, num_dim])),
                            **tkwargs,
                        )
                        sampler = SobolQMCNormalSampler(num_samples=3)
                        fm = model.fantasize(X=X_f, sampler=sampler)
                        self.assertIsInstance(fm, model.__class__)
                        fm = model.fantasize(
                            X=X_f, sampler=sampler, observation_noise=False
                        )
                        self.assertIsInstance(fm, model.__class__)

    def test_fantasize_cuda(self):
        if torch.cuda.is_available():
            self.test_fantasize(cuda=True)
