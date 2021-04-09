#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
from math import pi
from unittest import mock

import torch
from botorch.models.converter import batched_to_model_list
from botorch.models.deterministic import DeterministicModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.utils.gp_sampling import (
    GPDraw,
    RandomFourierFeatures,
    get_deterministic_model,
    get_weights_posterior,
    get_gp_samples,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, PeriodicKernel
from torch.distributions import MultivariateNormal


def _get_model(device, dtype, multi_output=False):
    train_X = torch.tensor(
        [
            [-0.1000],
            [0.4894],
            [1.0788],
            [1.6681],
            [2.2575],
            [2.8469],
            [3.4363],
            [4.0257],
            [4.6150],
            [5.2044],
            [5.7938],
            [6.3832],
        ]
    )
    train_Y = torch.tensor(
        [
            [-0.0274],
            [0.2612],
            [0.8114],
            [1.1916],
            [1.4870],
            [0.8611],
            [-0.9226],
            [-0.5916],
            [-1.3301],
            [-1.8847],
            [0.0647],
            [1.0900],
        ]
    )
    state_dict = {
        "likelihood.noise_covar.raw_noise": torch.tensor([0.0214]),
        "likelihood.noise_covar.noise_prior.concentration": torch.tensor(1.1000),
        "likelihood.noise_covar.noise_prior.rate": torch.tensor(0.0500),
        "mean_module.constant": torch.tensor([0.1398]),
        "covar_module.raw_outputscale": torch.tensor(0.6933),
        "covar_module.base_kernel.raw_lengthscale": torch.tensor([[-0.0444]]),
        "covar_module.base_kernel.lengthscale_prior.concentration": torch.tensor(3.0),
        "covar_module.base_kernel.lengthscale_prior.rate": torch.tensor(6.0),
        "covar_module.outputscale_prior.concentration": torch.tensor(2.0),
        "covar_module.outputscale_prior.rate": torch.tensor(0.1500),
    }
    if multi_output:
        train_Y2 = torch.tensor(
            [
                [0.9723],
                [1.0652],
                [0.7667],
                [-0.5542],
                [-0.6266],
                [-0.5350],
                [-0.8854],
                [-1.3024],
                [1.0408],
                [0.2485],
                [1.4924],
                [1.5393],
            ]
        )
        train_Y = torch.cat([train_Y, train_Y2], dim=-1)
        state_dict["likelihood.noise_covar.raw_noise"] = torch.stack(
            [state_dict["likelihood.noise_covar.raw_noise"], torch.tensor([0.0745])]
        )
        state_dict["mean_module.constant"] = torch.stack(
            [state_dict["mean_module.constant"], torch.tensor([0.3276])]
        )
        state_dict["covar_module.raw_outputscale"] = torch.stack(
            [state_dict["covar_module.raw_outputscale"], torch.tensor(0.4394)], dim=-1
        )
        state_dict["covar_module.base_kernel.raw_lengthscale"] = torch.stack(
            [
                state_dict["covar_module.base_kernel.raw_lengthscale"],
                torch.tensor([[-0.4617]]),
            ]
        )

    model = SingleTaskGP(train_X, train_Y)
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype)
    return model, train_X, train_Y


class TestGPDraw(BotorchTestCase):
    def test_gp_draw_single_output(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, _, _ = _get_model(**tkwargs)
            mean = model.mean_module.constant.detach().clone()
            gp = GPDraw(model)
            # test initialization
            self.assertIsNone(gp.Xs)
            self.assertIsNone(gp.Ys)
            self.assertIsNotNone(gp._seed)
            # make sure model is actually deepcopied
            model.mean_module.constant = None
            self.assertTrue(torch.equal(gp._model.mean_module.constant, mean))
            # test basic functionality
            test_X1 = torch.rand(1, 1, **tkwargs, requires_grad=True)
            Y1 = gp(test_X1)
            self.assertEqual(Y1.shape, torch.Size([1, 1]))
            Y1.backward()
            self.assertIsNotNone(test_X1.grad)
            initial_base_samples = gp._base_samples
            with torch.no_grad():
                Y2 = gp(torch.rand(1, 1, **tkwargs))
            self.assertEqual(Y2.shape, torch.Size([1, 1]))
            new_base_samples = gp._base_samples
            self.assertTrue(
                torch.equal(initial_base_samples, new_base_samples[..., :1, :])
            )
            # evaluate in batch mode (need a new model for this!)
            model, _, _ = _get_model(**tkwargs)
            gp = GPDraw(model)
            with torch.no_grad():
                Y_batch = gp(torch.rand(2, 1, 1, **tkwargs))
            self.assertEqual(Y_batch.shape, torch.Size([2, 1, 1]))
            # test random seed
            test_X = torch.rand(1, 1, **tkwargs)
            model, _, _ = _get_model(**tkwargs)
            gp_a = GPDraw(model=model, seed=0)
            self.assertEqual(int(gp_a._seed), 0)
            with torch.no_grad():
                Ya = gp_a(test_X)
            self.assertEqual(int(gp_a._seed), 1)
            model, _, _ = _get_model(**tkwargs)
            gp_b = GPDraw(model=model, seed=0)
            with torch.no_grad():
                Yb = gp_b(test_X)
            self.assertAlmostEqual(Ya, Yb)

    def test_gp_draw_multi_output(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, _, _ = _get_model(**tkwargs, multi_output=True)
            mean = model.mean_module.constant.detach().clone()
            gp = GPDraw(model)
            # test initialization
            self.assertIsNone(gp.Xs)
            self.assertIsNone(gp.Ys)
            # make sure model is actually deepcopied
            model.mean_module.constant = None
            self.assertTrue(torch.equal(gp._model.mean_module.constant, mean))
            # test basic functionality
            test_X1 = torch.rand(1, 1, **tkwargs, requires_grad=True)
            Y1 = gp(test_X1)
            self.assertEqual(Y1.shape, torch.Size([1, 2]))
            Y1[:, 1].backward()
            self.assertIsNotNone(test_X1.grad)
            initial_base_samples = gp._base_samples
            with torch.no_grad():
                Y2 = gp(torch.rand(1, 1, **tkwargs))
            self.assertEqual(Y2.shape, torch.Size([1, 2]))
            new_base_samples = gp._base_samples
            self.assertTrue(
                torch.equal(initial_base_samples, new_base_samples[..., :1, :])
            )
            # evaluate in batch mode (need a new model for this!)
            model = model, _, _ = _get_model(**tkwargs, multi_output=True)
            gp = GPDraw(model)
            with torch.no_grad():
                Y_batch = gp(torch.rand(2, 1, 1, **tkwargs))
            self.assertEqual(Y_batch.shape, torch.Size([2, 1, 2]))


class TestRandomFourierFeatures(BotorchTestCase):
    def test_random_fourier_features(self):
        # test kernel that is not Scale, RBF, or Matern
        with self.assertRaises(NotImplementedError):
            RandomFourierFeatures(
                kernel=PeriodicKernel(),
                input_dim=2,
                num_rff_features=3,
            )

        # test batched kernel
        with self.assertRaises(NotImplementedError):
            RandomFourierFeatures(
                kernel=RBFKernel(batch_shape=torch.Size([2])),
                input_dim=2,
                num_rff_features=3,
            )
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            # test init
            # test ScaleKernel
            base_kernel = RBFKernel(ard_num_dims=2)
            kernel = ScaleKernel(base_kernel).to(**tkwargs)
            rff = RandomFourierFeatures(
                kernel=kernel,
                input_dim=2,
                num_rff_features=3,
            )
            self.assertTrue(torch.equal(rff.outputscale, kernel.outputscale))
            # check that rff makes a copy
            self.assertFalse(rff.outputscale is kernel.outputscale)
            self.assertTrue(torch.equal(rff.lengthscale, base_kernel.lengthscale))
            # check that rff makes a copy
            self.assertFalse(rff.lengthscale is kernel.lengthscale)

            # test not ScaleKernel
            rff = RandomFourierFeatures(
                kernel=base_kernel,
                input_dim=2,
                num_rff_features=3,
            )
            self.assertTrue(torch.equal(rff.outputscale, torch.tensor(1, **tkwargs)))
            self.assertTrue(torch.equal(rff.lengthscale, base_kernel.lengthscale))
            # check that rff makes a copy
            self.assertFalse(rff.lengthscale is kernel.lengthscale)
            self.assertEqual(rff.weights.shape, torch.Size([2, 3]))
            self.assertEqual(rff.bias.shape, torch.Size([3]))
            self.assertTrue(((rff.bias <= 2 * pi) & (rff.bias >= 0.0)).all())

            # test forward
            rff = RandomFourierFeatures(
                kernel=kernel,
                input_dim=2,
                num_rff_features=3,
            )
            for batch_shape in (torch.Size([]), torch.Size([3])):
                X = torch.rand(*batch_shape, 1, 2, **tkwargs)
                Y = rff(X)
                self.assertTrue(Y.shape, torch.Size([*batch_shape, 1, 1]))
                expected_Y = torch.sqrt(2 * rff.outputscale / rff.weights.shape[-1]) * (
                    torch.cos(X / base_kernel.lengthscale @ rff.weights + rff.bias)
                )
                self.assertTrue(torch.equal(Y, expected_Y))

            # test get_weights
            with mock.patch("torch.randn", wraps=torch.randn) as mock_randn:
                rff._get_weights(
                    base_kernel=base_kernel, input_dim=2, num_rff_features=3
                )
                mock_randn.assert_called_once_with(
                    2,
                    3,
                    dtype=base_kernel.lengthscale.dtype,
                    device=base_kernel.lengthscale.device,
                )
            # test get_weights with Matern kernel
            with mock.patch("torch.randn", wraps=torch.randn) as mock_randn, mock.patch(
                "torch.distributions.Gamma", wraps=torch.distributions.Gamma
            ) as mock_gamma:
                base_kernel = MaternKernel(ard_num_dims=2).to(**tkwargs)
                rff._get_weights(
                    base_kernel=base_kernel, input_dim=2, num_rff_features=3
                )
                mock_randn.assert_called_once_with(
                    2,
                    3,
                    dtype=base_kernel.lengthscale.dtype,
                    device=base_kernel.lengthscale.device,
                )
                mock_gamma.assert_called_once_with(
                    base_kernel.nu,
                    base_kernel.nu,
                )

    def test_get_deterministic_model(self):
        tkwargs = {"device": self.device}
        for dtype, m in product((torch.float, torch.double), (1, 2)):
            tkwargs["dtype"] = dtype
            weights = []
            bases = []
            for i in range(m):
                num_rff = 2 * (i + 2)
                weights.append(torch.rand(num_rff, **tkwargs))
                kernel = ScaleKernel(RBFKernel(ard_num_dims=2)).to(**tkwargs)
                kernel.outputscale = 0.3 + torch.rand(1, **tkwargs).view(
                    kernel.outputscale.shape
                )
                kernel.base_kernel.lengthscale = 0.3 + torch.rand(2, **tkwargs).view(
                    kernel.base_kernel.lengthscale.shape
                )
                bases.append(
                    RandomFourierFeatures(
                        kernel=kernel,
                        input_dim=2,
                        num_rff_features=num_rff,
                    )
                )

            model = get_deterministic_model(weights=weights, bases=bases)
            self.assertIsInstance(model, DeterministicModel)
            self.assertEqual(model.num_outputs, m)
            for batch_shape in (torch.Size([]), torch.Size([3])):
                X = torch.rand(*batch_shape, 1, 2, **tkwargs)
                Y = model(X)
                expected_Y = torch.stack(
                    [basis(X) @ w for w, basis in zip(weights, bases)], dim=-1
                )
                self.assertTrue(torch.equal(Y, expected_Y))
                self.assertEqual(Y.shape, torch.Size([*batch_shape, 1, m]))

    def test_get_weights_posterior(self):
        tkwargs = {"device": self.device}
        sigma = 0.01
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            X = torch.rand(40, 2, **tkwargs)
            w = torch.rand(2, **tkwargs)
            Y_true = X @ w
            Y = Y_true + sigma * torch.randn_like(Y_true)
            posterior = get_weights_posterior(X=X, y=Y, sigma_sq=sigma ** 2)
            self.assertIsInstance(posterior, MultivariateNormal)
            self.assertTrue(torch.allclose(w, posterior.mean, atol=1e-1))
            w_samp = posterior.sample()
            self.assertEqual(w_samp.shape, w.shape)

    def test_get_gp_samples(self):
        # test multi-task model
        X = torch.stack([torch.rand(3), torch.tensor([1.0, 0.0, 1.0])], dim=-1)
        Y = torch.rand(3, 1)
        with self.assertRaises(NotImplementedError):
            gp_samples = get_gp_samples(
                model=MultiTaskGP(X, Y, task_feature=1),
                num_outputs=1,
                n_samples=20,
                num_rff_features=500,
            )
        tkwargs = {"device": self.device}
        for dtype, m in product((torch.float, torch.double), (1, 2)):
            tkwargs["dtype"] = dtype
            for mtype in range(2):
                model, X, Y = _get_model(**tkwargs, multi_output=m == 2)
                use_batch_model = mtype == 0 and m == 2
                gp_samples = get_gp_samples(
                    model=batched_to_model_list(model) if use_batch_model else model,
                    num_outputs=m,
                    n_samples=20,
                    num_rff_features=500,
                )
                self.assertEqual(len(gp_samples), 20)
                self.assertIsInstance(gp_samples[0], DeterministicModel)
                Y_hat_rff = torch.stack(
                    [gp_sample(X) for gp_sample in gp_samples], dim=0
                ).mean(dim=0)
                with torch.no_grad():
                    Y_hat = model.posterior(X).mean
                self.assertTrue(torch.allclose(Y_hat_rff, Y_hat, atol=2e-1))
