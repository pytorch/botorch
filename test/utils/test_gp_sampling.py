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
    get_deterministic_model_multi_samples,
    get_weights_posterior,
    get_gp_samples,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, PeriodicKernel
from torch.distributions import MultivariateNormal


def _get_model(dtype, device, multi_output=False):
    tkwargs = {"dtype": dtype, "device": device}
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
        ],
        **tkwargs,
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
        ],
        **tkwargs,
    )
    state_dict = {
        "likelihood.noise_covar.raw_noise": torch.tensor([0.0214], **tkwargs),
        "likelihood.noise_covar.noise_prior.concentration": torch.tensor(
            1.1000, **tkwargs
        ),
        "likelihood.noise_covar.noise_prior.rate": torch.tensor(0.0500, **tkwargs),
        "mean_module.constant": torch.tensor([0.1398], **tkwargs),
        "covar_module.raw_outputscale": torch.tensor(0.6933, **tkwargs),
        "covar_module.base_kernel.raw_lengthscale": torch.tensor(
            [[-0.0444]], **tkwargs
        ),
        "covar_module.base_kernel.lengthscale_prior.concentration": torch.tensor(
            3.0, **tkwargs
        ),
        "covar_module.base_kernel.lengthscale_prior.rate": torch.tensor(6.0, **tkwargs),
        "covar_module.outputscale_prior.concentration": torch.tensor(2.0, **tkwargs),
        "covar_module.outputscale_prior.rate": torch.tensor(0.1500, **tkwargs),
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
            ],
            **tkwargs,
        )
        train_Y = torch.cat([train_Y, train_Y2], dim=-1)
        state_dict["likelihood.noise_covar.raw_noise"] = torch.stack(
            [
                state_dict["likelihood.noise_covar.raw_noise"],
                torch.tensor([0.0745], **tkwargs),
            ]
        )
        state_dict["mean_module.constant"] = torch.stack(
            [state_dict["mean_module.constant"], torch.tensor([0.3276], **tkwargs)]
        )
        state_dict["covar_module.raw_outputscale"] = torch.stack(
            [
                state_dict["covar_module.raw_outputscale"],
                torch.tensor(0.4394, **tkwargs),
            ],
            dim=-1,
        )
        state_dict["covar_module.base_kernel.raw_lengthscale"] = torch.stack(
            [
                state_dict["covar_module.base_kernel.raw_lengthscale"],
                torch.tensor([[-0.4617]], **tkwargs),
            ]
        )

    model = SingleTaskGP(train_X, train_Y)
    model.load_state_dict(state_dict)
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

            for sample_shape in [torch.Size(), torch.Size([5])]:
                # test not ScaleKernel
                rff = RandomFourierFeatures(
                    kernel=base_kernel,
                    input_dim=2,
                    num_rff_features=3,
                    sample_shape=sample_shape,
                )
                self.assertTrue(
                    torch.equal(rff.outputscale, torch.tensor(1, **tkwargs))
                )
                self.assertTrue(torch.equal(rff.lengthscale, base_kernel.lengthscale))
                # check that rff makes a copy
                self.assertFalse(rff.lengthscale is kernel.lengthscale)
                self.assertEqual(rff.weights.shape, torch.Size([*sample_shape, 2, 3]))
                self.assertEqual(rff.bias.shape, torch.Size([*sample_shape, 3]))
                self.assertTrue(((rff.bias <= 2 * pi) & (rff.bias >= 0.0)).all())

            # test forward
            for sample_shape in [torch.Size(), torch.Size([7])]:
                rff = RandomFourierFeatures(
                    kernel=kernel,
                    input_dim=2,
                    num_rff_features=3,
                    sample_shape=sample_shape,
                )
                for input_batch_shape in [torch.Size([]), torch.Size([5])]:
                    X = torch.rand(*input_batch_shape, *sample_shape, 1, 2, **tkwargs)
                    Y = rff(X)
                    self.assertTrue(
                        Y.shape, torch.Size([*input_batch_shape, *sample_shape, 1, 1])
                    )
                    _constant = torch.sqrt(2 * rff.outputscale / rff.weights.shape[-1])
                    _arg_to_cos = X / base_kernel.lengthscale @ rff.weights
                    _bias_expanded = rff.bias.unsqueeze(-2)
                    expected_Y = _constant * (torch.cos(_arg_to_cos + _bias_expanded))
                    self.assertTrue(torch.allclose(Y, expected_Y))

            # test get_weights
            for sample_shape in [torch.Size(), torch.Size([5])]:
                with mock.patch("torch.randn", wraps=torch.randn) as mock_randn:
                    rff._get_weights(
                        base_kernel=base_kernel,
                        input_dim=2,
                        num_rff_features=3,
                        sample_shape=sample_shape,
                    )
                    mock_randn.assert_called_once_with(
                        *sample_shape,
                        2,
                        3,
                        dtype=base_kernel.lengthscale.dtype,
                        device=base_kernel.lengthscale.device,
                    )
                # test get_weights with Matern kernel
                with mock.patch(
                    "torch.randn", wraps=torch.randn
                ) as mock_randn, mock.patch(
                    "torch.distributions.Gamma", wraps=torch.distributions.Gamma
                ) as mock_gamma:
                    base_kernel = MaternKernel(ard_num_dims=2).to(**tkwargs)
                    rff._get_weights(
                        base_kernel=base_kernel,
                        input_dim=2,
                        num_rff_features=3,
                        sample_shape=sample_shape,
                    )
                    mock_randn.assert_called_once_with(
                        *sample_shape,
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

    def test_get_deterministic_model_multi_samples(self):
        tkwargs = {"device": self.device}
        n_samples = 5
        for dtype, m in product((torch.float, torch.double), (1, 2)):
            tkwargs["dtype"] = dtype
            for batch_shape_w, batch_shape_x in product(
                [torch.Size([]), torch.Size([3])], repeat=2
            ):
                weights = []
                bases = []
                for i in range(m):
                    num_rff = 2 * (i + 2)
                    # we require weights to be of shape
                    # `n_samples x (batch_shape) x num_rff`
                    weights.append(
                        torch.rand(*batch_shape_w, n_samples, num_rff, **tkwargs)
                    )
                    kernel = ScaleKernel(RBFKernel(ard_num_dims=2)).to(**tkwargs)
                    kernel.outputscale = 0.3 + torch.rand(1, **tkwargs).view(
                        kernel.outputscale.shape
                    )
                    kernel.base_kernel.lengthscale = 0.3 + torch.rand(
                        2, **tkwargs
                    ).view(kernel.base_kernel.lengthscale.shape)
                    bases.append(
                        RandomFourierFeatures(
                            kernel=kernel,
                            input_dim=2,
                            num_rff_features=num_rff,
                            sample_shape=torch.Size([n_samples]),
                        )
                    )

                model = get_deterministic_model_multi_samples(
                    weights=weights, bases=bases
                )
                self.assertIsInstance(model, DeterministicModel)
                self.assertEqual(model.num_outputs, m)
                X = torch.rand(*batch_shape_x, n_samples, 1, 2, **tkwargs)
                Y = model(X)
                for i in range(m):
                    wi = weights[i]
                    for _ in range(len(batch_shape_x)):
                        wi = wi.unsqueeze(-3)
                    wi = wi.expand(*batch_shape_w, *batch_shape_x, *wi.shape[-2:])
                    expected_Yi = (bases[i](X) @ wi.unsqueeze(-1)).squeeze(-1)
                    self.assertTrue(torch.allclose(Y[..., i], expected_Yi))
                self.assertEqual(
                    Y.shape,
                    torch.Size([*batch_shape_w, *batch_shape_x, n_samples, 1, m]),
                )

    def test_get_weights_posterior(self):
        tkwargs = {"device": self.device}
        sigma = 0.01
        input_dim = 2
        for dtype in (torch.float, torch.double):
            for input_batch_shape in [torch.Size(), torch.Size([5])]:
                for sample_shape in [torch.Size(), torch.Size([3])]:
                    tkwargs["dtype"] = dtype
                    X = torch.rand(*input_batch_shape, 40, input_dim, **tkwargs)
                    w = torch.rand(*sample_shape, input_dim, **tkwargs)
                    # We have to share each sample of weights with the X.
                    # Therefore, the effective size of w is
                    # (sample_shape) x (input_batch_shape) x input_dim.
                    for _ in range(len(input_batch_shape)):
                        w.unsqueeze_(-2)
                    w = w.expand(*sample_shape, *input_batch_shape, input_dim)
                    Y_true = (X @ w.unsqueeze(-1)).squeeze(-1)
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
            for mtype in [True, False]:
                model, X, Y = _get_model(**tkwargs, multi_output=m == 2)
                use_batch_model = mtype and m == 2
                gp_samples = get_gp_samples(
                    model=batched_to_model_list(model) if use_batch_model else model,
                    num_outputs=m,
                    n_samples=20,
                    num_rff_features=500,
                )
                self.assertEqual(len(gp_samples(X)), 20)
                self.assertIsInstance(gp_samples, DeterministicModel)
                Y_hat_rff = gp_samples(X).mean(dim=0)
                with torch.no_grad():
                    Y_hat = model.posterior(X).mean
                self.assertTrue(torch.allclose(Y_hat_rff, Y_hat, atol=2e-1))

                # test batched evaluation
                Y_batched = gp_samples(torch.randn(13, 20, 3, X.shape[-1], **tkwargs))
                self.assertEqual(Y_batched.shape, torch.Size([13, 20, 3, m]))

        # test incorrect batch shape check
        with self.assertRaises(ValueError):
            gp_samples(torch.randn(13, 23, 3, X.shape[-1], **tkwargs))
