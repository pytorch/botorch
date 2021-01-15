#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.gp_sampling import GPDraw
from botorch.utils.testing import BotorchTestCase


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
    return model


class TestGPDraw(BotorchTestCase):
    def test_gp_draw_single_output(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            model = _get_model(**tkwargs)
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
            gp = GPDraw(_get_model(**tkwargs))
            with torch.no_grad():
                Y_batch = gp(torch.rand(2, 1, 1, **tkwargs))
            self.assertEqual(Y_batch.shape, torch.Size([2, 1, 1]))
            # test random seed
            test_X = torch.rand(1, 1, **tkwargs)
            gp_a = GPDraw(model=_get_model(**tkwargs), seed=0)
            self.assertEqual(int(gp_a._seed), 0)
            with torch.no_grad():
                Ya = gp_a(test_X)
            self.assertEqual(int(gp_a._seed), 1)
            gp_b = GPDraw(model=_get_model(**tkwargs), seed=0)
            with torch.no_grad():
                Yb = gp_b(test_X)
            self.assertAlmostEqual(Ya, Yb)

    def test_gp_draw_multi_output(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            model = _get_model(**tkwargs, multi_output=True)
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
            gp = GPDraw(_get_model(**tkwargs, multi_output=True))
            with torch.no_grad():
                Y_batch = gp(torch.rand(2, 1, 1, **tkwargs))
            self.assertEqual(Y_batch.shape, torch.Size([2, 1, 2]))
