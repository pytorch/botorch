#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import product

import torch
from botorch.models import (
    FixedNoiseGP,
    ModelListGP,
    SingleTaskGP,
    SingleTaskVariationalGP,
)
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.pathwise import draw_matheron_paths, MatheronPath, PathList
from botorch.sampling.pathwise.utils import get_train_inputs
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import MaternKernel, ScaleKernel
from torch import Size
from torch.nn.functional import pad

from .helpers import get_sample_moments, standardize_moments


class TestPosteriorSamplers(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.models = defaultdict(list)

        seed = 0
        for kernel in (
            ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2, batch_shape=Size([]))),
        ):
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                tkwargs = {"device": self.device, "dtype": torch.float64}

                base = kernel.base_kernel if isinstance(kernel, ScaleKernel) else kernel
                base.lengthscale = 0.1 + 0.3 * torch.rand_like(base.lengthscale)
                kernel.to(**tkwargs)

                uppers = 1 + 9 * torch.rand(base.lengthscale.shape[-1], **tkwargs)
                bounds = pad(uppers.unsqueeze(0), (0, 0, 1, 0))

                X = uppers * torch.rand(4, base.lengthscale.shape[-1], **tkwargs)
                Y = 10 * kernel(X).cholesky() @ torch.randn(4, 1, **tkwargs)
                if kernel.batch_shape:
                    Y = Y.squeeze(-1).transpose(0, 1)  # n x m

                input_transform = Normalize(d=X.shape[-1], bounds=bounds)
                outcome_transform = Standardize(m=Y.shape[-1])

                # SingleTaskGP in eval mode
                self.models[SingleTaskGP].append(
                    SingleTaskGP(
                        train_X=X,
                        train_Y=Y,
                        covar_module=deepcopy(kernel),
                        input_transform=deepcopy(input_transform),
                        outcome_transform=deepcopy(outcome_transform),
                    )
                    .to(**tkwargs)
                    .eval()
                )

                # FixedNoiseGP in train mode
                self.models[FixedNoiseGP].append(
                    FixedNoiseGP(
                        train_X=X,
                        train_Y=Y,
                        train_Yvar=0.01 * torch.rand_like(Y),
                        covar_module=kernel,
                        input_transform=input_transform,
                        outcome_transform=outcome_transform,
                    ).to(**tkwargs)
                )

                # SingleTaskVariationalGP in train mode
                self.models[SingleTaskVariationalGP].append(
                    SingleTaskVariationalGP(
                        train_X=X,
                        train_Y=Y,
                        covar_module=kernel,
                        input_transform=input_transform,
                        outcome_transform=outcome_transform,
                    ).to(**tkwargs)
                )

            seed += 1

    def test_draw_matheron_paths(self):
        for seed, models in enumerate(self.models.values()):
            for model, sample_shape in product(models, [Size([1024]), Size([32, 32])]):
                with torch.random.fork_rng():
                    torch.random.manual_seed(seed)
                    paths = draw_matheron_paths(model=model, sample_shape=sample_shape)
                    self.assertIsInstance(paths, MatheronPath)
                    self._test_draw_matheron_paths(model, paths, sample_shape)

        with self.subTest("test_model_list"):
            model_list = ModelListGP(
                self.models[SingleTaskGP][0], self.models[FixedNoiseGP][0]
            )
            path_list = draw_matheron_paths(model_list, sample_shape=sample_shape)
            (train_X,) = get_train_inputs(model_list.models[0], transformed=False)
            X = torch.zeros(
                4, train_X.shape[-1], dtype=train_X.dtype, device=self.device
            )
            sample_list = path_list(X)
            self.assertIsInstance(path_list, PathList)
            self.assertIsInstance(sample_list, list)
            self.assertEqual(len(sample_list), len(path_list.paths))

    def _test_draw_matheron_paths(self, model, paths, sample_shape, atol=3):
        (train_X,) = get_train_inputs(model, transformed=False)
        X = torch.rand(16, train_X.shape[-1], dtype=train_X.dtype, device=self.device)

        # Evaluate sample paths and compute sample statistics
        samples = paths(X)
        batch_shape = (
            model.model.covar_module.batch_shape
            if isinstance(model, SingleTaskVariationalGP)
            else model.covar_module.batch_shape
        )
        self.assertEqual(samples.shape, sample_shape + batch_shape + X.shape[-2:-1])

        sample_moments = get_sample_moments(samples, sample_shape)
        if hasattr(model, "outcome_transform"):
            # Do this instead of untransforming exact moments
            sample_moments = standardize_moments(
                model.outcome_transform, *sample_moments
            )

        if model.training:
            model.eval()
            mvn = model(model.transform_inputs(X))
            model.train()
        else:
            mvn = model(model.transform_inputs(X))
        exact_moments = (mvn.loc, mvn.covariance_matrix)

        # Compare moments
        num_features = paths["prior_paths"].weight.shape[-1]
        tol = atol * (num_features**-0.5 + sample_shape.numel() ** -0.5)
        for exact, estimate in zip(exact_moments, sample_moments):
            self.assertTrue(exact.allclose(estimate, atol=tol, rtol=0))
