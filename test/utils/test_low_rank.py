#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.exceptions.errors import BotorchError
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.samplers import IIDNormalSampler
from botorch.utils.low_rank import (
    extract_batch_covar,
    sample_cached_cholesky,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from gpytorch.lazy import lazify
from gpytorch.lazy.block_diag_lazy_tensor import BlockDiagLazyTensor
from gpytorch.utils.errors import NanError


class TestExtractBatchCovar(BotorchTestCase):
    def test_extract_batch_covar(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            base_covar = torch.tensor(
                [[1.0, 0.6, 0.9], [0.6, 1.0, 0.5], [0.9, 0.5, 1.0]], **tkwargs
            )
            lazy_covar = lazify(torch.stack([base_covar, base_covar * 2], dim=0))
            block_diag_covar = BlockDiagLazyTensor(lazy_covar)
            mt_mvn = MultitaskMultivariateNormal(
                torch.zeros(3, 2, **tkwargs), block_diag_covar
            )
            batch_covar = extract_batch_covar(mt_mvn=mt_mvn)
            self.assertTrue(torch.equal(batch_covar.evaluate(), lazy_covar.evaluate()))
            # test non BlockDiagLazyTensor
            mt_mvn = MultitaskMultivariateNormal(
                torch.zeros(3, 2, **tkwargs), block_diag_covar.evaluate()
            )
            with self.assertRaises(BotorchError):
                extract_batch_covar(mt_mvn=mt_mvn)


class TestSampleCachedCholesky(BotorchTestCase):
    def test_sample_cached_cholesky(self):
        torch.manual_seed(0)
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            train_X = torch.rand(10, 2, **tkwargs)
            train_Y = torch.randn(10, 2, **tkwargs)
            for m in (1, 2):
                model_list_values = (True, False) if m == 2 else (False,)
                for use_model_list in model_list_values:
                    if use_model_list:
                        model = ModelListGP(
                            SingleTaskGP(
                                train_X,
                                train_Y[..., :1],
                            ),
                            SingleTaskGP(
                                train_X,
                                train_Y[..., 1:],
                            ),
                        )
                    else:
                        model = SingleTaskGP(
                            train_X,
                            train_Y[:, :m],
                        )
                    sampler = IIDNormalSampler(3)
                    for q in (1, 3):
                        # test batched baseline_L
                        for train_batch_shape in (
                            torch.Size([]),
                            torch.Size([3]),
                            torch.Size([3, 2]),
                        ):
                            # test batched test points
                            for test_batch_shape in (
                                torch.Size([]),
                                torch.Size([4]),
                                torch.Size([4, 2]),
                            ):

                                if len(train_batch_shape) > 0:
                                    train_X_ex = train_X.unsqueeze(0).expand(
                                        train_batch_shape + train_X.shape
                                    )
                                else:
                                    train_X_ex = train_X
                                if len(test_batch_shape) > 0:
                                    test_X = train_X_ex.unsqueeze(0).expand(
                                        test_batch_shape + train_X_ex.shape
                                    )
                                else:
                                    test_X = train_X_ex
                                with torch.no_grad():
                                    base_posterior = model.posterior(
                                        train_X_ex[..., :-q, :]
                                    )
                                    mvn = base_posterior.mvn
                                    lazy_covar = mvn.lazy_covariance_matrix
                                    if m == 2:
                                        lazy_covar = lazy_covar.base_lazy_tensor
                                    baseline_L = lazy_covar.root_decomposition()
                                    baseline_L = baseline_L.root.evaluate()
                                test_X = test_X.clone().requires_grad_(True)
                                new_posterior = model.posterior(test_X)
                                samples = sampler(new_posterior)
                                samples[..., -q:, :].sum().backward()
                                test_X2 = test_X.detach().clone().requires_grad_(True)
                                new_posterior2 = model.posterior(test_X2)
                                q_samples = sample_cached_cholesky(
                                    posterior=new_posterior2,
                                    baseline_L=baseline_L,
                                    q=q,
                                    base_samples=sampler.base_samples.detach().clone(),
                                    sample_shape=sampler.sample_shape,
                                )
                                q_samples.sum().backward()
                                all_close_kwargs = (
                                    {
                                        "atol": 1e-4,
                                        "rtol": 1e-2,
                                    }
                                    if dtype == torch.float
                                    else {}
                                )
                                self.assertTrue(
                                    torch.allclose(
                                        q_samples.detach(),
                                        samples[..., -q:, :].detach(),
                                        **all_close_kwargs,
                                    )
                                )
                                self.assertTrue(
                                    torch.allclose(
                                        test_X2.grad[..., -q:, :],
                                        test_X.grad[..., -q:, :],
                                        **all_close_kwargs,
                                    )
                                )
                            # test nans
                            with torch.no_grad():
                                test_posterior = model.posterior(test_X2)
                            test_posterior.mvn.loc = torch.full_like(
                                test_posterior.mvn.loc, float("nan")
                            )
                            with self.assertRaises(NanError):
                                sample_cached_cholesky(
                                    posterior=test_posterior,
                                    baseline_L=baseline_L,
                                    q=q,
                                    base_samples=sampler.base_samples.detach().clone(),
                                    sample_shape=sampler.sample_shape,
                                )
                            # test infs
                            test_posterior.mvn.loc = torch.full_like(
                                test_posterior.mvn.loc, float("inf")
                            )
                            with self.assertRaises(NanError):
                                sample_cached_cholesky(
                                    posterior=test_posterior,
                                    baseline_L=baseline_L,
                                    q=q,
                                    base_samples=sampler.base_samples.detach().clone(),
                                    sample_shape=sampler.sample_shape,
                                )
