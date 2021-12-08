#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from contextlib import ExitStack
from unittest import mock

import torch
from botorch import settings
from botorch.acquisition.multi_objective.objective import (
    MCMultiOutputObjective,
    UnstandardizeMCMultiOutputObjective,
)
from botorch.acquisition.multi_objective.utils import (
    get_default_partitioning_alpha,
    extract_batch_covar,
    sample_cached_cholesky,
    prune_inferior_points_multi_objective,
)
from botorch.exceptions.errors import BotorchError, UnsupportedError
from botorch.exceptions.warnings import SamplingWarning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.samplers import IIDNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.lazy import lazify
from gpytorch.lazy.block_diag_lazy_tensor import BlockDiagLazyTensor
from gpytorch.utils.errors import NanError
from torch import Tensor


class TestUtils(BotorchTestCase):
    def test_get_default_partitioning_alpha(self):
        for m in range(2, 7):
            expected_val = 0.0 if m < 5 else 10 ** (-8 + m)
            self.assertEqual(
                expected_val, get_default_partitioning_alpha(num_objectives=m)
            )
        # In `BotorchTestCase.setUp` warnings are filtered, so here we
        # remove the filter to ensure a warning is issued as expected.
        warnings.resetwarnings()
        with warnings.catch_warnings(record=True) as ws:
            self.assertEqual(0.1, get_default_partitioning_alpha(num_objectives=7))
        self.assertEqual(len(ws), 1)


class DummyMCMultiOutputObjective(MCMultiOutputObjective):
    def forward(self, samples: Tensor) -> Tensor:
        return samples


class TestMultiObjectiveUtils(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.model = mock.MagicMock()
        self.objective = DummyMCMultiOutputObjective()
        self.X_observed = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        self.X_pending = torch.tensor([[1.0, 3.0, 4.0]])
        self.mc_samples = 250
        self.qmc = True
        self.ref_point = [0.0, 0.0]
        self.Y = torch.tensor([[1.0, 2.0]])
        self.seed = 1

    def test_prune_inferior_points_multi_objective(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            X = torch.rand(3, 2, **tkwargs)
            ref_point = torch.tensor([0.25, 0.25], **tkwargs)
            # the event shape is `q x m` = 3 x 2
            samples = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], **tkwargs)
            mm = MockModel(MockPosterior(samples=samples))
            # test that a batched X raises errors
            with self.assertRaises(UnsupportedError):
                prune_inferior_points_multi_objective(
                    model=mm, X=X.expand(2, 3, 2), ref_point=ref_point
                )
            # test that a batched model raises errors (event shape is `q x m` = 3 x m)
            mm2 = MockModel(MockPosterior(samples=samples.expand(2, 3, 2)))
            with self.assertRaises(UnsupportedError):
                prune_inferior_points_multi_objective(
                    model=mm2, X=X, ref_point=ref_point
                )
            # test that invalid max_frac is checked properly
            with self.assertRaises(ValueError):
                prune_inferior_points_multi_objective(
                    model=mm, X=X, max_frac=1.1, ref_point=ref_point
                )
            # test basic behaviour
            X_pruned = prune_inferior_points_multi_objective(
                model=mm, X=X, ref_point=ref_point
            )
            self.assertTrue(torch.equal(X_pruned, X[[-1]]))
            # test unstd objective
            unstd_obj = UnstandardizeMCMultiOutputObjective(
                Y_mean=samples.mean(dim=0), Y_std=samples.std(dim=0), outcomes=[0, 1]
            )
            X_pruned = prune_inferior_points_multi_objective(
                model=mm, X=X, ref_point=ref_point, objective=unstd_obj
            )
            self.assertTrue(torch.equal(X_pruned, X[[-1]]))
            # test constraints
            samples_constrained = torch.tensor(
                [[1.0, 2.0, -1.0], [2.0, 1.0, -1.0], [3.0, 4.0, 1.0]], **tkwargs
            )
            mm_constrained = MockModel(MockPosterior(samples=samples_constrained))
            X_pruned = prune_inferior_points_multi_objective(
                model=mm_constrained,
                X=X,
                ref_point=ref_point,
                objective=unstd_obj,
                constraints=[lambda Y: Y[..., -1]],
            )
            self.assertTrue(torch.equal(X_pruned, X[:2]))

            # test non-repeated samples (requires mocking out MockPosterior's rsample)
            samples = torch.tensor(
                [[[3.0], [0.0], [0.0]], [[0.0], [2.0], [0.0]], [[0.0], [0.0], [1.0]]],
                device=self.device,
                dtype=dtype,
            )
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points_multi_objective(
                    model=mm, X=X, ref_point=ref_point
                )
            self.assertTrue(torch.equal(X_pruned, X))
            # test max_frac limiting
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points_multi_objective(
                    model=mm, X=X, ref_point=ref_point, max_frac=2 / 3
                )
            if self.device.type == "cuda":
                # sorting has different order on cuda
                self.assertTrue(torch.equal(X_pruned, torch.stack([X[2], X[1]], dim=0)))
            else:
                self.assertTrue(torch.equal(X_pruned, X[:2]))
            # test that zero-probability is in fact pruned
            samples[2, 0, 0] = 10
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points_multi_objective(
                    model=mm, X=X, ref_point=ref_point
                )
            self.assertTrue(torch.equal(X_pruned, X[:2]))
            # test high-dim sampling
            with ExitStack() as es:
                mock_event_shape = es.enter_context(
                    mock.patch(
                        "botorch.utils.testing.MockPosterior.event_shape",
                        new_callable=mock.PropertyMock,
                    )
                )
                mock_event_shape.return_value = torch.Size(
                    [1, 1, torch.quasirandom.SobolEngine.MAXDIM + 1]
                )
                es.enter_context(
                    mock.patch.object(MockPosterior, "rsample", return_value=samples)
                )
                mm = MockModel(MockPosterior(samples=samples))
                with warnings.catch_warnings(record=True) as ws, settings.debug(True):
                    prune_inferior_points_multi_objective(
                        model=mm, X=X, ref_point=ref_point
                    )
                    self.assertTrue(issubclass(ws[-1].category, SamplingWarning))

            # test marginalize_dim and constraints
            samples = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], **tkwargs)
            samples = samples.unsqueeze(-3).expand(
                *samples.shape[:-2],
                2,
                *samples.shape[-2:],
            )
            mm = MockModel(MockPosterior(samples=samples))
            X_pruned = prune_inferior_points_multi_objective(
                model=mm,
                X=X,
                ref_point=ref_point,
                objective=unstd_obj,
                constraints=[lambda Y: Y[..., -1] - 3.0],
                marginalize_dim=-3,
            )
            self.assertTrue(torch.equal(X_pruned, X[:2]))


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
        # test single output posterior
        with self.assertRaises(NotImplementedError):
            sample_cached_cholesky(
                posterior=GPyTorchPosterior(
                    MultivariateNormal(torch.zeros(2), torch.eye(2))
                ),
                baseline_L=None,
                q=1,
                base_samples=None,
                sample_shape=None,
            )
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            train_X = torch.rand(10, 2, **tkwargs)
            train_Y = torch.randn(10, 2, **tkwargs)
            for use_model_list in (True, False):
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
                        train_Y,
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
                                lazy_covar = mvn.lazy_covariance_matrix.base_lazy_tensor
                                lazy_root = lazy_covar.root_decomposition()
                                baseline_L = lazy_root.root.evaluate()
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
