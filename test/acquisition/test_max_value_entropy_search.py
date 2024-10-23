#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from unittest import mock

import torch
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.max_value_entropy_search import (
    _sample_max_value_Gumbel,
    _sample_max_value_Thompson,
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
    qMultiFidelityLowerBoundMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.acquisition.objective import (
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from torch import Tensor


class MESMockModel(MockModel):
    r"""Mock object that implements dummy methods and feeds through specified outputs"""

    def __init__(self, num_outputs=1, batch_shape=None):
        r"""
        Args:
            num_outputs: The number of outputs.
            batch_shape: The batch shape of the model. For details see
                `botorch.models.model.Model.batch_shape`.
        """
        super().__init__(None)
        self._num_outputs = num_outputs
        self._batch_shape = torch.Size() if batch_shape is None else batch_shape

    def posterior(
        self,
        X: Tensor,
        observation_noise: bool = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> MockPosterior:
        m_shape = X.shape[:-1]
        r_shape = list(X.shape[:-2]) + [1, 1]
        mvn = MultivariateNormal(
            mean=torch.zeros(m_shape, dtype=X.dtype, device=X.device),
            covariance_matrix=torch.eye(
                m_shape[-1], dtype=X.dtype, device=X.device
            ).repeat(r_shape),
        )
        if self.num_outputs > 1:
            mvn = mvn = MultitaskMultivariateNormal.from_independent_mvns(
                mvns=[mvn] * self.num_outputs
            )
        posterior = GPyTorchPosterior(mvn)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    def forward(self, X: Tensor) -> MultivariateNormal:
        return self.posterior(X).distribution

    @property
    def batch_shape(self) -> torch.Size:
        return self._batch_shape

    @property
    def num_outputs(self) -> int:
        return self._num_outputs


class NoBatchShapeMESMockModel(MESMockModel):
    # For some reason it's really hard to mock this property to raise a
    # NotImplementedError, so let's just make a class for it.
    @property
    def batch_shape(self) -> torch.Size:
        raise NotImplementedError


class TestMaxValueEntropySearch(BotorchTestCase):
    def test_q_max_value_entropy(self):
        for dtype in (torch.float, torch.double):
            torch.manual_seed(7)
            mm = MESMockModel()
            with self.assertRaises(TypeError):
                qMaxValueEntropy(mm)

            candidate_set = torch.rand(1000, 2, device=self.device, dtype=dtype)

            # test error in case of batch GP model
            mm = MESMockModel(batch_shape=torch.Size([2]))
            with self.assertRaises(NotImplementedError):
                qMaxValueEntropy(mm, candidate_set, num_mv_samples=10)
            mm = MESMockModel()
            train_inputs = torch.rand(5, 10, 2, device=self.device, dtype=dtype)
            with self.assertRaises(NotImplementedError):
                qMaxValueEntropy(
                    mm, candidate_set, num_mv_samples=10, train_inputs=train_inputs
                )

            # test that init works if batch_shape is not implemented on the model
            mm = NoBatchShapeMESMockModel()
            qMaxValueEntropy(
                mm,
                candidate_set,
                num_mv_samples=10,
            )

            # test error when number of outputs > 1 and no transform is given.
            mm = MESMockModel()
            mm._num_outputs = 2
            with self.assertRaises(UnsupportedError):
                qMaxValueEntropy(mm, candidate_set, num_mv_samples=10)

            # test with X_pending is None
            mm = MESMockModel()
            train_inputs = torch.rand(10, 2, device=self.device, dtype=dtype)
            mm.train_inputs = (train_inputs,)
            qMVE = qMaxValueEntropy(mm, candidate_set, num_mv_samples=10)

            # test initialization
            self.assertEqual(qMVE.num_fantasies, 16)
            self.assertEqual(qMVE.num_mv_samples, 10)
            self.assertIsInstance(qMVE.sampler, SobolQMCNormalSampler)
            self.assertEqual(qMVE.sampler.sample_shape, torch.Size([128]))
            self.assertIsInstance(qMVE.fantasies_sampler, SobolQMCNormalSampler)
            self.assertEqual(qMVE.fantasies_sampler.sample_shape, torch.Size([16]))
            self.assertEqual(qMVE.use_gumbel, True)
            self.assertEqual(qMVE.posterior_max_values.shape, torch.Size([10, 1]))

            # test evaluation
            X = torch.rand(1, 2, device=self.device, dtype=dtype)
            self.assertEqual(qMVE(X).shape, torch.Size([1]))

            # test set X pending to None in case of _init_model exists
            qMVE.set_X_pending(None)
            self.assertEqual(qMVE.model, qMVE._init_model)

            # test with use_gumbel = False
            qMVE = qMaxValueEntropy(
                mm, candidate_set, num_mv_samples=10, use_gumbel=False
            )
            self.assertEqual(qMVE(X).shape, torch.Size([1]))

            # test with X_pending is not None
            with mock.patch.object(
                MESMockModel, "fantasize", return_value=mm
            ) as patch_f:
                qMVE = qMaxValueEntropy(
                    mm,
                    candidate_set,
                    num_mv_samples=10,
                    X_pending=torch.rand(1, 2, device=self.device, dtype=dtype),
                )
                patch_f.assert_called_once()

            # Test with multi-output model w/ transform.
            mm = MESMockModel(num_outputs=2)
            pt = ScalarizedPosteriorTransform(
                weights=torch.ones(2, device=self.device, dtype=dtype)
            )
            for gumbel in (True, False):
                qMVE = qMaxValueEntropy(
                    mm,
                    candidate_set,
                    num_mv_samples=10,
                    use_gumbel=gumbel,
                    posterior_transform=pt,
                )
                self.assertEqual(qMVE(X).shape, torch.Size([1]))

    def test_q_lower_bound_max_value_entropy(self):
        for dtype in (torch.float, torch.double):
            torch.manual_seed(7)
            mm = MESMockModel()
            with self.assertRaises(TypeError):
                qLowerBoundMaxValueEntropy(mm)

            candidate_set = torch.rand(1000, 2, device=self.device, dtype=dtype)

            # test error in case of batch GP model
            # train_inputs = torch.rand(5, 10, 2, device=self.device, dtype=dtype)
            # mm.train_inputs = (train_inputs,)
            mm = MESMockModel(batch_shape=torch.Size([2]))
            with self.assertRaises(NotImplementedError):
                qLowerBoundMaxValueEntropy(mm, candidate_set, num_mv_samples=10)

            # test error when number of outputs > 1 and no transform
            mm = MESMockModel()
            mm._num_outputs = 2
            with self.assertRaises(UnsupportedError):
                qLowerBoundMaxValueEntropy(mm, candidate_set, num_mv_samples=10)
            mm._num_outputs = 1

            # test with X_pending is None
            mm = MESMockModel()
            train_inputs = torch.rand(10, 2, device=self.device, dtype=dtype)
            mm.train_inputs = (train_inputs,)
            qGIBBON = qLowerBoundMaxValueEntropy(mm, candidate_set, num_mv_samples=10)

            # test initialization
            self.assertEqual(qGIBBON.num_mv_samples, 10)
            self.assertEqual(qGIBBON.use_gumbel, True)
            self.assertEqual(qGIBBON.posterior_max_values.shape, torch.Size([10, 1]))

            # test evaluation
            X = torch.rand(1, 2, device=self.device, dtype=dtype)
            self.assertEqual(qGIBBON(X).shape, torch.Size([1]))

            # test with use_gumbel = False
            qGIBBON = qLowerBoundMaxValueEntropy(
                mm, candidate_set, num_mv_samples=10, use_gumbel=False
            )
            self.assertEqual(qGIBBON(X).shape, torch.Size([1]))

            # test with X_pending is not None
            qGIBBON = qLowerBoundMaxValueEntropy(
                mm,
                candidate_set,
                num_mv_samples=10,
                use_gumbel=False,
                X_pending=torch.rand(1, 2, device=self.device, dtype=dtype),
            )
            self.assertEqual(qGIBBON(X).shape, torch.Size([1]))

            # Test with multi-output model w/ transform.
            mm = MESMockModel(num_outputs=2)
            pt = ScalarizedPosteriorTransform(
                weights=torch.ones(2, device=self.device, dtype=dtype)
            )
            qGIBBON = qLowerBoundMaxValueEntropy(
                mm,
                candidate_set,
                num_mv_samples=10,
                use_gumbel=False,
                X_pending=torch.rand(1, 2, device=self.device, dtype=dtype),
                posterior_transform=pt,
            )
            with self.assertRaisesRegex(UnsupportedError, "X_pending is not None"):
                qGIBBON(X)

    def test_q_multi_fidelity_max_value_entropy(
        self, acqf_class=qMultiFidelityMaxValueEntropy
    ):
        for dtype in (torch.float, torch.double):
            torch.manual_seed(7)
            mm = MESMockModel()
            train_inputs = torch.rand(10, 2, device=self.device, dtype=dtype)
            mm.train_inputs = (train_inputs,)
            candidate_set = torch.rand(10, 2, device=self.device, dtype=dtype)
            qMF_MVE = acqf_class(
                model=mm, candidate_set=candidate_set, num_mv_samples=10
            )

            # test initialization
            self.assertEqual(qMF_MVE.num_fantasies, 16)
            self.assertEqual(qMF_MVE.num_mv_samples, 10)
            self.assertIsInstance(qMF_MVE.sampler, SobolQMCNormalSampler)
            self.assertIsInstance(qMF_MVE.cost_sampler, SobolQMCNormalSampler)
            self.assertEqual(qMF_MVE.sampler.sample_shape, torch.Size([128]))
            self.assertIsInstance(qMF_MVE.fantasies_sampler, SobolQMCNormalSampler)
            self.assertEqual(qMF_MVE.fantasies_sampler.sample_shape, torch.Size([16]))
            self.assertIsInstance(qMF_MVE.expand, Callable)
            self.assertIsInstance(qMF_MVE.project, Callable)
            self.assertIsNone(qMF_MVE.X_pending)
            self.assertEqual(qMF_MVE.posterior_max_values.shape, torch.Size([10, 1]))
            self.assertIsInstance(
                qMF_MVE.cost_aware_utility, InverseCostWeightedUtility
            )

            # test evaluation
            X = torch.rand(1, 2, device=self.device, dtype=dtype)
            self.assertEqual(qMF_MVE(X).shape, torch.Size([1]))

            # Test with multi-output model w/ transform.
            mm = MESMockModel(num_outputs=2)
            pt = ScalarizedPosteriorTransform(
                weights=torch.ones(2, device=self.device, dtype=dtype)
            )
            qMF_MVE = acqf_class(
                model=mm,
                candidate_set=candidate_set,
                num_mv_samples=10,
                posterior_transform=pt,
            )
            X = torch.rand(1, 2, device=self.device, dtype=dtype)
            self.assertEqual(qMF_MVE(X).shape, torch.Size([1]))

    def test_q_multi_fidelity_lower_bound_max_value_entropy(self):
        # Same test as for MF-MES since GIBBON only changes in the way it computes the
        # information gain.
        self.test_q_multi_fidelity_max_value_entropy(
            acqf_class=qMultiFidelityLowerBoundMaxValueEntropy
        )

    def test_sample_max_value_Gumbel(self):
        for dtype in (torch.float, torch.double):
            torch.manual_seed(7)
            mm = MESMockModel()
            candidate_set = torch.rand(3, 10, 2, device=self.device, dtype=dtype)
            samples = _sample_max_value_Gumbel(mm, candidate_set, 5)
            self.assertEqual(samples.shape, torch.Size([5, 3]))

            # Test with multi-output model w/ transform.
            mm = MESMockModel(num_outputs=2)
            pt = ScalarizedPosteriorTransform(
                weights=torch.ones(2, device=self.device, dtype=dtype)
            )
            samples = _sample_max_value_Gumbel(
                mm, candidate_set, 5, posterior_transform=pt
            )
            self.assertEqual(samples.shape, torch.Size([5, 3]))

    def test_sample_max_value_Thompson(self):
        for dtype in (torch.float, torch.double):
            torch.manual_seed(7)
            mm = MESMockModel()
            candidate_set = torch.rand(3, 10, 2, device=self.device, dtype=dtype)
            samples = _sample_max_value_Thompson(mm, candidate_set, 5)
            self.assertEqual(samples.shape, torch.Size([5, 3]))

            # Test with multi-output model w/ transform.
            mm = MESMockModel(num_outputs=2)
            pt = ScalarizedPosteriorTransform(
                weights=torch.ones(2, device=self.device, dtype=dtype)
            )
            samples = _sample_max_value_Thompson(
                mm, candidate_set, 5, posterior_transform=pt
            )
            self.assertEqual(samples.shape, torch.Size([5, 3]))
