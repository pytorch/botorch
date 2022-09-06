#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from typing import List, Optional, Tuple
from unittest import TestCase

import torch
from botorch import settings
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.test_functions.base import BaseTestProblem
from botorch.utils.transforms import unnormalize
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from linear_operator.operators import AddedDiagLinearOperator, DiagLinearOperator
from torch import Tensor


EMPTY_SIZE = torch.Size()


class BotorchTestCase(TestCase):
    r"""Basic test case for Botorch.

    This
        1. sets the default device to be `torch.device("cpu")`
        2. ensures that no warnings are suppressed by default.
    """

    device = torch.device("cpu")

    def setUp(self):
        warnings.resetwarnings()
        settings.debug._set_state(False)
        warnings.simplefilter("always", append=True)


class BaseTestProblemBaseTestCase:

    functions: List[BaseTestProblem]

    def test_forward(self):
        for dtype in (torch.float, torch.double):
            for batch_shape in (torch.Size(), torch.Size([2]), torch.Size([2, 3])):
                for f in self.functions:
                    f.to(device=self.device, dtype=dtype)
                    X = torch.rand(*batch_shape, f.dim, device=self.device, dtype=dtype)
                    X = f.bounds[0] + X * (f.bounds[1] - f.bounds[0])
                    res = f(X)
                    f(X, noise=False)
                    self.assertEqual(res.dtype, dtype)
                    self.assertEqual(res.device.type, self.device.type)
                    tail_shape = torch.Size(
                        [f.num_objectives] if f.num_objectives > 1 else []
                    )
                    self.assertEqual(res.shape, batch_shape + tail_shape)


class SyntheticTestFunctionBaseTestCase(BaseTestProblemBaseTestCase):
    def test_optimal_value(self):
        for dtype in (torch.float, torch.double):
            for f in self.functions:
                f.to(device=self.device, dtype=dtype)
                try:
                    optval = f.optimal_value
                    optval_exp = -f._optimal_value if f.negate else f._optimal_value
                    self.assertEqual(optval, optval_exp)
                except NotImplementedError:
                    pass

    def test_optimizer(self):
        for dtype in (torch.float, torch.double):
            for f in self.functions:
                f.to(device=self.device, dtype=dtype)
                try:
                    Xopt = f.optimizers.clone().requires_grad_(True)
                except NotImplementedError:
                    continue
                res = f(Xopt, noise=False)
                # if we have optimizers, we have the optimal value
                res_exp = torch.full_like(res, f.optimal_value)
                self.assertTrue(torch.allclose(res, res_exp, atol=1e-3, rtol=1e-3))
                if f._check_grad_at_opt:
                    grad = torch.autograd.grad([*res], Xopt)[0]
                    self.assertLess(grad.abs().max().item(), 1e-3)


class MockPosterior(Posterior):
    r"""Mock object that implements dummy methods and feeds through specified outputs"""

    def __init__(self, mean=None, variance=None, samples=None):
        r"""
        Args:
            mean: The mean of the posterior.
            variance: The variance of the posterior.
            samples: Samples to return from `rsample`, unless `base_samples` is
                provided.
        """
        self._mean = mean
        self._variance = variance
        self._samples = samples

    @property
    def device(self) -> torch.device:
        for t in (self._mean, self._variance, self._samples):
            if torch.is_tensor(t):
                return t.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        for t in (self._mean, self._variance, self._samples):
            if torch.is_tensor(t):
                return t.dtype
        return torch.float32

    @property
    def event_shape(self) -> torch.Size:
        if self._samples is not None:
            return self._samples.shape
        if self._mean is not None:
            return self._mean.shape
        if self._variance is not None:
            return self._variance.shape
        return torch.Size()

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        """Mock sample by repeating self._samples. If base_samples is provided,
        do a shape check but return the same mock samples."""
        if sample_shape is None:
            sample_shape = torch.Size()
        if sample_shape is not None and base_samples is not None:
            # check the base_samples shape is consistent with the sample_shape
            if base_samples.shape[: len(sample_shape)] != sample_shape:
                raise RuntimeError("sample_shape disagrees with base_samples.")
        return self._samples.expand(sample_shape + self._samples.shape)


class MockModel(Model):
    r"""Mock object that implements dummy methods and feeds through specified outputs"""

    def __init__(self, posterior: MockPosterior) -> None:  # noqa: D107
        super(Model, self).__init__()
        self._posterior = posterior

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        observation_noise: bool = False,
    ) -> MockPosterior:
        if posterior_transform is not None:
            return posterior_transform(self._posterior)
        else:
            return self._posterior

    @property
    def num_outputs(self) -> int:
        event_shape = self._posterior.event_shape
        return event_shape[-1] if len(event_shape) > 0 else 0

    @property
    def batch_shape(self) -> torch.Size:
        event_shape = self._posterior.event_shape
        return event_shape[:-2]

    def state_dict(self) -> None:
        pass

    def load_state_dict(
        self, state_dict: Optional[OrderedDict] = None, strict: bool = False
    ) -> None:
        pass


class MockAcquisitionFunction:
    r"""Mock acquisition function object that implements dummy methods."""

    def __init__(self):  # noqa: D107
        self.model = None
        self.X_pending = None

    def __call__(self, X):
        return X[..., 0].max(dim=-1).values

    def set_X_pending(self, X_pending: Optional[Tensor] = None):
        self.X_pending = X_pending


def _get_random_data(
    batch_shape: torch.Size, m: int, d: int = 1, n: int = 10, **tkwargs
) -> Tuple[Tensor, Tensor]:
    r"""Generate random data for testing purposes.

    Args:
        batch_shape: The batch shape of the data.
        m: The number of outputs.
        d: The dimension of the input.
        n: The number of data points.
        tkwargs: `device` and `dtype` tensor constructor kwargs.

    Returns:
        A tuple `(train_X, train_Y)` with randomly generated training data.
    """
    rep_shape = batch_shape + torch.Size([1, 1])
    train_x = torch.stack(
        [torch.linspace(0, 0.95, n, **tkwargs) for _ in range(d)], dim=-1
    )
    train_x = train_x + 0.05 * torch.rand_like(train_x).repeat(rep_shape)
    train_y = torch.sin(train_x[..., :1] * (2 * math.pi))
    train_y = train_y + 0.2 * torch.randn(n, m, **tkwargs).repeat(rep_shape)
    return train_x, train_y


def _get_test_posterior(
    batch_shape: torch.Size,
    q: int = 1,
    m: int = 1,
    interleaved: bool = True,
    lazy: bool = False,
    independent: bool = False,
    **tkwargs,
) -> GPyTorchPosterior:
    r"""Generate a Posterior for testing purposes.

    Args:
        batch_shape: The batch shape of the data.
        q: The number of candidates
        m: The number of outputs.
        interleaved: A boolean indicating the format of the
            MultitaskMultivariateNormal
        lazy: A boolean indicating if the posterior should be lazy
        independent: A boolean indicating whether the outputs are independent
        tkwargs: `device` and `dtype` tensor constructor kwargs.


    """
    if independent:
        mvns = []
        for _ in range(m):
            mean = torch.rand(*batch_shape, q, **tkwargs)
            a = torch.rand(*batch_shape, q, q, **tkwargs)
            covar = a @ a.transpose(-1, -2)
            flat_diag = torch.rand(*batch_shape, q, **tkwargs)
            covar = covar + torch.diag_embed(flat_diag)
            mvns.append(MultivariateNormal(mean, covar))
        mtmvn = MultitaskMultivariateNormal.from_independent_mvns(mvns)
    else:
        mean = torch.rand(*batch_shape, q, m, **tkwargs)
        a = torch.rand(*batch_shape, q * m, q * m, **tkwargs)
        covar = a @ a.transpose(-1, -2)
        flat_diag = torch.rand(*batch_shape, q * m, **tkwargs)
        if lazy:
            covar = AddedDiagLinearOperator(covar, DiagLinearOperator(flat_diag))
        else:
            covar = covar + torch.diag_embed(flat_diag)
        mtmvn = MultitaskMultivariateNormal(mean, covar, interleaved=interleaved)
    return GPyTorchPosterior(mtmvn)


class MultiObjectiveTestProblemBaseTestCase(BaseTestProblemBaseTestCase):
    def test_attributes(self):
        for f in self.functions:
            self.assertTrue(hasattr(f, "dim"))
            self.assertTrue(hasattr(f, "num_objectives"))
            self.assertEqual(f.bounds.shape, torch.Size([2, f.dim]))

    def test_max_hv(self):
        for dtype in (torch.float, torch.double):
            for f in self.functions:
                f.to(device=self.device, dtype=dtype)
                if not hasattr(f, "_max_hv"):
                    with self.assertRaises(NotImplementedError):
                        f.max_hv
                else:
                    self.assertEqual(f.max_hv, f._max_hv)

    def test_ref_point(self):
        for dtype in (torch.float, torch.double):
            for f in self.functions:
                f.to(dtype=dtype, device=self.device)
                self.assertTrue(
                    torch.allclose(
                        f.ref_point,
                        torch.tensor(f._ref_point, dtype=dtype, device=self.device),
                    )
                )


class ConstrainedMultiObjectiveTestProblemBaseTestCase(
    MultiObjectiveTestProblemBaseTestCase
):
    def test_num_constraints(self):
        for f in self.functions:
            self.assertTrue(hasattr(f, "num_constraints"))

    def test_evaluate_slack_true(self):
        for dtype in (torch.float, torch.double):
            for f in self.functions:
                f.to(device=self.device, dtype=dtype)
                X = unnormalize(
                    torch.rand(1, f.dim, device=self.device, dtype=dtype),
                    bounds=f.bounds,
                )
                slack = f.evaluate_slack_true(X)
                self.assertEqual(slack.shape, torch.Size([1, f.num_constraints]))
