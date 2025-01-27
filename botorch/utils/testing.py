#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import warnings
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from itertools import product
from typing import Any
from unittest import mock, TestCase

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.warnings import (
    BotorchTensorDimensionWarning,
    InputDataWarning,
    NumericsWarning,
)
from botorch.models.model import FantasizeMixin, Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.sampling.base import MCSampler
from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.test_functions.base import BaseTestProblem, CorruptedTestProblem
from botorch.test_functions.synthetic import Rosenbrock
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

    def setUp(self, suppress_input_warnings: bool = True) -> None:
        warnings.resetwarnings()
        warnings.simplefilter("always", append=True)
        if suppress_input_warnings:
            warnings.filterwarnings(
                "ignore",
                message="The model inputs are of type",
                category=InputDataWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Non-strict enforcement of botorch tensor conventions.",
                category=BotorchTensorDimensionWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Data \(outcome observations\) is not standardized ",
                category=InputDataWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Data \(input features\) is not",
                category=InputDataWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="has known numerical issues",
                category=NumericsWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Model converter code is deprecated",
                category=DeprecationWarning,
            )

    def assertAllClose(
        self,
        input: Any,
        other: Any,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> None:
        r"""
        Calls torch.testing.assert_close, using the signature and default behavior
        of torch.allclose.

        Example output:
            AssertionError: Scalars are not close!

            Absolute difference: 1.0000034868717194 (up to 0.0001 allowed)
            Relative difference: 0.8348668001940709 (up to 1e-05 allowed)
        """
        # Why not just use the signature and behavior of `torch.testing.assert_close`?
        # Because we used `torch.allclose` for testing in the past, and the two don't
        # behave exactly the same. In particular, `assert_close` requires both `atol`
        # and `rtol` to be set if either one is.
        torch.testing.assert_close(
            input,
            other,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )


class BaseTestProblemTestCaseMixIn:
    def test_forward_and_evaluate_true(self):
        dtypes = (torch.float, torch.double)
        batch_shapes = (torch.Size(), torch.Size([2]), torch.Size([2, 3]))
        for dtype, batch_shape, f in product(dtypes, batch_shapes, self.functions):
            f.to(device=self.device, dtype=dtype)
            X = torch.rand(*batch_shape, f.dim, device=self.device, dtype=dtype)
            X = f.bounds[0] + X * (f.bounds[1] - f.bounds[0])
            res_forward = f(X)
            res_evaluate_true = f.evaluate_true(X)
            for method, res in {
                "forward": res_forward,
                "evaluate_true": res_evaluate_true,
            }.items():
                with self.subTest(
                    f"{dtype}_{batch_shape}_{f.__class__.__name__}_{method}"
                ):
                    self.assertEqual(res.dtype, dtype)
                    self.assertEqual(res.device.type, self.device.type)
                    tail_shape = torch.Size(
                        [f.num_objectives] if f.num_objectives > 1 else []
                    )
                    self.assertEqual(res.shape, batch_shape + tail_shape)

    @property
    @abstractmethod
    def functions(self) -> Sequence[BaseTestProblem]:
        # The functions that should be tested. Typically defined as a class
        # attribute on the test case subclassing this class.
        pass  # pragma: no cover


class SyntheticTestFunctionTestCaseMixin:
    def test_optimal_value(self):
        for dtype in (torch.float, torch.double):
            for f in self.functions:
                f.to(device=self.device, dtype=dtype)
                if f._optimal_value is None:
                    with self.assertRaisesRegex(NotImplementedError, "optimal value"):
                        f.optimal_value
                else:
                    optval = f.optimal_value
                    optval_exp = -f._optimal_value if f.negate else f._optimal_value
                    self.assertEqual(optval, optval_exp)

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
                self.assertAllClose(res, res_exp, atol=1e-3, rtol=1e-3)
                if f._check_grad_at_opt:
                    grad = torch.autograd.grad([*res], Xopt)[0]
                    self.assertLess(grad.abs().max().item(), 1e-3)


class MultiObjectiveTestProblemTestCaseMixin:
    def test_attributes(self):
        for f in self.functions:
            self.assertTrue(hasattr(f, "dim"))
            self.assertTrue(hasattr(f, "num_objectives"))
            self.assertEqual(f.bounds.shape, torch.Size([2, f.dim]))

    def test_max_hv(self):
        for dtype in (torch.float, torch.double):
            for f in self.functions:
                f.to(device=self.device, dtype=dtype)
                if f._max_hv is None:
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


class ConstrainedTestProblemTestCaseMixin:
    def test_num_constraints(self):
        for f in self.functions:
            self.assertTrue(hasattr(f, "num_constraints"))

    def test_evaluate_slack(self):
        for dtype in (torch.float, torch.double):
            for f in self.functions:
                f.to(device=self.device, dtype=dtype)
                X = unnormalize(
                    torch.rand(1, f.dim, device=self.device, dtype=dtype),
                    bounds=f.bounds,
                )
                slack_true = f.evaluate_slack_true(X)
                # Mock out the random generator to ensure that noise realizations are
                # sizable so we don't run into any floating point comparison issues.
                with mock.patch(
                    "botorch.test_functions.base.torch.randn_like",
                    side_effect=lambda y: y,
                ):
                    slack_observed = f.evaluate_slack(X)

                self.assertEqual(slack_true.shape, torch.Size([1, f.num_constraints]))
                self.assertEqual(
                    slack_observed.shape, torch.Size([1, f.num_constraints])
                )
                is_equal = (slack_observed == slack_true).bool()
                if isinstance(f.constraint_noise_std, float):
                    self.assertEqual(
                        is_equal.all().item(), f.constraint_noise_std == 0.0
                    )
                elif isinstance(f.constraint_noise_std, list):
                    for i, noise_std in enumerate(f.constraint_noise_std):
                        self.assertEqual(
                            is_equal[:, i].item(), noise_std in (0.0, None)
                        )
                else:
                    self.assertTrue(is_equal.all().item())


class TestCorruptedProblemsMixin(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)

        def outlier_generator(
            problem: torch.Tensor | BaseTestProblem, X: Any, bounds: Any
        ) -> torch.Tensor:
            return torch.ones(X.shape[0])

        self.outlier_generator = outlier_generator

        self.rosenbrock_problem = CorruptedTestProblem(
            base_test_problem=Rosenbrock(),
            outlier_fraction=1.0,
            outlier_generator=outlier_generator,
            seeds=[1, 2],
        )


class MockPosterior(Posterior):
    r"""Mock object that implements dummy methods and feeds through specified outputs"""

    def __init__(
        self, mean=None, variance=None, samples=None, base_shape=None, batch_range=None
    ) -> None:
        r"""
        Args:
            mean: The mean of the posterior.
            variance: The variance of the posterior.
            samples: Samples to return from `rsample`, unless `base_samples` is
                provided.
            base_shape: If given, this is returned as `base_sample_shape`, and also
                used as the base of the `_extended_shape`.
            batch_range: If given, this is returned as `batch_range`.
                Defaults to (0, -2).
        """
        self._mean = mean
        self._variance = variance
        self._samples = samples
        self._base_shape = base_shape
        self._batch_range = batch_range or (0, -2)

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
    def batch_shape(self) -> torch.Size:
        for t in (self._mean, self._variance, self._samples):
            if torch.is_tensor(t):
                return t.shape[:-2]
        raise NotImplementedError  # pragma: no cover

    def _extended_shape(
        self,
        sample_shape: torch.Size = torch.Size(),  # noqa: B008
    ) -> torch.Size:
        return sample_shape + self.base_sample_shape

    @property
    def base_sample_shape(self) -> torch.Size:
        if self._base_shape is not None:
            return self._base_shape
        if self._samples is not None:
            return self._samples.shape
        if self._mean is not None:
            return self._mean.shape
        if self._variance is not None:
            return self._variance.shape
        return torch.Size()

    @property
    def batch_range(self) -> tuple[int, int]:
        return self._batch_range

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def rsample(
        self,
        sample_shape: torch.Size | None = None,
    ) -> Tensor:
        """Mock sample by repeating self._samples. If base_samples is provided,
        do a shape check but return the same mock samples."""
        if sample_shape is None:
            sample_shape = torch.Size()
        extended_shape = self._extended_shape(sample_shape)
        return self._samples.expand(extended_shape)

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        if base_samples.shape[: len(sample_shape)] != sample_shape:
            raise RuntimeError(
                "`sample_shape` disagrees with shape of `base_samples`. "
                f"Got {sample_shape=} and {base_samples.shape=}."
            )
        return self.rsample(sample_shape)


@GetSampler.register(MockPosterior)
def _get_sampler_mock(
    posterior: MockPosterior, sample_shape: torch.Size, **kwargs: Any
) -> MCSampler:
    r"""Get the dummy `StochasticSampler` for `MockPosterior`."""
    return StochasticSampler(sample_shape=sample_shape, **kwargs)


class MockModel(Model, FantasizeMixin):
    r"""Mock object that implements dummy methods and feeds through specified outputs"""

    def __init__(self, posterior: MockPosterior) -> None:  # noqa: D107
        super(Model, self).__init__()
        self._posterior = posterior

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        posterior_transform: PosteriorTransform | None = None,
        observation_noise: bool | torch.Tensor = False,
    ) -> MockPosterior:
        if posterior_transform is not None:
            return posterior_transform(self._posterior)
        else:
            return self._posterior

    @property
    def num_outputs(self) -> int:
        extended_shape = self._posterior._extended_shape()
        return extended_shape[-1] if len(extended_shape) > 0 else 0

    @property
    def batch_shape(self) -> torch.Size:
        extended_shape = self._posterior._extended_shape()
        return extended_shape[:-2]

    def state_dict(self, *args, **kwargs) -> None:
        pass

    def load_state_dict(
        self, state_dict: OrderedDict | None = None, strict: bool = False
    ) -> None:
        pass


class MockAcquisitionFunction:
    r"""Mock acquisition function object that implements dummy methods."""

    def __init__(self):  # noqa: D107
        self.model = None
        self.X_pending = None

    def __call__(self, X):
        return X[..., 0].max(dim=-1).values

    def set_X_pending(self, X_pending: Tensor | None = None):
        self.X_pending = X_pending


def _get_random_data(
    batch_shape: torch.Size, m: int, d: int = 1, n: int = 10, **tkwargs
) -> tuple[Tensor, Tensor]:
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
    train_x[0] += 0.02  # modify the first batch
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


def _get_max_violation_of_bounds(samples: torch.Tensor, bounds: torch.Tensor) -> float:
    """
    The maximum value by which samples lie outside bounds.

    A negative value indicates that all samples lie within bounds.

    Args:
        samples: An `n x q x d` - dimension tensor, as might be returned from
            `sample_q_batches_from_polytope`.
        bounds: A `2 x d` tensor of lower and upper bounds for each column.
    """
    n, q, d = samples.shape
    samples = samples.reshape((n * q, d))
    lower = samples.min(0).values
    upper = samples.max(0).values
    lower_dist = (bounds[0, :] - lower).max().item()
    upper_dist = (upper - bounds[1, :]).max().item()
    return max(lower_dist, upper_dist)


def _get_max_violation_of_constraints(
    samples: torch.Tensor,
    constraints: list[tuple[Tensor, Tensor, float]] | None,
    equality: bool,
) -> float:
    r"""
    Amount by which equality constraints are not obeyed.

    Args:
        samples: An `n x q x d` - dimension tensor, as might be returned from
            `sample_q_batches_from_polytope`.
        constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`, or `>=` if
            `equality` is False.
        equality: Whether these are equality constraints (not inequality).
    """
    n, q, d = samples.shape
    max_error = 0
    if constraints is not None:
        for ind, coef, rhs in constraints:
            if ind.ndim == 1:
                constr = samples[:, :, ind] @ coef
            else:
                constr = samples[:, ind[:, 0], ind[:, 1]] @ coef

            if equality:
                error = (constr - rhs).abs().max()
            else:
                error = (rhs - constr).max()
            max_error = max(max_error, error)
    return max_error
