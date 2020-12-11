#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Probability Distributions.

This is adapted from https://github.com/probtorch/pytorch/pull/143.

TODO: replace with PyTorch version once the PR is up and landed.
"""
import random
import unittest
from collections import namedtuple
from itertools import product
from numbers import Number

import torch
from botorch.distributions import Kumaraswamy
from botorch.utils.testing import BotorchTestCase
from torch._six import inf, string_classes
from torch.autograd import grad
from torch.distributions import Distribution, Independent
from torch.distributions.constraints import Constraint, is_dependent

SEED = 1234

Example = namedtuple("Example", ["Dist", "params"])
EXAMPLES = [
    Example(
        Kumaraswamy,
        [
            # avoid extreme parameters
            {
                "concentration1": 0.5 + 3 * torch.rand(2, 3).requires_grad_(),
                "concentration0": 0.5 + 3 * torch.rand(2, 3).requires_grad_(),
            },
            {
                "concentration1": 0.5 + 3 * torch.rand(4).requires_grad_(),
                "concentration0": 0.5 + 3 * torch.rand(4).requires_grad_(),
            },
        ],
    ),
]


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def iter_indices(tensor):
    if tensor.dim() == 0:
        return range(0)
    if tensor.dim() == 1:
        return range(tensor.size(0))
    return product(*(range(s) for s in tensor.size()))


class TestCase(unittest.TestCase):
    precision = 1e-5

    def setUp(self):
        set_rng_seed(SEED)

    def assertEqual(self, x, y, prec=None, message="", allow_inf=False):
        if isinstance(prec, str) and message == "":
            message = prec
            prec = None
        if prec is None:
            prec = self.precision

        if isinstance(x, torch.Tensor) and isinstance(y, Number):
            self.assertEqual(x.item(), y, prec, message, allow_inf)
        elif isinstance(y, torch.Tensor) and isinstance(x, Number):
            self.assertEqual(x, y.item(), prec, message, allow_inf)
        elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):

            def assertTensorsEqual(a, b):
                super(TestCase, self).assertEqual(a.size(), b.size(), message)
                if a.numel() > 0:
                    b = b.type_as(a)
                    b = b.cuda(device=a.get_device()) if a.is_cuda else b.cpu()
                    # check that NaNs are in the same locations
                    nan_mask = a != a
                    self.assertTrue(torch.equal(nan_mask, b != b), message)
                    diff = a - b
                    diff[nan_mask] = 0
                    # TODO: implement abs on CharTensor
                    if diff.is_signed() and "CharTensor" not in diff.type():
                        diff = diff.abs()
                    max_err = diff.max()
                    self.assertLessEqual(max_err, prec, message)

            super(TestCase, self).assertEqual(x.is_sparse, y.is_sparse, message)
            if x.is_sparse:
                x = self.safeCoalesce(x)
                y = self.safeCoalesce(y)
                assertTensorsEqual(x._indices(), y._indices())
                assertTensorsEqual(x._values(), y._values())
            else:
                assertTensorsEqual(x, y)
        elif isinstance(x, string_classes) and isinstance(y, string_classes):
            super(TestCase, self).assertEqual(x, y, message)
        elif type(x) == set and type(y) == set:
            super(TestCase, self).assertEqual(x, y, message)
        elif is_iterable(x) and is_iterable(y):
            super(TestCase, self).assertEqual(len(x), len(y), message)
            for x_, y_ in zip(x, y):
                self.assertEqual(x_, y_, prec, message)
        elif isinstance(x, bool) and isinstance(y, bool):
            super(TestCase, self).assertEqual(x, y, message)
        elif isinstance(x, Number) and isinstance(y, Number):
            if abs(x) == inf or abs(y) == inf:
                if allow_inf:
                    super(TestCase, self).assertEqual(x, y, message)
                else:
                    self.fail(
                        "Expected finite numeric values - x={}, y={}".format(x, y)
                    )
                return
            super(TestCase, self).assertLessEqual(abs(x - y), prec, message)
        else:
            super(TestCase, self).assertEqual(x, y, message)


class TestKumaraswamy(BotorchTestCase, TestCase):
    def test_kumaraswamy_shape(self):
        concentration1 = torch.randn(2, 3).abs().requires_grad_(True)
        concentration0 = torch.randn(2, 3).abs().requires_grad_(True)
        concentration1_1d = torch.randn(1).abs().requires_grad_(True)
        concentration0_1d = torch.randn(1).abs().requires_grad_(True)
        self.assertEqual(
            Kumaraswamy(concentration1, concentration0).sample().size(), (2, 3)
        )
        self.assertEqual(
            Kumaraswamy(concentration1, concentration0).sample((5,)).size(), (5, 2, 3)
        )
        self.assertEqual(
            Kumaraswamy(concentration1_1d, concentration0_1d).sample().size(), (1,)
        )
        self.assertEqual(
            Kumaraswamy(concentration1_1d, concentration0_1d).sample((1,)).size(),
            (1, 1),
        )
        self.assertEqual(Kumaraswamy(1.0, 1.0).sample().size(), ())
        self.assertEqual(Kumaraswamy(1.0, 1.0).sample((1,)).size(), (1,))

    # Kumaraswamy distribution is not implemented in SciPy
    # Hence these tests are explicit
    def test_kumaraswamy_mean_variance(self):
        c1_1 = torch.randn(2, 3).abs().requires_grad_(True)
        c0_1 = torch.randn(2, 3).abs().requires_grad_(True)
        c1_2 = torch.randn(4).abs().requires_grad_(True)
        c0_2 = torch.randn(4).abs().requires_grad_(True)
        cases = [(c1_1, c0_1), (c1_2, c0_2)]
        for i, (a, b) in enumerate(cases):
            m = Kumaraswamy(a, b)
            samples = m.sample((60000,))
            expected = samples.mean(0)
            actual = m.mean
            error = (expected - actual).abs()
            max_error = max(error[error == error])
            self.assertLess(
                max_error,
                0.01,
                "Kumaraswamy example {}/{}, incorrect .mean".format(i + 1, len(cases)),
            )
            expected = samples.var(0)
            actual = m.variance
            error = (expected - actual).abs()
            max_error = max(error[error == error])
            self.assertLess(
                max_error,
                0.01,
                "Kumaraswamy example {}/{}, incorrect .variance".format(
                    i + 1, len(cases)
                ),
            )

    def test_valid_parameter_broadcasting(self):
        valid_examples = [
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0, 1.0]), concentration0=1.0
                ),
                (2,),
            ),
            (
                Kumaraswamy(concentration1=1, concentration0=torch.tensor([1.0, 1.0])),
                (2,),
            ),
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0, 1.0]),
                    concentration0=torch.tensor([1.0]),
                ),
                (2,),
            ),
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0, 1.0]),
                    concentration0=torch.tensor([[1.0], [1.0]]),
                ),
                (2, 2),
            ),
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0, 1.0]),
                    concentration0=torch.tensor([[1.0]]),
                ),
                (1, 2),
            ),
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0]),
                    concentration0=torch.tensor([[1.0]]),
                ),
                (1, 1),
            ),
        ]
        for dist, expected_size in valid_examples:
            dist_sample_size = dist.sample().size()
            self.assertEqual(
                dist_sample_size,
                expected_size,
                "actual size: {} != expected size: {}".format(
                    dist_sample_size, expected_size
                ),
            )

    def test_invalid_parameter_broadcasting(self):
        # invalid broadcasting cases; should throw error
        # example type (distribution class, distribution params)
        invalid_examples = [
            (
                Kumaraswamy,
                {
                    "concentration1": torch.tensor([[1, 1]]),
                    "concentration0": torch.tensor([1, 1, 1, 1]),
                },
            ),
            (
                Kumaraswamy,
                {
                    "concentration1": torch.tensor([[[1, 1, 1], [1, 1, 1]]]),
                    "concentration0": torch.tensor([1, 1]),
                },
            ),
        ]
        for dist, kwargs in invalid_examples:
            self.assertRaises(RuntimeError, dist, **kwargs)

    def _check_enumerate_support(self, dist, examples):
        for params, expected in examples:
            params = {k: torch.tensor(v) for k, v in params.items()}
            expected = torch.tensor(expected)
            d = dist(**params)
            actual = d.enumerate_support(expand=False)
            self.assertEqual(actual, expected)
            actual = d.enumerate_support(expand=True)
            expected_with_expand = expected.expand(
                (-1,) + d.batch_shape + d.event_shape
            )
            self.assertEqual(actual, expected_with_expand)

    def test_repr(self):
        for Dist, params in EXAMPLES:
            for param in params:
                dist = Dist(**param)
                self.assertTrue(repr(dist).startswith(dist.__class__.__name__))

    def test_sample_detached(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                variable_params = [
                    p for p in param.values() if getattr(p, "requires_grad", False)
                ]
                if not variable_params:
                    continue
                dist = Dist(**param)
                sample = dist.sample()
                self.assertFalse(
                    sample.requires_grad,
                    msg="{} example {}/{}, .sample() is not detached".format(
                        Dist.__name__, i + 1, len(params)
                    ),
                )

    def test_rsample_requires_grad(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                if not any(getattr(p, "requires_grad", False) for p in param.values()):
                    continue
                dist = Dist(**param)
                if not dist.has_rsample:
                    continue
                sample = dist.rsample()
                self.assertTrue(
                    sample.requires_grad,
                    msg="{} example {}/{}, .rsample() does not require grad".format(
                        Dist.__name__, i + 1, len(params)
                    ),
                )

    def test_enumerate_support_type(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                try:
                    self.assertIsInstance(
                        dist.sample(),
                        type(dist.enumerate_support()),
                        msg=(
                            "{} example {}/{}, return type mismatch between "
                            + "sample and enumerate_support."
                        ).format(Dist.__name__, i + 1, len(params)),
                    )
                except NotImplementedError:
                    pass

    def test_distribution_expand(self):
        shapes = [torch.Size(), torch.Size((2,)), torch.Size((2, 1))]
        for Dist, params in EXAMPLES:
            for param in params:
                for shape in shapes:
                    d = Dist(**param)
                    expanded_shape = shape + d.batch_shape
                    original_shape = d.batch_shape + d.event_shape
                    expected_shape = shape + original_shape
                    expanded = d.expand(batch_shape=list(expanded_shape))
                    sample = expanded.sample()
                    actual_shape = expanded.sample().shape
                    self.assertEqual(expanded.__class__, d.__class__)
                    self.assertEqual(d.sample().shape, original_shape)
                    self.assertEqual(expanded.log_prob(sample), d.log_prob(sample))
                    self.assertEqual(actual_shape, expected_shape)
                    self.assertEqual(expanded.batch_shape, expanded_shape)
                    try:
                        self.assertEqual(
                            expanded.mean,
                            d.mean.expand(expanded_shape + d.event_shape),
                            allow_inf=True,
                        )
                        self.assertEqual(
                            expanded.variance,
                            d.variance.expand(expanded_shape + d.event_shape),
                            allow_inf=True,
                        )
                    except NotImplementedError:
                        pass

    def test_distribution_subclass_expand(self):
        expand_by = torch.Size((2,))
        for Dist, params in EXAMPLES:

            class SubClass(Dist):
                pass

            for param in params:
                d = SubClass(**param)
                expanded_shape = expand_by + d.batch_shape
                original_shape = d.batch_shape + d.event_shape
                expected_shape = expand_by + original_shape
                expanded = d.expand(batch_shape=expanded_shape)
                sample = expanded.sample()
                actual_shape = expanded.sample().shape
                self.assertEqual(expanded.__class__, d.__class__)
                self.assertEqual(d.sample().shape, original_shape)
                self.assertEqual(expanded.log_prob(sample), d.log_prob(sample))
                self.assertEqual(actual_shape, expected_shape)

    def test_independent_shape(self):
        for Dist, params in EXAMPLES:
            for param in params:
                base_dist = Dist(**param)
                x = base_dist.sample()
                base_log_prob_shape = base_dist.log_prob(x).shape
                for reinterpreted_batch_ndims in range(len(base_dist.batch_shape) + 1):
                    indep_dist = Independent(base_dist, reinterpreted_batch_ndims)
                    indep_log_prob_shape = base_log_prob_shape[
                        : len(base_log_prob_shape) - reinterpreted_batch_ndims
                    ]
                    self.assertEqual(indep_dist.log_prob(x).shape, indep_log_prob_shape)
                    self.assertEqual(
                        indep_dist.sample().shape, base_dist.sample().shape
                    )
                    self.assertEqual(indep_dist.has_rsample, base_dist.has_rsample)
                    if indep_dist.has_rsample:
                        self.assertEqual(
                            indep_dist.sample().shape, base_dist.sample().shape
                        )
                    try:
                        self.assertEqual(
                            indep_dist.enumerate_support().shape,
                            base_dist.enumerate_support().shape,
                        )
                        self.assertEqual(indep_dist.mean.shape, base_dist.mean.shape)
                    except NotImplementedError:
                        pass
                    try:
                        self.assertEqual(
                            indep_dist.variance.shape, base_dist.variance.shape
                        )
                    except NotImplementedError:
                        pass
                    try:
                        self.assertEqual(
                            indep_dist.entropy().shape, indep_log_prob_shape
                        )
                    except NotImplementedError:
                        pass

    def test_independent_expand(self):
        for Dist, params in EXAMPLES:
            for param in params:
                base_dist = Dist(**param)
                for reinterpreted_batch_ndims in range(len(base_dist.batch_shape) + 1):
                    for s in [torch.Size(), torch.Size((2,)), torch.Size((2, 3))]:
                        indep_dist = Independent(base_dist, reinterpreted_batch_ndims)
                        expanded_shape = s + indep_dist.batch_shape
                        expanded = indep_dist.expand(expanded_shape)
                        expanded_sample = expanded.sample()
                        expected_shape = expanded_shape + indep_dist.event_shape
                        self.assertEqual(expanded_sample.shape, expected_shape)
                        self.assertEqual(
                            expanded.log_prob(expanded_sample),
                            indep_dist.log_prob(expanded_sample),
                        )
                        self.assertEqual(expanded.event_shape, indep_dist.event_shape)
                        self.assertEqual(expanded.batch_shape, expanded_shape)

    def test_cdf_icdf_inverse(self):
        # Tests the invertibility property on the distributions
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                samples = dist.sample(sample_shape=(20,))
                try:
                    cdf = dist.cdf(samples)
                    actual = dist.icdf(cdf)
                except NotImplementedError:
                    continue
                rel_error = torch.abs(actual - samples) / (1e-10 + torch.abs(samples))
                self.assertLess(
                    rel_error.max(),
                    1e-4,
                    msg="\n".join(
                        [
                            "{} example {}/{}, icdf(cdf(x)) != x".format(
                                Dist.__name__, i + 1, len(params)
                            ),
                            "x = {}".format(samples),
                            "cdf(x) = {}".format(cdf),
                            "icdf(cdf(x)) = {}".format(actual),
                        ]
                    ),
                )

    def test_cdf_log_prob(self):
        # Tests if the differentiation of the CDF gives the PDF at a given value
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                samples = dist.sample().clone().detach()
                if samples.dtype.is_floating_point:
                    samples.requires_grad_()
                try:
                    cdfs = dist.cdf(samples)
                    pdfs = dist.log_prob(samples).exp()
                except NotImplementedError:
                    continue
                cdfs_derivative = grad(cdfs.sum(), [samples])[
                    0
                ]  # this should not be wrapped in torch.abs()
                self.assertEqual(
                    cdfs_derivative,
                    pdfs,
                    prec=0.2,
                    message="\n".join(
                        [
                            "{} example {}/{}, d(cdf)/dx != pdf(x)".format(
                                Dist.__name__, i + 1, len(params)
                            ),
                            "x = {}".format(samples),
                            "cdf = {}".format(cdfs),
                            "pdf = {}".format(pdfs),
                            "grad(cdf) = {}".format(cdfs_derivative),
                        ]
                    ),
                )

    def test_entropy_monte_carlo(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                # use double precision for better numerical stability
                dist = Dist(**{k: v.double() for k, v in param.items()})
                try:
                    actual = dist.entropy()
                except NotImplementedError:
                    continue
                # use a lot of samples for better MC approximation
                x = dist.sample(sample_shape=(120000,))
                expected = -dist.log_prob(
                    x.clamp_max(1 - 2 * torch.finfo(x.dtype).eps)
                ).mean(0)
                ignore = expected == inf
                expected[ignore] = actual[ignore]
                self.assertEqual(
                    actual,
                    expected,
                    prec=0.2,
                    message="\n".join(
                        [
                            "{} example {}/{}, incorrect .entropy().".format(
                                Dist.__name__, i + 1, len(params)
                            ),
                            "Expected (monte carlo) {}".format(expected),
                            "Actual (analytic) {}".format(actual),
                            "max error = {}".format(torch.abs(actual - expected).max()),
                        ]
                    ),
                )

    def test_params_contains(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                for name, value in param.items():
                    if isinstance(value, Number):
                        value = torch.tensor([value])
                    try:
                        constraint = dist.arg_constraints[name]
                    except KeyError:
                        continue  # ignore optional parameters

                    if is_dependent(constraint):
                        continue

                    message = "{} example {}/{} parameter {} = {}".format(
                        Dist.__name__, i + 1, len(params), name, value
                    )
                    self.assertTrue(constraint.check(value).all(), msg=message)

    def test_support_contains(self):
        for Dist, params in EXAMPLES:
            self.assertIsInstance(Dist.support, Constraint)
            for i, param in enumerate(params):
                dist = Dist(**param)
                value = dist.sample()
                constraint = dist.support
                message = "{} example {}/{} sample = {}".format(
                    Dist.__name__, i + 1, len(params), value
                )
                self.assertTrue(constraint.check(value).all(), msg=message)


class TestDistributionShapes(BotorchTestCase, TestCase):
    def setUp(self):
        super().setUp()
        self.scalar_sample = 1
        self.tensor_sample_1 = torch.ones(3, 2)
        self.tensor_sample_2 = torch.ones(3, 2, 3)
        Distribution.set_default_validate_args(True)

    def tearDown(self):
        super().tearDown()
        Distribution.set_default_validate_args(False)

    def test_kumaraswamy_shape_scalar_params(self):
        kumaraswamy = Kumaraswamy(1, 1)
        self.assertEqual(kumaraswamy._batch_shape, torch.Size())
        self.assertEqual(kumaraswamy._event_shape, torch.Size())
        self.assertEqual(kumaraswamy.sample().size(), torch.Size())
        self.assertEqual(kumaraswamy.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(
            kumaraswamy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            kumaraswamy.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_entropy_shape(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(validate_args=False, **param)
                try:
                    actual_shape = dist.entropy().size()
                    expected_shape = (
                        dist.batch_shape if dist.batch_shape else torch.Size()
                    )
                    message = (
                        f"{Dist.__name__} example {i + 1}/{len(params)}, "
                        f"shape mismatch. expected {expected_shape}, "
                        f"actual {actual_shape}"
                    )
                    self.assertEqual(actual_shape, expected_shape, message=message)
                except NotImplementedError:
                    continue
