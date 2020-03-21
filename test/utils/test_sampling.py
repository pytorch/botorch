#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch import settings
from botorch.exceptions.warnings import SamplingWarning
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.sampling import (
    batched_multinomial,
    construct_base_samples,
    construct_base_samples_from_posterior,
    manual_seed,
    sample_hypersphere,
    sample_simplex,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from torch.quasirandom import SobolEngine


class TestManualSeed(BotorchTestCase):
    def test_manual_seed(self):
        initial_state = torch.random.get_rng_state()
        with manual_seed():
            self.assertTrue(torch.all(torch.random.get_rng_state() == initial_state))
        with manual_seed(1234):
            self.assertFalse(torch.all(torch.random.get_rng_state() == initial_state))
        self.assertTrue(torch.all(torch.random.get_rng_state() == initial_state))


class TestConstructBaseSamples(BotorchTestCase):
    def test_construct_base_samples(self):
        test_shapes = [
            {"batch": [2], "output": [4, 3], "sample": [5]},
            {"batch": [1], "output": [5, 3], "sample": [5, 6]},
            {"batch": [2, 3], "output": [2, 3], "sample": [5]},
        ]
        for tshape, qmc, seed, dtype in itertools.product(
            test_shapes, (False, True), (None, 1234), (torch.float, torch.double)
        ):
            batch_shape = torch.Size(tshape["batch"])
            output_shape = torch.Size(tshape["output"])
            sample_shape = torch.Size(tshape["sample"])
            expected_shape = sample_shape + batch_shape + output_shape
            samples = construct_base_samples(
                batch_shape=batch_shape,
                output_shape=output_shape,
                sample_shape=sample_shape,
                qmc=qmc,
                seed=seed,
                device=self.device,
                dtype=dtype,
            )
            self.assertEqual(samples.shape, expected_shape)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertEqual(samples.dtype, dtype)
        # check that warning is issued if dimensionality is too large
        with warnings.catch_warnings(record=True) as w, settings.debug(True):
            construct_base_samples(
                batch_shape=torch.Size(),
                output_shape=torch.Size([200, 6]),
                sample_shape=torch.Size([1]),
                qmc=True,
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, SamplingWarning))
            exp_str = f"maximum supported by qmc ({SobolEngine.MAXDIM})"
            self.assertTrue(exp_str in str(w[-1].message))

    def test_construct_base_samples_from_posterior(self):  # noqa: C901
        for dtype in (torch.float, torch.double):
            # single-output
            mean = torch.zeros(2, device=self.device, dtype=dtype)
            cov = torch.eye(2, device=self.device, dtype=dtype)
            mvn = MultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn=mvn)
            for sample_shape, qmc, seed in itertools.product(
                (torch.Size([5]), torch.Size([5, 3])), (False, True), (None, 1234)
            ):
                expected_shape = sample_shape + torch.Size([2, 1])
                samples = construct_base_samples_from_posterior(
                    posterior=posterior, sample_shape=sample_shape, qmc=qmc, seed=seed
                )
                self.assertEqual(samples.shape, expected_shape)
                self.assertEqual(samples.device.type, self.device.type)
                self.assertEqual(samples.dtype, dtype)
            # single-output, batch mode
            mean = torch.zeros(2, 2, device=self.device, dtype=dtype)
            cov = torch.eye(2, device=self.device, dtype=dtype).expand(2, 2, 2)
            mvn = MultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn=mvn)
            for sample_shape, qmc, seed, collapse_batch_dims in itertools.product(
                (torch.Size([5]), torch.Size([5, 3])),
                (False, True),
                (None, 1234),
                (False, True),
            ):
                if collapse_batch_dims:
                    expected_shape = sample_shape + torch.Size([1, 2, 1])
                else:
                    expected_shape = sample_shape + torch.Size([2, 2, 1])
                samples = construct_base_samples_from_posterior(
                    posterior=posterior,
                    sample_shape=sample_shape,
                    qmc=qmc,
                    collapse_batch_dims=collapse_batch_dims,
                    seed=seed,
                )
                self.assertEqual(samples.shape, expected_shape)
                self.assertEqual(samples.device.type, self.device.type)
                self.assertEqual(samples.dtype, dtype)
            # multi-output
            mean = torch.zeros(2, 2, device=self.device, dtype=dtype)
            cov = torch.eye(4, device=self.device, dtype=dtype)
            mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn=mtmvn)
            for sample_shape, qmc, seed in itertools.product(
                (torch.Size([5]), torch.Size([5, 3])), (False, True), (None, 1234)
            ):
                expected_shape = sample_shape + torch.Size([2, 2])
                samples = construct_base_samples_from_posterior(
                    posterior=posterior, sample_shape=sample_shape, qmc=qmc, seed=seed
                )
                self.assertEqual(samples.shape, expected_shape)
                self.assertEqual(samples.device.type, self.device.type)
                self.assertEqual(samples.dtype, dtype)
            # multi-output, batch mode
            mean = torch.zeros(2, 2, 2, device=self.device, dtype=dtype)
            cov = torch.eye(4, device=self.device, dtype=dtype).expand(2, 4, 4)
            mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn=mtmvn)
            for sample_shape, qmc, seed, collapse_batch_dims in itertools.product(
                (torch.Size([5]), torch.Size([5, 3])),
                (False, True),
                (None, 1234),
                (False, True),
            ):
                if collapse_batch_dims:
                    expected_shape = sample_shape + torch.Size([1, 2, 2])
                else:
                    expected_shape = sample_shape + torch.Size([2, 2, 2])
                samples = construct_base_samples_from_posterior(
                    posterior=posterior,
                    sample_shape=sample_shape,
                    qmc=qmc,
                    collapse_batch_dims=collapse_batch_dims,
                    seed=seed,
                )
                self.assertEqual(samples.shape, expected_shape)
                self.assertEqual(samples.device.type, self.device.type)
                self.assertEqual(samples.dtype, dtype)


class TestSampleUtils(BotorchTestCase):
    def test_sample_simplex(self):
        for d, n, qmc, seed, dtype in itertools.product(
            (1, 2, 3), (2, 5), (False, True), (None, 1234), (torch.float, torch.double)
        ):
            samples = sample_simplex(
                d=d, n=n, qmc=qmc, seed=seed, device=self.device, dtype=dtype
            )
            self.assertEqual(samples.shape, torch.Size([n, d]))
            self.assertTrue(torch.all(samples >= 0))
            self.assertTrue(torch.all(samples <= 1))
            self.assertTrue(torch.max((samples.sum(dim=-1) - 1).abs()) < 1e-5)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertEqual(samples.dtype, dtype)

    def test_sample_hypersphere(self):
        for d, n, qmc, seed, dtype in itertools.product(
            (1, 2, 3), (2, 5), (False, True), (None, 1234), (torch.float, torch.double)
        ):
            samples = sample_hypersphere(
                d=d, n=n, qmc=qmc, seed=seed, device=self.device, dtype=dtype
            )
            self.assertEqual(samples.shape, torch.Size([n, d]))
            self.assertTrue(torch.max((samples.pow(2).sum(dim=-1) - 1).abs()) < 1e-5)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertEqual(samples.dtype, dtype)

    def test_batched_multinomial(self):
        num_categories = 5
        num_samples = 4
        Trulse = (True, False)
        for batch_shape, dtype, replacement, use_gen, use_out in itertools.product(
            ([], [3], [2, 3]), (torch.float, torch.double), Trulse, Trulse, Trulse
        ):
            weights = torch.rand(*batch_shape, num_categories, dtype=dtype)
            out = None
            if use_out:
                out = torch.empty(*batch_shape, num_samples, dtype=torch.long)
            samples = batched_multinomial(
                weights,
                num_samples,
                replacement=replacement,
                generator=torch.Generator() if use_gen else None,
                out=out,
            )
            self.assertEqual(samples.shape, torch.Size([*batch_shape, num_samples]))
            if use_out:
                self.assertTrue(torch.equal(samples, out))
            if not replacement:
                for s in samples.view(-1, num_samples):
                    self.assertTrue(torch.unique(s).size(0), num_samples)
