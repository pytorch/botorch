#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from copy import deepcopy

import torch
from botorch.models.transforms.outcome import (
    ChainedOutcomeTransform,
    Log,
    OutcomeTransform,
    Standardize,
)
from botorch.models.transforms.utils import (
    norm_to_lognorm_mean,
    norm_to_lognorm_variance,
)
from botorch.posteriors import GPyTorchPosterior, TransformedPosterior
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.lazy import NonLazyTensor


def _get_test_posterior(shape, device, dtype, interleaved=True, lazy=False):
    mean = torch.rand(shape, device=device, dtype=dtype)
    n_covar = shape[-2:].numel()
    diag = torch.rand(shape, device=device, dtype=dtype)
    diag = diag.view(*diag.shape[:-2], n_covar)
    a = torch.rand(*shape[:-2], n_covar, n_covar, device=device, dtype=dtype)
    covar = a @ a.transpose(-1, -2) + torch.diag_embed(diag)
    if lazy:
        covar = NonLazyTensor(covar)
    if shape[-1] == 1:
        mvn = MultivariateNormal(mean.squeeze(-1), covar)
    else:
        mvn = MultitaskMultivariateNormal(mean, covar, interleaved=interleaved)
    return GPyTorchPosterior(mvn)


class NotSoAbstractOutcomeTransform(OutcomeTransform):
    def forward(self, Y, Yvar):
        pass


class TestOutcomeTransforms(BotorchTestCase):
    def test_abstract_base_outcome_transform(self):
        with self.assertRaises(TypeError):
            OutcomeTransform()
        oct = NotSoAbstractOutcomeTransform()
        with self.assertRaises(NotImplementedError):
            oct.untransform(None, None)
        with self.assertRaises(NotImplementedError):
            oct.untransform_posterior(None)

    def test_standardize(self):

        # test error on incompatible dim
        tf = Standardize(m=1)
        with self.assertRaises(RuntimeError):
            tf(torch.zeros(3, 2, device=self.device), None)
        # test error on incompatible batch shape
        with self.assertRaises(RuntimeError):
            tf(torch.zeros(2, 3, 1, device=self.device), None)

        ms = (1, 2)
        batch_shapes = (torch.Size(), torch.Size([2]))
        dtypes = (torch.float, torch.double)

        # test transform, untransform, untransform_posterior
        for m, batch_shape, dtype in itertools.product(ms, batch_shapes, dtypes):

            # test init
            tf = Standardize(m=m, batch_shape=batch_shape)
            self.assertTrue(tf.training)
            self.assertEqual(tf._m, m)
            self.assertIsNone(tf._outputs)
            self.assertEqual(tf._batch_shape, batch_shape)
            self.assertEqual(tf._min_stdv, 1e-8)

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertTrue(torch.all(Y_tf.mean(dim=-2).abs() < 1e-4))
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # with observation noise
            tf = Standardize(m=m, batch_shape=batch_shape)
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            Y_tf, Yvar_tf = tf(Y, Yvar)
            self.assertTrue(tf.training)
            self.assertTrue(torch.all(Y_tf.mean(dim=-2).abs() < 1e-4))
            Yvar_tf_expected = Yvar / Y.std(dim=-2, keepdim=True) ** 2
            self.assertTrue(torch.allclose(Yvar_tf, Yvar_tf_expected))
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            torch.allclose(Yvar_utf, Yvar)

            # untransform_posterior
            for interleaved, lazy in itertools.product((True, False), (True, False)):
                if m == 1 and interleaved:  # interleave has no meaning for m=1
                    continue
                shape = batch_shape + torch.Size([3, m])
                posterior = _get_test_posterior(
                    shape,
                    device=self.device,
                    dtype=dtype,
                    interleaved=interleaved,
                    lazy=lazy,
                )
                p_utf = tf.untransform_posterior(posterior)
                self.assertEqual(p_utf.device.type, self.device.type)
                self.assertTrue(p_utf.dtype == dtype)
                mean_expected = tf.means + tf.stdvs * posterior.mean
                variance_expected = tf.stdvs ** 2 * posterior.variance
                self.assertTrue(torch.allclose(p_utf.mean, mean_expected))
                self.assertTrue(torch.allclose(p_utf.variance, variance_expected))
                samples = p_utf.rsample()
                self.assertEqual(samples.shape, torch.Size([1]) + shape)
                samples = p_utf.rsample(sample_shape=torch.Size([4]))
                self.assertEqual(samples.shape, torch.Size([4]) + shape)
                samples2 = p_utf.rsample(sample_shape=torch.Size([4, 2]))
                self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)
                # TODO: Test expected covar (both interleaved and non-interleaved)

            # untransform_posterior for non-GPyTorch posterior
            posterior2 = TransformedPosterior(
                posterior=posterior,
                sample_transform=lambda s: s,
                mean_transform=lambda m, v: m,
                variance_transform=lambda m, v: v,
            )
            p_utf2 = tf.untransform_posterior(posterior2)
            self.assertEqual(p_utf2.device.type, self.device.type)
            self.assertTrue(p_utf2.dtype == dtype)
            mean_expected = tf.means + tf.stdvs * posterior.mean
            variance_expected = tf.stdvs ** 2 * posterior.variance
            self.assertTrue(torch.allclose(p_utf2.mean, mean_expected))
            self.assertTrue(torch.allclose(p_utf2.variance, variance_expected))
            # TODO: Test expected covar (both interleaved and non-interleaved)
            samples = p_utf2.rsample()
            self.assertEqual(samples.shape, torch.Size([1]) + shape)
            samples = p_utf2.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4]) + shape)
            samples2 = p_utf2.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)

            # test error on incompatible output dimension
            tf_big = Standardize(m=4).eval()
            with self.assertRaises(RuntimeError) as e:
                tf_big.untransform_posterior(posterior2)
                self.assertTrue("Incompatible output dimensions" in str(e))

        # test subset outcomes
        for batch_shape, dtype in itertools.product(batch_shapes, dtypes):

            m = 2
            outputs = [-1]

            # test init
            tf = Standardize(m=m, outputs=outputs, batch_shape=batch_shape)
            self.assertTrue(tf.training)
            self.assertEqual(tf._m, m)
            self.assertEqual(tf._outputs, [1])
            self.assertEqual(tf._batch_shape, batch_shape)
            self.assertEqual(tf._min_stdv, 1e-8)

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            Y_tf_mean = Y_tf.mean(dim=-2)
            self.assertTrue(torch.all(Y_tf_mean[..., 1].abs() < 1e-4))
            self.assertTrue(torch.allclose(Y_tf_mean[..., 0], Y.mean(dim=-2)[..., 0]))
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # with observation noise
            tf = Standardize(m=m, outputs=outputs, batch_shape=batch_shape)
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            Y_tf, Yvar_tf = tf(Y, Yvar)
            self.assertTrue(tf.training)
            Y_tf_mean = Y_tf.mean(dim=-2)
            self.assertTrue(torch.all(Y_tf_mean[..., 1].abs() < 1e-4))
            self.assertTrue(torch.allclose(Y_tf_mean[..., 0], Y.mean(dim=-2)[..., 0]))
            Yvar_tf_expected = Yvar / Y.std(dim=-2, keepdim=True) ** 2
            self.assertTrue(torch.allclose(Yvar_tf[..., 1], Yvar_tf_expected[..., 1]))
            self.assertTrue(torch.allclose(Yvar_tf[..., 0], Yvar[..., 0]))
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            torch.allclose(Yvar_utf, Yvar)

            # error on untransform_posterior
            with self.assertRaises(NotImplementedError):
                tf.untransform_posterior(None)

    def test_log(self):

        ms = (1, 2)
        batch_shapes = (torch.Size(), torch.Size([2]))
        dtypes = (torch.float, torch.double)

        # test transform and untransform
        for m, batch_shape, dtype in itertools.product(ms, batch_shapes, dtypes):

            # test init
            tf = Log()
            self.assertTrue(tf.training)
            self.assertIsNone(tf._outputs)

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertTrue(torch.allclose(Y_tf, torch.log(Y)))
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # test error if observation noise present
            tf = Log()
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            with self.assertRaises(NotImplementedError):
                tf(Y, Yvar)
            tf.eval()
            with self.assertRaises(NotImplementedError):
                tf.untransform(Y, Yvar)

            # untransform_posterior
            tf = Log()
            Y_tf, Yvar_tf = tf(Y, None)
            tf.eval()
            shape = batch_shape + torch.Size([3, m])
            posterior = _get_test_posterior(shape, device=self.device, dtype=dtype)
            p_utf = tf.untransform_posterior(posterior)
            self.assertIsInstance(p_utf, TransformedPosterior)
            self.assertEqual(p_utf.device.type, self.device.type)
            self.assertTrue(p_utf.dtype == dtype)
            self.assertTrue(p_utf._sample_transform == torch.exp)
            mean_expected = norm_to_lognorm_mean(posterior.mean, posterior.variance)
            variance_expected = norm_to_lognorm_variance(
                posterior.mean, posterior.variance
            )
            self.assertTrue(torch.allclose(p_utf.mean, mean_expected))
            self.assertTrue(torch.allclose(p_utf.variance, variance_expected))
            samples = p_utf.rsample()
            self.assertEqual(samples.shape, torch.Size([1]) + shape)
            samples = p_utf.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4]) + shape)
            samples2 = p_utf.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)

        # test subset outcomes
        for batch_shape, dtype in itertools.product(batch_shapes, dtypes):

            m = 2
            outputs = [-1]

            # test init
            tf = Log(outputs=outputs)
            self.assertTrue(tf.training)
            # cannot normalize indices b/c we don't know dimension yet
            self.assertEqual(tf._outputs, [-1])

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertTrue(torch.allclose(Y_tf[..., 1], torch.log(Y[..., 1])))
            self.assertTrue(torch.allclose(Y_tf[..., 0], Y[..., 0]))
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # with observation noise
            tf = Log(outputs=outputs)
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            with self.assertRaises(NotImplementedError):
                tf(Y, Yvar)

            # error on untransform_posterior
            with self.assertRaises(NotImplementedError):
                tf.untransform_posterior(None)

    def test_chained_outcome_transform(self):

        ms = (1, 2)
        batch_shapes = (torch.Size(), torch.Size([2]))
        dtypes = (torch.float, torch.double)

        # test transform and untransform
        for m, batch_shape, dtype in itertools.product(ms, batch_shapes, dtypes):

            # test init
            tf1 = Log()
            tf2 = Standardize(m=m, batch_shape=batch_shape)
            tf = ChainedOutcomeTransform(b=tf1, a=tf2)
            self.assertTrue(tf.training)
            self.assertEqual(list(tf.keys()), ["b", "a"])
            self.assertEqual(tf["b"], tf1)
            self.assertEqual(tf["a"], tf2)

            # make copies for validation below
            tf1_, tf2_ = deepcopy(tf1), deepcopy(tf2)

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            Y_tf_, Yvar_tf_ = tf2_(*tf1_(Y, None))
            self.assertTrue(tf.training)
            self.assertIsNone(Yvar_tf_)
            self.assertTrue(torch.allclose(Y_tf, Y_tf_))
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # test error if observation noise present
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            with self.assertRaises(NotImplementedError):
                tf(Y, Yvar)

            # untransform_posterior
            tf1 = Log()
            tf2 = Standardize(m=m, batch_shape=batch_shape)
            tf = ChainedOutcomeTransform(log=tf1, standardize=tf2)
            Y_tf, Yvar_tf = tf(Y, None)
            tf.eval()
            shape = batch_shape + torch.Size([3, m])
            posterior = _get_test_posterior(shape, device=self.device, dtype=dtype)
            p_utf = tf.untransform_posterior(posterior)
            self.assertIsInstance(p_utf, TransformedPosterior)
            self.assertEqual(p_utf.device.type, self.device.type)
            self.assertTrue(p_utf.dtype == dtype)
            samples = p_utf.rsample()
            self.assertEqual(samples.shape, torch.Size([1]) + shape)
            samples = p_utf.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4]) + shape)
            samples2 = p_utf.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)

        # test subset outcomes
        for batch_shape, dtype in itertools.product(batch_shapes, dtypes):

            m = 2
            outputs = [-1]

            # test init
            tf1 = Log(outputs=outputs)
            tf2 = Standardize(m=m, outputs=outputs, batch_shape=batch_shape)
            tf = ChainedOutcomeTransform(log=tf1, standardize=tf2)
            self.assertTrue(tf.training)
            self.assertEqual(sorted(tf.keys()), ["log", "standardize"])
            self.assertEqual(tf["log"], tf1)
            self.assertEqual(tf["standardize"], tf2)
            self.assertEqual(tf["log"]._outputs, [-1])  # don't know dimension yet
            self.assertEqual(tf["standardize"]._outputs, [1])

            # make copies for validation below
            tf1_, tf2_ = deepcopy(tf1), deepcopy(tf2)

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            Y_tf_, Yvar_tf_ = tf2_(*tf1_(Y, None))
            self.assertTrue(tf.training)
            self.assertIsNone(Yvar_tf_)
            self.assertTrue(torch.allclose(Y_tf, Y_tf_))
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # with observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            with self.assertRaises(NotImplementedError):
                tf(Y, Yvar)

            # error on untransform_posterior
            with self.assertRaises(NotImplementedError):
                tf.untransform_posterior(None)
