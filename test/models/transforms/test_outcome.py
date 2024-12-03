#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from copy import deepcopy

import torch
from botorch.models.transforms.outcome import (
    _nanmax,
    _nanmin,
    Bilog,
    ChainedOutcomeTransform,
    HalfRankTransform,
    InfeasibleTransform,
    Log,
    LogWarperTransform,
    OutcomeTransform,
    Power,
    Standardize,
)
from botorch.models.transforms.utils import (
    norm_to_lognorm_mean,
    norm_to_lognorm_variance,
)
from botorch.posteriors import GPyTorchPosterior, TransformedPosterior
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from linear_operator.operators import (
    BlockDiagLinearOperator,
    DenseLinearOperator,
    DiagLinearOperator,
)


def _get_test_posterior(shape, device, dtype, interleaved=True, lazy=False):
    mean = torch.rand(shape, device=device, dtype=dtype)
    n_covar = shape[-2:].numel()
    diag = torch.rand(shape, device=device, dtype=dtype)
    diag = diag.view(*diag.shape[:-2], n_covar)
    a = torch.rand(*shape[:-2], n_covar, n_covar, device=device, dtype=dtype)
    covar = a @ a.transpose(-1, -2) + torch.diag_embed(diag)
    if lazy:
        covar = DenseLinearOperator(covar)
    if shape[-1] == 1:
        mvn = MultivariateNormal(mean.squeeze(-1), covar)
    else:
        mvn = MultitaskMultivariateNormal(mean, covar, interleaved=interleaved)
    return GPyTorchPosterior(mvn)


class NotSoAbstractOutcomeTransform(OutcomeTransform):
    def forward(self, Y, Yvar):
        pass


class TestNanMax(BotorchTestCase):
    def test_nanmax_basic(self):
        tensor = torch.tensor([1.0, float("nan"), 3.0, 2.0])
        result = _nanmax(tensor)
        expected = torch.tensor(3.0)
        self.assertEqual(result, expected)

    def test_nanmax_with_dim(self):
        tensor = torch.tensor([[1.0, float("nan")], [3.0, 2.0]])
        result = _nanmax(tensor, dim=1)
        expected = torch.tensor([1.0, 3.0])
        self.assertTrue(torch.equal(result.values, expected))

    def test_nanmax_with_keepdim(self):
        tensor = torch.tensor([[1.0, float("nan")], [3.0, 2.0]])
        result = _nanmax(tensor, dim=1, keepdim=True)
        expected = torch.tensor([[1.0], [3.0]])
        self.assertTrue(torch.equal(result.values, expected))

    def test_nanmax_all_nan(self):
        tensor = torch.tensor([float("nan"), float("nan")])
        result = _nanmax(tensor)
        expected = torch.tensor(torch.finfo(tensor.dtype).min)
        self.assertEqual(result, expected)

    def test_nanmax_no_nan(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = _nanmax(tensor)
        expected = torch.tensor(3.0)
        self.assertEqual(result, expected)


class TestNanMin(BotorchTestCase):
    def test_nanmin_basic(self):
        tensor = torch.tensor([1.0, float("nan"), 3.0, 2.0])
        result = _nanmin(tensor)
        expected = torch.tensor(1.0)
        self.assertEqual(result, expected)

    def test_nanmin_with_dim(self):
        tensor = torch.tensor([[1.0, float("nan")], [3.0, 2.0]])
        result = _nanmin(tensor, dim=1)
        expected = torch.tensor([1.0, 2.0])
        self.assertTrue(torch.equal(result.values, expected))

    def test_nanmin_with_keepdim(self):
        tensor = torch.tensor([[1.0, float("nan")], [3.0, 2.0]])
        result = _nanmin(tensor, dim=1, keepdim=True)
        expected = torch.tensor([[1.0], [2.0]])
        self.assertTrue(torch.equal(result.values, expected))

    def test_nanmin_all_nan(self):
        tensor = torch.tensor([float("nan"), float("nan")])
        result = _nanmin(tensor)
        expected = torch.tensor(torch.finfo(tensor.dtype).max)
        self.assertEqual(result, expected)

    def test_nanmin_no_nan(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = _nanmin(tensor)
        expected = torch.tensor(1.0)
        self.assertEqual(result, expected)


class TestOutcomeTransforms(BotorchTestCase):
    def test_abstract_base_outcome_transform(self):
        with self.assertRaises(TypeError):
            OutcomeTransform()
        oct = NotSoAbstractOutcomeTransform()
        with self.assertRaises(NotImplementedError):
            oct.subset_output(None)
        with self.assertRaises(NotImplementedError):
            oct.untransform(None, None)
        with self.assertRaises(NotImplementedError):
            oct.untransform_posterior(None)

    def test_standardize_raises_when_mean_not_set(self) -> None:
        posterior = _get_test_posterior(
            shape=torch.Size([1, 1]), device=self.device, dtype=torch.float64
        )
        for transform in [
            Standardize(m=1),
            ChainedOutcomeTransform(
                chained=ChainedOutcomeTransform(stand=Standardize(m=1))
            ),
        ]:
            with self.assertRaises(
                RuntimeError,
                msg="`Standardize` transforms must be called on outcome data "
                "(e.g. `transform(Y)`) before calling `untransform_posterior`, since "
                "means and standard deviations need to be computed.",
            ):
                transform.untransform_posterior(posterior)

            new_tf = transform.subset_output([0])
            assert isinstance(new_tf, type(transform))

            y = torch.arange(3, device=self.device, dtype=torch.float64)
            with self.assertRaises(
                RuntimeError,
                msg="`Standardize` transforms must be called on outcome data "
                "(e.g. `transform(Y)`) before calling `untransform`, since "
                "means and standard deviations need to be computed.",
            ):
                transform.untransform(y)

    def test_is_linear(self) -> None:
        posterior = _get_test_posterior(
            shape=torch.Size([1, 1]), device=self.device, dtype=torch.float64
        )
        y = torch.arange(2, dtype=torch.float64, device=self.device)[:, None]
        standardize_tf = Standardize(m=1)
        standardize_tf(y)

        for transform in [
            standardize_tf,
            Power(power=0.5),
            Log(),
            ChainedOutcomeTransform(
                chained=ChainedOutcomeTransform(stand=standardize_tf)
            ),
            ChainedOutcomeTransform(log=Log()),
        ]:
            posterior_is_gpt = isinstance(
                transform.untransform_posterior(posterior), GPyTorchPosterior
            )
            self.assertEqual(posterior_is_gpt, transform._is_linear)

    def test_standardize_raises_when_no_observations(self) -> None:
        tf = Standardize(m=1)
        with self.assertRaisesRegex(
            ValueError, "Can't standardize with no observations."
        ):
            tf(torch.zeros(0, 1, device=self.device), None)

    def test_standardize(self) -> None:
        # test error on incompatible dim
        tf = Standardize(m=1)
        with self.assertRaisesRegex(
            RuntimeError, r"Wrong output dimension. Y.size\(-1\) is 2; expected 1."
        ):
            tf(torch.zeros(3, 2, device=self.device), None)
        # test error on incompatible batch shape
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected Y.shape\[:-2\] to be torch.Size\(\[\]\), matching the "
            "`batch_shape` argument to `Standardize`, but got "
            r"Y.shape\[:-2\]=torch.Size\(\[2\]\).",
        ):
            tf(torch.zeros(2, 3, 1, device=self.device), None)

        ms = (1, 2)
        batch_shapes = (torch.Size(), torch.Size([2]))
        dtypes = (torch.float, torch.double)
        ns = [1, 3]

        # test transform, untransform, untransform_posterior
        for m, batch_shape, dtype, n in itertools.product(ms, batch_shapes, dtypes, ns):
            # test init
            tf = Standardize(m=m, batch_shape=batch_shape)
            self.assertTrue(tf.training)
            self.assertEqual(tf._m, m)
            self.assertIsNone(tf._outputs)
            self.assertEqual(tf._batch_shape, batch_shape)
            self.assertEqual(tf._min_stdv, 1e-8)

            # no observation noise
            with torch.random.fork_rng():
                torch.manual_seed(0)
                Y = torch.rand(*batch_shape, n, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertTrue(torch.all(Y_tf.mean(dim=-2).abs() < 1e-4))
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]])
            self.assertTrue(torch.equal(Y_tf[..., [0]], Y_tf_subset))
            self.assertIsNone(Yvar_tf_subset)
            with self.assertRaises(RuntimeError):
                tf.subset_output(idcs=[0, 1, 2])

            # with observation noise
            tf = Standardize(m=m, batch_shape=batch_shape)
            with torch.random.fork_rng():
                torch.manual_seed(0)
                Y = torch.rand(*batch_shape, n, m, device=self.device, dtype=dtype)
                Yvar = 1e-8 + torch.rand(
                    *batch_shape, n, m, device=self.device, dtype=dtype
                )
            Y_tf, Yvar_tf = tf(Y, Yvar)
            self.assertTrue(tf.training)
            self.assertTrue(torch.all(Y_tf.mean(dim=-2).abs() < 1e-4))
            Yvar_tf_expected = (
                Yvar if n == 1 else Yvar / Y.std(dim=-2, keepdim=True) ** 2
            )
            self.assertAllClose(Yvar_tf, Yvar_tf_expected)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y)
            self.assertAllClose(Yvar_utf, Yvar)

            # untransform_posterior
            for interleaved, lazy in itertools.product((True, False), (True, False)):
                if m == 1 and interleaved:  # interleave has no meaning for m=1
                    continue
                shape = batch_shape + torch.Size([n, m])
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
                variance_expected = tf.stdvs**2 * posterior.variance
                self.assertAllClose(p_utf.mean, mean_expected)
                self.assertAllClose(p_utf.variance, variance_expected)
                samples = p_utf.rsample()
                self.assertEqual(samples.shape, torch.Size([1]) + shape)
                samples = p_utf.rsample(sample_shape=torch.Size([4]))
                self.assertEqual(samples.shape, torch.Size([4]) + shape)
                samples2 = p_utf.rsample(sample_shape=torch.Size([4, 2]))
                self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)
                # TODO: Test expected covar (both interleaved and non-interleaved)

            # Untransform BlockDiagLinearOperator.
            if m > 1:
                base_lcv = DiagLinearOperator(
                    torch.rand(*batch_shape, m, n, device=self.device, dtype=dtype)
                )
                lcv = BlockDiagLinearOperator(base_lcv)
                mvn = MultitaskMultivariateNormal(
                    mean=torch.rand(
                        *batch_shape, n, m, device=self.device, dtype=dtype
                    ),
                    covariance_matrix=lcv,
                    interleaved=False,
                )
                posterior = GPyTorchPosterior(distribution=mvn)
                p_utf = tf.untransform_posterior(posterior)
                self.assertEqual(p_utf.device.type, self.device.type)
                self.assertTrue(p_utf.dtype == dtype)
                mean_expected = tf.means + tf.stdvs * posterior.mean
                variance_expected = tf.stdvs**2 * posterior.variance
                self.assertAllClose(p_utf.mean, mean_expected)
                self.assertAllClose(p_utf.variance, variance_expected)
                self.assertIsInstance(
                    p_utf.distribution.lazy_covariance_matrix, DiagLinearOperator
                )
                samples2 = p_utf.rsample(sample_shape=torch.Size([4, 2]))
                self.assertEqual(
                    samples2.shape,
                    torch.Size([4, 2]) + batch_shape + torch.Size([n, m]),
                )

            # untransform_posterior for non-GPyTorch posterior
            posterior2 = TransformedPosterior(
                posterior=posterior,
                sample_transform=lambda s: s,
                mean_transform=lambda m, v: m,
                variance_transform=lambda m, v: v,
            )
            p_utf2 = tf.untransform_posterior(posterior2)
            self.assertEqual(p_utf2.device.type, self.device.type)
            self.assertEqual(p_utf2.dtype, dtype)
            mean_expected = tf.means + tf.stdvs * posterior.mean
            variance_expected = tf.stdvs**2 * posterior.variance
            self.assertAllClose(p_utf2.mean, mean_expected)
            self.assertAllClose(p_utf2.variance, variance_expected)
            # TODO: Test expected covar (both interleaved and non-interleaved)
            samples = p_utf2.rsample()
            self.assertEqual(samples.shape, torch.Size([1]) + shape)
            samples = p_utf2.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4]) + shape)
            samples2 = p_utf2.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)

            # test error on incompatible output dimension
            # TODO: add a unit test for MTGP posterior once #840 goes in
            tf_big = Standardize(m=4)
            Y = torch.arange(4, device=self.device, dtype=dtype).reshape((1, 4))
            tf_big(Y)
            with self.assertRaisesRegex(
                RuntimeError,
                "Incompatible output dimensions encountered. Transform has output "
                f"dimension {tf_big._m} and posterior has "
                f"{posterior._extended_shape()[-1]}.",
            ):
                tf_big.untransform_posterior(posterior2)

        # test transforming a subset of outcomes
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
            with torch.random.fork_rng():
                torch.manual_seed(0)
                Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            Y_tf_mean = Y_tf.mean(dim=-2)
            self.assertTrue(torch.all(Y_tf_mean[..., 1].abs() < 1e-4))
            self.assertAllClose(Y_tf_mean[..., 0], Y.mean(dim=-2)[..., 0])
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]])
            self.assertTrue(torch.equal(Y_tf[..., [0]], Y_tf_subset))
            self.assertIsNone(Yvar_tf_subset)
            with self.assertRaises(RuntimeError):
                tf.subset_output(idcs=[0, 1, 2])

            # with observation noise
            tf = Standardize(m=m, outputs=outputs, batch_shape=batch_shape)
            with torch.random.fork_rng():
                torch.manual_seed(0)
                Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
                Yvar = 1e-8 + torch.rand(
                    *batch_shape, 3, m, device=self.device, dtype=dtype
                )
            Y_tf, Yvar_tf = tf(Y, Yvar)
            self.assertTrue(tf.training)
            Y_tf_mean = Y_tf.mean(dim=-2)
            self.assertTrue(torch.all(Y_tf_mean[..., 1].abs() < 1e-4))
            self.assertAllClose(Y_tf_mean[..., 0], Y.mean(dim=-2)[..., 0])
            Yvar_tf_expected = Yvar / Y.std(dim=-2, keepdim=True) ** 2
            self.assertAllClose(Yvar_tf[..., 1], Yvar_tf_expected[..., 1])
            self.assertAllClose(Yvar_tf[..., 0], Yvar[..., 0])
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y)
            self.assertAllClose(Yvar_utf, Yvar)

            # error on untransform_posterior
            with self.assertRaises(NotImplementedError):
                tf.untransform_posterior(None)

    def test_standardize_state_dict(self):
        for m in (1, 2):
            with self.subTest(m=2):
                transform = Standardize(m=m)
                self.assertFalse(transform._is_trained)
                self.assertTrue(transform.training)
                Y = torch.rand(2, m)
                transform(Y)
                state_dict = transform.state_dict()
                new_transform = Standardize(m=m)
                self.assertFalse(new_transform._is_trained)
                new_transform.load_state_dict(state_dict)
                self.assertTrue(new_transform._is_trained)

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
            self.assertAllClose(Y_tf, torch.log(Y))
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]])
            self.assertTrue(torch.equal(Y_tf[..., [0]], Y_tf_subset))
            self.assertIsNone(Yvar_tf_subset)

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
            self.assertAllClose(p_utf.mean, mean_expected)
            self.assertAllClose(p_utf.variance, variance_expected)
            samples = p_utf.rsample()
            self.assertEqual(samples.shape, torch.Size([1]) + shape)
            samples = p_utf.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4]) + shape)
            samples2 = p_utf.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)

        # test transforming a subset of outcomes
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
            self.assertAllClose(Y_tf[..., 1], torch.log(Y[..., 1]))
            self.assertAllClose(Y_tf[..., 0], Y[..., 0])
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            with self.assertRaises(NotImplementedError):
                tf_subset = tf.subset_output(idcs=[0])

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

            # test subset_output with positive on subset of outcomes (pos. index)
            tf = Log(outputs=[0])
            Y_tf, Yvar_tf = tf(Y, None)
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]], None)
            self.assertTrue(torch.equal(Y_tf_subset, Y_tf[..., [0]]))
            self.assertIsNone(Yvar_tf_subset)

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
            self.assertAllClose(Y_tf, Y_tf_)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]])
            self.assertTrue(torch.equal(Y_tf[..., [0]], Y_tf_subset))
            self.assertIsNone(Yvar_tf_subset)
            with self.assertRaises(RuntimeError):
                tf.subset_output(idcs=[0, 1, 2])

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

        # test transforming a subset of outcomes
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
            self.assertAllClose(Y_tf, Y_tf_)
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

    def test_power(self, seed=0):
        torch.random.manual_seed(seed)

        ms = (1, 2)
        batch_shapes = (torch.Size(), torch.Size([2]))
        dtypes = (torch.float, torch.double)
        power = 1 / 3

        # test transform and untransform
        for m, batch_shape, dtype in itertools.product(ms, batch_shapes, dtypes):
            # test init
            tf = Power(power=power)
            self.assertTrue(tf.training)
            self.assertIsNone(tf._outputs)

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertAllClose(Y_tf, Y.pow(power))
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]])
            self.assertTrue(torch.equal(Y_tf[..., [0]], Y_tf_subset))
            self.assertIsNone(Yvar_tf_subset)

            # test error if observation noise present
            tf = Power(power=power)
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
            tf = Power(power=power)
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

        # test transforming a subset of outcomes
        for batch_shape, dtype in itertools.product(batch_shapes, dtypes):
            m = 2
            outputs = [-1]

            # test init
            tf = Power(power=power, outputs=outputs)
            self.assertTrue(tf.training)
            # cannot normalize indices b/c we don't know dimension yet
            self.assertEqual(tf._outputs, [-1])

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertAllClose(Y_tf[..., 1], Y[..., 1].pow(power))
            self.assertAllClose(Y_tf[..., 0], Y[..., 0])
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            with self.assertRaises(NotImplementedError):
                tf_subset = tf.subset_output(idcs=[0])

            # with observation noise
            tf = Power(power=power, outputs=outputs)
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            with self.assertRaises(NotImplementedError):
                tf(Y, Yvar)

            # error on untransform_posterior
            with self.assertRaises(NotImplementedError):
                tf.untransform_posterior(None)

            # test subset_output with positive on subset of outcomes (pos. index)
            tf = Power(power=power, outputs=[0])
            Y_tf, Yvar_tf = tf(Y, None)
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]], None)
            self.assertTrue(torch.equal(Y_tf_subset, Y_tf[..., [0]]))
            self.assertIsNone(Yvar_tf_subset)

    def test_bilog(self, seed=0):
        torch.random.manual_seed(seed)

        ms = (1, 2)
        batch_shapes = (torch.Size(), torch.Size([2]))
        dtypes = (torch.float, torch.double)

        # test transform and untransform
        for m, batch_shape, dtype in itertools.product(ms, batch_shapes, dtypes):
            # test init
            tf = Bilog()
            self.assertTrue(tf.training)
            self.assertIsNone(tf._outputs)

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertAllClose(Y_tf, Y.sign() * (Y.abs() + 1).log())
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y, atol=1e-7)
            self.assertIsNone(Yvar_utf)

            # subset_output
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]])
            self.assertTrue(torch.equal(Y_tf[..., [0]], Y_tf_subset))
            self.assertIsNone(Yvar_tf_subset)

            # test error if observation noise present
            tf = Bilog()
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
            tf = Bilog()
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

        # test transforming a subset of outcomes
        for batch_shape, dtype in itertools.product(batch_shapes, dtypes):
            m = 2
            outputs = [-1]

            # test init
            tf = Bilog(outputs=outputs)
            self.assertTrue(tf.training)
            # cannot normalize indices b/c we don't know dimension yet
            self.assertEqual(tf._outputs, [-1])

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertTrue(
                torch.allclose(
                    Y_tf[..., 1], Y[..., 1].sign() * (Y[..., 1].abs() + 1).log()
                )
            )
            self.assertAllClose(Y_tf[..., 0], Y[..., 0])
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            with self.assertRaises(NotImplementedError):
                tf_subset = tf.subset_output(idcs=[0])

            # with observation noise
            tf = Bilog(outputs=outputs)
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            with self.assertRaises(NotImplementedError):
                tf(Y, Yvar)

            # error on untransform_posterior
            with self.assertRaises(NotImplementedError):
                tf.untransform_posterior(None)

            # test subset_output with positive on subset of outcomes (pos. index)
            tf = Bilog(outputs=[0])
            Y_tf, Yvar_tf = tf(Y, None)
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]], None)
            self.assertTrue(torch.equal(Y_tf_subset, Y_tf[..., [0]]))
            self.assertIsNone(Yvar_tf_subset)


class TestInfeasibleTransform(BotorchTestCase):
    def test_infeasible_transform_init(self):
        """Test initialization of InfeasibleTransform."""
        batch_shape = torch.Size([2, 3])
        transform = InfeasibleTransform(batch_shape=batch_shape)
        assert transform._batch_shape == batch_shape
        assert not transform._is_trained
        assert transform._shift is None
        assert torch.isnan(transform.warped_bad_value)

    def test_infeasible_transform_forward(self):
        """Test forward transformation with NaN values."""
        batch_shape = torch.Size([2])
        transform = InfeasibleTransform(batch_shape=batch_shape)

        # Create test data with NaN values
        Y = torch.randn(*batch_shape, 3, 2)
        Y[..., 0, 0] = float("nan")
        Y_orig = Y.clone()

        # Test forward pass in training mode
        transform.train()
        Y_tf, _ = transform.forward(Y, None)

        # Check that transform is now trained
        assert transform._is_trained
        assert transform._shift is not None
        assert not torch.isnan(transform.warped_bad_value).all()

        # Check that NaN values are replaced with warped_bad_value
        assert not torch.isnan(Y_tf).any()

        # Test forward pass in eval mode
        transform.eval()
        Y_tf_eval, _ = transform.forward(Y_orig, None)

        # Check that NaN values are replaced consistently
        assert not torch.isnan(Y_tf_eval).any()

    def test_infeasible_transform_untransform(self):
        """Test untransform functionality."""
        transform = InfeasibleTransform(batch_shape=torch.Size([]))

        # Should raise error if not trained
        with self.assertRaises(RuntimeError):
            transform.untransform(torch.tensor([1.0, 2.0]), None)

        # Train the transform first
        batch_shape = torch.Size([2])
        transform = InfeasibleTransform(batch_shape=batch_shape)
        Y = torch.randn(*batch_shape, 3, 2)
        Y[..., 0, 0] = float("nan")

        transform.train()
        Y_tf, _ = transform.forward(Y, None)

        # Test untransform
        Y_untf, _ = transform.untransform(Y_tf, None)

        # Check that values are properly untransformed
        assert torch.allclose(Y_untf[:, 1:], Y[:, 1:], rtol=1e-4)

        # test the unwarped_bad_value
        assert torch.allclose(transform.warped_bad_value[:, 0], Y_untf[..., 0, 0])

    def test_infeasible_transform_batch_shape_validation(self):
        """Test batch shape validation."""
        transform = InfeasibleTransform(batch_shape=torch.Size([2]))

        # Wrong batch shape should raise error
        with self.assertRaises(RuntimeError):
            transform.forward(torch.randn(3, 4, 2), None)

    def test_infeasible_transform_empty_input(self):
        """Test handling of empty input."""
        transform = InfeasibleTransform(batch_shape=torch.Size([]))

        # Empty input should raise error
        with self.assertRaises(ValueError):
            transform.forward(torch.tensor([]).reshape(0, 1), None)

    def test_infeasible_transform_all_nan(self):
        """Test handling of all-NaN input."""
        transform = InfeasibleTransform(batch_shape=torch.Size([]))

        Y = torch.tensor([[float("nan"), float("nan")]])
        transform.train()
        with self.assertRaises(RuntimeError):
            transform.forward(Y, None)

    def test_infeasible_transform_no_nan(self):
        """Test handling of input with no NaN values."""
        transform = InfeasibleTransform(batch_shape=torch.Size([]))

        Y = torch.tensor([[1.0, 2.0, 3.0]])
        transform.train()
        Y_tf, _ = transform.forward(Y, None)

        # Check that transformation preserves finite values
        assert not torch.isnan(Y_tf).any()
        Y_untf, _ = transform.untransform(Y_tf, None)
        assert torch.allclose(Y_untf, Y, rtol=1e-4)


class TestLogWarperTransform(BotorchTestCase):
    def test_log_warper_transform_init(self):
        """Test initialization of LogWarperTransform."""
        batch_shape = torch.Size([2, 3])
        transform = LogWarperTransform(offset=2.0, batch_shape=batch_shape)
        self.assertEqual(transform._batch_shape, batch_shape)
        self.assertEqual(transform.offset.item(), 2.0)

        # Test invalid offset
        with self.assertRaisesRegex(ValueError, "offset must be positive"):
            LogWarperTransform(offset=0.0)
        with self.assertRaisesRegex(ValueError, "offset must be positive"):
            LogWarperTransform(offset=-1.0)

    def test_log_warper_transform_forward(self):
        """Test forward transformation."""
        batch_shape = torch.Size([2])
        transform = LogWarperTransform(offset=2.0, batch_shape=batch_shape)

        # Create test data with NaN values
        Y = torch.randn(*batch_shape, 3, 2)
        Y[..., 0, 0] = float("nan")
        Y_orig = Y.clone()

        # Test forward pass in training mode
        transform.train()
        Y_tf, _ = transform.forward(Y, None)

        # Check that transform is now trained
        labels_min = transform._labels_min.clone()
        labels_max = transform._labels_max.clone()

        assert transform._is_trained
        assert torch.isfinite(labels_min).all()
        assert torch.isfinite(labels_max).all()
        assert (torch.isnan(Y_tf) == torch.isnan(Y_orig)).all()

        # Test forward pass in eval mode
        transform.eval()
        Y_tf_eval, _ = transform.forward(Y_tf, None)

        # Check that NaN values are replaced consistently
        assert (torch.isnan(Y_tf_eval) == torch.isnan(Y_tf)).all()
        assert torch.allclose(labels_min, transform._labels_min)
        assert torch.allclose(labels_max, transform._labels_max)

    def test_log_warper_transform_untransform(self):
        """Test untransform functionality."""
        batch_shape = torch.Size([2])
        transform = LogWarperTransform(offset=2.0, batch_shape=batch_shape)

        # Should raise error if not trained
        with self.assertRaises(RuntimeError):
            transform.untransform(torch.tensor([1.0, 2.0]), None)

        # Train the transform first
        Y = torch.randn(*batch_shape, 3, 2)
        Y[..., 0, 0] = float("nan")

        transform.train()
        Y_tf, _ = transform.forward(Y, None)

        # Test untransform
        Y_untf, _ = transform.untransform(Y_tf, None)

        # Check that values are properly untransformed
        assert torch.allclose(Y_untf[:, 1:], Y[:, 1:], rtol=1e-4)

        # test the nan values don't change
        assert torch.isnan(Y_untf[..., 0, 0]).all()

    def test_log_warper_transform_batch_shape_validation(self):
        """Test batch shape validation."""
        transform = LogWarperTransform(offset=2.0, batch_shape=torch.Size([2]))

        # Wrong batch shape should raise error
        with self.assertRaises(RuntimeError):
            transform.forward(torch.randn(3, 4, 2), None)

    def test_log_warper_transform_empty_input(self):
        """Test handling of empty input."""
        transform = LogWarperTransform(offset=2.0, batch_shape=torch.Size([]))

        # Empty input should raise error
        with self.assertRaises(ValueError):
            transform.forward(torch.tensor([]).reshape(0, 1), None)


class TestHalfRankTransform(BotorchTestCase):
    def test_init(self):
        # Test initialization
        transform = HalfRankTransform()
        self.assertIsNone(transform._batch_shape)
        self.assertFalse(transform._is_trained)
        self.assertEqual(transform._unique_labels, {})
        self.assertEqual(transform._warped_labels, {})

        # Test with batch shape
        batch_shape = torch.Size([2, 3])
        transform = HalfRankTransform(batch_shape=batch_shape)
        self.assertEqual(transform._batch_shape, batch_shape)

    def test_transform_simple_case(self):
        # Test with simple 1D tensor
        transform = HalfRankTransform()
        Y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
        Y_transformed, _ = transform.forward(Y)

        # Values above median should remain unchanged
        self.assertTrue(
            torch.allclose(Y_transformed[Y.squeeze() > 3.0], Y[Y.squeeze() > 3.0])
        )

        # Check if transform is trained
        self.assertTrue(transform._is_trained)

        # Test untransform
        Y_untransformed, _ = transform.untransform(Y_transformed)
        self.assertTrue(torch.allclose(Y_untransformed, Y, rtol=1e-4))

    def test_transform_with_nans(self):
        transform = HalfRankTransform()
        Y = torch.tensor([1.0, float("nan"), 3.0, 4.0, 5.0]).reshape(-1, 1)
        Y_transformed, _ = transform.forward(Y)

        # NaN values should remain NaN
        self.assertTrue(torch.isnan(Y_transformed[torch.isnan(Y)]).all())

        # Non-NaN values above median should remain unchanged
        valid_mask = ~torch.isnan(Y.squeeze())
        median = torch.nanmedian(Y)
        self.assertTrue(
            torch.allclose(
                Y_transformed[valid_mask & (Y.squeeze() > median)],
                Y[valid_mask & (Y.squeeze() > median)],
            )
        )

    def test_transform_batch(self):
        batch_shape = torch.Size([2])
        transform = HalfRankTransform(batch_shape=batch_shape)
        Y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).reshape(2, 3, 1)
        Y_transformed, _ = transform.forward(Y)

        # Shape should be preserved
        self.assertEqual(Y_transformed.shape, Y.shape)

        # Test untransform
        Y_untransformed, _ = transform.untransform(Y_transformed)
        self.assertTrue(torch.allclose(Y_untransformed, Y, rtol=1e-4))

    def test_transform_multi_output(self):
        transform = HalfRankTransform()
        Y = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        Y_transformed, _ = transform.forward(Y)

        # Each output dimension should be transformed independently
        self.assertEqual(Y_transformed.shape, Y.shape)

        # Test untransform
        Y_untransformed, _ = transform.untransform(Y_transformed)
        self.assertTrue(torch.allclose(Y_untransformed, Y, rtol=1e-4))

    def test_error_cases(self):
        transform = HalfRankTransform()

        # Test all NaN case
        Y = torch.tensor([[float("nan")], [float("nan")]])
        with self.assertRaisesRegex(
            RuntimeError, "For at least one batch, all outcomes are NaN"
        ):
            transform.forward(Y)

        # Test untransform before training
        Y = torch.tensor([[1.0], [2.0]])
        with self.assertRaisesRegex(
            RuntimeError, "needs to be called before untransform"
        ):
            transform.untransform(Y)

        # Test with observation noise
        Y = torch.tensor([[1.0], [2.0]])
        Yvar = torch.tensor([[0.1], [0.1]])
        with self.assertRaisesRegex(
            NotImplementedError,
            "HalfRankTransform does not support transforming observation noise",
        ):
            transform.forward(Y, Yvar)

    def test_batch_shape_mismatch(self):
        batch_shape = torch.Size([2])
        transform = HalfRankTransform(batch_shape=batch_shape)
        Y = torch.tensor([[1.0], [2.0], [3.0]])  # Wrong batch shape
        with self.assertRaises(RuntimeError):
            transform.forward(Y)

    def test_extrapolation(self):
        transform = HalfRankTransform()
        Y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
        Y_transformed, _ = transform.forward(Y)

        # Test extrapolation below minimum
        Y_test = torch.tensor([0.0]).reshape(-1, 1)
        Y_test_transformed, _ = transform.forward(Y_test)
        Y_test_untransformed, _ = transform.untransform(Y_test_transformed)

        # The untransformed value should be close to but below the minimum
        self.assertLess(Y_test_untransformed.item(), Y.min())

    def test_interpolation(self):
        transform = HalfRankTransform()
        Y = torch.tensor([1.0, 3.0, 5.0]).reshape(-1, 1)
        Y_transformed, _ = transform.forward(Y)

        # Test interpolation between values
        Y_test = torch.tensor([2.0]).reshape(-1, 1)
        Y_test_transformed, _ = transform.forward(Y_test)
        Y_test_untransformed, _ = transform.untransform(Y_test_transformed)

        # The untransformed value should be close to the original
        self.assertTrue(torch.allclose(Y_test_untransformed, Y_test, rtol=1e-4))
