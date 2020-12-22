#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from copy import deepcopy

import torch
from botorch.distributions.distributions import Kumaraswamy
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputTransform,
    Warp,
    Log10,
    Normalize,
    Round,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.priors import LogNormalPrior
from torch.nn import Module


def get_test_warp(indices, **kwargs):
    warp_tf = Warp(indices=indices, **kwargs)
    warp_tf._set_concentration(0, torch.tensor([1.0, 2.0]))
    warp_tf._set_concentration(1, torch.tensor([2.0, 3.0]))
    return warp_tf


class NotSoAbstractInputTransform(InputTransform, Module):
    def __init__(
        self, transform_on_train, transform_on_eval, transform_on_preprocess=False
    ):
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_preprocess = transform_on_preprocess

    def transform(self, X):
        return X + 1


class TestInputTransforms(BotorchTestCase):
    def test_abstract_base_input_transform(self):
        with self.assertRaises(TypeError):
            InputTransform()
        X = torch.zeros([1])
        X_tf = torch.ones([1])
        # test transform_on_train and transform_on_eval
        ipt = NotSoAbstractInputTransform(
            transform_on_train=True, transform_on_eval=True
        )
        self.assertTrue(torch.equal(ipt(X), X_tf))
        ipt.eval()
        self.assertTrue(torch.equal(ipt(X), X_tf))
        ipt = NotSoAbstractInputTransform(
            transform_on_train=True, transform_on_eval=False
        )
        self.assertTrue(torch.equal(ipt(X), X_tf))
        ipt.eval()
        self.assertTrue(torch.equal(ipt(X), X))
        ipt = NotSoAbstractInputTransform(
            transform_on_train=False, transform_on_eval=True
        )
        self.assertTrue(torch.equal(ipt(X), X))
        ipt.eval()
        self.assertTrue(torch.equal(ipt(X), X_tf))
        ipt = NotSoAbstractInputTransform(
            transform_on_train=False, transform_on_eval=False
        )
        self.assertTrue(torch.equal(ipt(X), X))
        ipt.eval()
        self.assertTrue(torch.equal(ipt(X), X))

        # test equals
        ipt2 = NotSoAbstractInputTransform(
            transform_on_train=False, transform_on_eval=False
        )
        self.assertTrue(ipt.equals(ipt2))
        ipt3 = NotSoAbstractInputTransform(
            transform_on_train=True, transform_on_eval=False
        )
        self.assertFalse(ipt.equals(ipt3))

        with self.assertRaises(NotImplementedError):
            ipt.untransform(None)

        # test preprocess_transform
        ipt4 = NotSoAbstractInputTransform(
            transform_on_train=True,
            transform_on_eval=False,
            transform_on_preprocess=True,
        )
        self.assertTrue(torch.equal(ipt4.preprocess_transform(X), X + 1))
        ipt4.transform_on_preprocess = False
        self.assertTrue(torch.equal(ipt4.preprocess_transform(X), X))

    def test_normalize(self):
        for dtype in (torch.float, torch.double):

            # basic init, learned bounds
            nlz = Normalize(d=2)
            self.assertTrue(nlz.learn_bounds)
            self.assertTrue(nlz.training)
            self.assertEqual(nlz._d, 2)
            self.assertEqual(nlz.mins.shape, torch.Size([1, 2]))
            self.assertEqual(nlz.ranges.shape, torch.Size([1, 2]))
            nlz = Normalize(d=2, batch_shape=torch.Size([3]))
            self.assertTrue(nlz.learn_bounds)
            self.assertTrue(nlz.training)
            self.assertEqual(nlz._d, 2)
            self.assertEqual(nlz.mins.shape, torch.Size([3, 1, 2]))
            self.assertEqual(nlz.ranges.shape, torch.Size([3, 1, 2]))

            # basic init, fixed bounds
            bounds = torch.zeros(2, 2, device=self.device, dtype=dtype)
            nlz = Normalize(d=2, bounds=bounds)
            self.assertFalse(nlz.learn_bounds)
            self.assertTrue(nlz.training)
            self.assertEqual(nlz._d, 2)
            self.assertTrue(torch.equal(nlz.mins, bounds[..., 0:1, :]))
            self.assertTrue(
                torch.equal(nlz.mins, bounds[..., 1:2, :] - bounds[..., 0:1, :])
            )
            # test .to
            other_dtype = torch.float if dtype == torch.double else torch.double
            nlz.to(other_dtype)
            self.assertTrue(nlz.mins.dtype == other_dtype)
            # test incompatible dimensions of specified bounds
            with self.assertRaises(BotorchTensorDimensionError):
                bounds = torch.zeros(2, 3, device=self.device, dtype=dtype)
                Normalize(d=2, bounds=bounds)

            # basic usage
            for batch_shape in (torch.Size(), torch.Size([3])):
                # learned bounds
                nlz = Normalize(d=2, batch_shape=batch_shape)
                X = torch.randn(*batch_shape, 4, 2, device=self.device, dtype=dtype)
                X_nlzd = nlz(X)
                self.assertEqual(X_nlzd.min().item(), 0.0)
                self.assertEqual(X_nlzd.max().item(), 1.0)
                nlz.eval()
                X_unnlzd = nlz.untransform(X_nlzd)
                self.assertTrue(torch.allclose(X, X_unnlzd, atol=1e-4, rtol=1e-4))
                expected_bounds = torch.cat(
                    [X.min(dim=-2, keepdim=True)[0], X.max(dim=-2, keepdim=True)[0]],
                    dim=-2,
                )
                self.assertTrue(torch.allclose(nlz.bounds, expected_bounds))
                # test errors on wrong shape
                nlz = Normalize(d=2, batch_shape=batch_shape)
                X = torch.randn(*batch_shape, 2, 1, device=self.device, dtype=dtype)
                with self.assertRaises(BotorchTensorDimensionError):
                    nlz(X)

                # fixed bounds
                bounds = torch.tensor(
                    [[-2.0, -1], [1, 2.0]], device=self.device, dtype=dtype
                ).expand(*batch_shape, 2, 2)
                nlz = Normalize(d=2, bounds=bounds)
                X = torch.rand(*batch_shape, 4, 2, device=self.device, dtype=dtype)
                X_nlzd = nlz(X)
                self.assertTrue(torch.equal(nlz.bounds, bounds))
                X_unnlzd = nlz.untransform(X_nlzd)
                self.assertTrue(torch.allclose(X, X_unnlzd, atol=1e-4, rtol=1e-4))

                # test no normalization on eval
                nlz = Normalize(
                    d=2, bounds=bounds, batch_shape=batch_shape, transform_on_eval=False
                )
                X_nlzd = nlz(X)
                X_unnlzd = nlz.untransform(X_nlzd)
                self.assertTrue(torch.allclose(X, X_unnlzd, atol=1e-4, rtol=1e-4))
                nlz.eval()
                self.assertTrue(torch.equal(nlz(X), X))

                # test no normalization on train
                nlz = Normalize(
                    d=2,
                    bounds=bounds,
                    batch_shape=batch_shape,
                    transform_on_train=False,
                )
                X_nlzd = nlz(X)
                self.assertTrue(torch.equal(nlz(X), X))
                nlz.eval()
                X_nlzd = nlz(X)
                X_unnlzd = nlz.untransform(X_nlzd)
                self.assertTrue(torch.allclose(X, X_unnlzd, atol=1e-4, rtol=1e-4))

                # test reverse
                nlz = Normalize(
                    d=2, bounds=bounds, batch_shape=batch_shape, reverse=True
                )
                X2 = nlz(X_nlzd)
                self.assertTrue(torch.allclose(X2, X, atol=1e-4, rtol=1e-4))
                X_nlzd2 = nlz.untransform(X2)
                self.assertTrue(torch.allclose(X_nlzd, X_nlzd2, atol=1e-4, rtol=1e-4))

                # test equals
                nlz2 = Normalize(
                    d=2, bounds=bounds, batch_shape=batch_shape, reverse=False
                )
                self.assertFalse(nlz.equals(nlz2))
                nlz3 = Normalize(
                    d=2, bounds=bounds, batch_shape=batch_shape, reverse=True
                )
                self.assertTrue(nlz.equals(nlz3))
                new_bounds = bounds + 1
                nlz4 = Normalize(
                    d=2, bounds=new_bounds, batch_shape=batch_shape, reverse=True
                )
                self.assertFalse(nlz.equals(nlz4))
                nlz5 = Normalize(d=2, batch_shape=batch_shape)
                self.assertFalse(nlz.equals(nlz5))

    def test_chained_input_transform(self):

        ds = (1, 2)
        batch_shapes = (torch.Size(), torch.Size([2]))
        dtypes = (torch.float, torch.double)

        for d, batch_shape, dtype in itertools.product(ds, batch_shapes, dtypes):
            bounds = torch.tensor(
                [[-2.0] * d, [2.0] * d], device=self.device, dtype=dtype
            )
            tf1 = Normalize(d=d, bounds=bounds, batch_shape=batch_shape)
            tf2 = Normalize(d=d, batch_shape=batch_shape)
            tf = ChainedInputTransform(stz_fixed=tf1, stz_learned=tf2)
            tf1_, tf2_ = deepcopy(tf1), deepcopy(tf2)
            self.assertTrue(tf.training)
            self.assertEqual(sorted(tf.keys()), ["stz_fixed", "stz_learned"])
            self.assertEqual(tf["stz_fixed"], tf1)
            self.assertEqual(tf["stz_learned"], tf2)

            # make copies for validation below
            tf1_, tf2_ = deepcopy(tf1), deepcopy(tf2)

            X = torch.rand(*batch_shape, 4, d, device=self.device, dtype=dtype)
            X_tf = tf(X)
            X_tf_ = tf2_(tf1_(X))
            self.assertTrue(tf1.training)
            self.assertTrue(tf2.training)
            self.assertTrue(torch.equal(X_tf, X_tf_))
            X_utf = tf.untransform(X_tf)
            self.assertTrue(torch.allclose(X_utf, X, atol=1e-4, rtol=1e-4))

            # test not transformed on eval
            tf1.transform_on_eval = False
            tf2.transform_on_eval = False
            tf = ChainedInputTransform(stz_fixed=tf1, stz_learned=tf2)
            tf.eval()
            self.assertTrue(torch.equal(tf(X), X))
            tf1.transform_on_eval = True
            tf2.transform_on_eval = True
            tf = ChainedInputTransform(stz_fixed=tf1, stz_learned=tf2)
            tf.eval()
            self.assertTrue(torch.equal(tf(X), X_tf))

            # test not transformed on train
            tf1.transform_on_train = False
            tf2.transform_on_train = False
            tf = ChainedInputTransform(stz_fixed=tf1, stz_learned=tf2)
            self.assertTrue(torch.equal(tf(X), X))

            # test __eq__
            other_tf = ChainedInputTransform(stz_fixed=tf1, stz_learned=tf2)
            self.assertTrue(tf.equals(other_tf))
            # change order
            other_tf = ChainedInputTransform(stz_learned=tf2, stz_fixed=tf1)
            self.assertFalse(tf.equals(other_tf))

            # test preprocess_transform
            tf2.transform_on_preprocess = False
            tf1.transform_on_preprocess = True
            tf = ChainedInputTransform(stz_fixed=tf1, stz_learned=tf2)
            self.assertTrue(torch.equal(tf.preprocess_transform(X), tf1.transform(X)))

    def test_round_transform(self):
        for dtype in (torch.float, torch.double):
            # basic init
            int_idcs = [0, 2]
            round_tf = Round(indices=[0, 2])
            self.assertEqual(round_tf.indices.tolist(), int_idcs)
            self.assertTrue(round_tf.training)
            self.assertTrue(round_tf.approximate)
            self.assertEqual(round_tf.tau, 1e-3)

            # basic usage
            for batch_shape, approx in itertools.product(
                (torch.Size(), torch.Size([3])), (False, True)
            ):
                X = 5 * torch.rand(*batch_shape, 4, 3, device=self.device, dtype=dtype)
                round_tf = Round(indices=[0, 2], approximate=approx)
                X_rounded = round_tf(X)
                exact_rounded_X_ints = X[..., int_idcs].round()
                # check non-integers parameters are unchanged
                self.assertTrue(torch.equal(X_rounded[..., 1], X[..., 1]))
                if approx:
                    # check that approximate rounding is closer to rounded values than
                    # the original inputs
                    self.assertTrue(
                        (
                            (X_rounded[..., int_idcs] - exact_rounded_X_ints).abs()
                            <= (X[..., int_idcs] - exact_rounded_X_ints).abs()
                        ).all()
                    )
                else:
                    # check that exact rounding behaves as expected
                    self.assertTrue(
                        torch.equal(X_rounded[..., int_idcs], exact_rounded_X_ints)
                    )
                with self.assertRaises(NotImplementedError):
                    round_tf.untransform(X_rounded)

                # test no transform on eval
                round_tf = Round(
                    indices=int_idcs, approximate=approx, transform_on_eval=False
                )
                X_rounded = round_tf(X)
                self.assertFalse(torch.equal(X, X_rounded))
                round_tf.eval()
                X_rounded = round_tf(X)
                self.assertTrue(torch.equal(X, X_rounded))

                # test no transform on train
                round_tf = Round(
                    indices=int_idcs, approximate=approx, transform_on_train=False
                )
                X_rounded = round_tf(X)
                self.assertTrue(torch.equal(X, X_rounded))
                round_tf.eval()
                X_rounded = round_tf(X)
                self.assertFalse(torch.equal(X, X_rounded))

                # test equals
                round_tf2 = Round(
                    indices=int_idcs, approximate=approx, transform_on_train=False
                )
                self.assertTrue(round_tf.equals(round_tf2))
                # test different transform_on_train
                round_tf2 = Round(indices=int_idcs, approximate=approx)
                self.assertFalse(round_tf.equals(round_tf2))
                # test different approx
                round_tf2 = Round(
                    indices=int_idcs, approximate=not approx, transform_on_train=False
                )
                self.assertFalse(round_tf.equals(round_tf2))
                # test different indices
                round_tf2 = Round(
                    indices=[0, 1], approximate=approx, transform_on_train=False
                )
                self.assertFalse(round_tf.equals(round_tf2))

                # test preprocess_transform
                self.assertTrue(torch.equal(round_tf.preprocess_transform(X), X))
                round_tf.transform_on_preprocess = True
                self.assertTrue(
                    torch.equal(round_tf.preprocess_transform(X), X_rounded)
                )

    def test_log10_transform(self):
        for dtype in (torch.float, torch.double):
            # basic init
            indices = [0, 2]
            log_tf = Log10(indices=indices)
            self.assertEqual(log_tf.indices.tolist(), indices)
            self.assertTrue(log_tf.training)

            # basic usage
            for batch_shape in (torch.Size(), torch.Size([3])):
                X = 1 + 5 * torch.rand(
                    *batch_shape, 4, 3, device=self.device, dtype=dtype
                )
                log_tf = Log10(indices=indices)
                X_tf = log_tf(X)
                expected_X_tf = X.clone()
                expected_X_tf[..., indices] = expected_X_tf[..., indices].log10()
                # check non-integers parameters are unchanged
                self.assertTrue(torch.equal(expected_X_tf, X_tf))
                untransformed_X = log_tf.untransform(X_tf)
                self.assertTrue(torch.allclose(untransformed_X, X))

                # test no transform on eval
                log_tf = Log10(indices=indices, transform_on_eval=False)
                X_tf = log_tf(X)
                self.assertFalse(torch.equal(X, X_tf))
                log_tf.eval()
                X_tf = log_tf(X)
                self.assertTrue(torch.equal(X, X_tf))

                # test no transform on train
                log_tf = Log10(indices=indices, transform_on_train=False)
                X_tf = log_tf(X)
                self.assertTrue(torch.equal(X, X_tf))
                log_tf.eval()
                X_tf = log_tf(X)
                self.assertFalse(torch.equal(X, X_tf))

                # test equals
                log_tf2 = Log10(indices=indices, transform_on_train=False)
                self.assertTrue(log_tf.equals(log_tf2))
                # test different transform_on_train
                log_tf2 = Log10(indices=indices)
                self.assertFalse(log_tf.equals(log_tf2))
                # test different indices
                log_tf2 = Log10(indices=[0, 1], transform_on_train=False)
                self.assertFalse(log_tf.equals(log_tf2))

                # test preprocess_transform
                self.assertTrue(torch.equal(log_tf.preprocess_transform(X), X))
                log_tf.transform_on_preprocess = True
                self.assertTrue(torch.equal(log_tf.preprocess_transform(X), X_tf))

    def test_warp_transform(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}

            # basic init
            indices = [0, 2]
            warp_tf = get_test_warp(indices).to(**tkwargs)
            self.assertTrue(warp_tf.training)

            k = Kumaraswamy(warp_tf.concentration1, warp_tf.concentration0)

            self.assertEqual(warp_tf.indices.tolist(), indices)

            # basic usage
            for batch_shape in (torch.Size(), torch.Size([3])):
                X = torch.rand(*batch_shape, 4, 3, **tkwargs)
                with torch.no_grad():
                    warp_tf = get_test_warp(indices=indices).to(**tkwargs)
                    X_tf = warp_tf(X)
                    expected_X_tf = X.clone()
                    expected_X_tf[..., indices] = k.cdf(
                        expected_X_tf[..., indices]
                        .clamp_min(warp_tf.eps)
                        .clamp_max(1 - warp_tf.eps)
                    )
                    # check non-integers parameters are unchanged
                    self.assertTrue(torch.equal(expected_X_tf, X_tf))

                    # test untransform
                    untransformed_X = warp_tf.untransform(X_tf)

                    print(untransformed_X)
                    print(X)
                    self.assertTrue(
                        torch.allclose(untransformed_X, X, rtol=1e-3, atol=1e-3)
                    )

                    # test no transform on eval
                    warp_tf = get_test_warp(indices, transform_on_eval=False).to(
                        **tkwargs
                    )
                    X_tf = warp_tf(X)
                    self.assertFalse(torch.equal(X, X_tf))
                    warp_tf.eval()
                    X_tf = warp_tf(X)
                    self.assertTrue(torch.equal(X, X_tf))

                    # test no transform on train
                    warp_tf = get_test_warp(
                        indices=indices, transform_on_train=False
                    ).to(**tkwargs)
                    X_tf = warp_tf(X)
                    self.assertTrue(torch.equal(X, X_tf))
                    warp_tf.eval()
                    X_tf = warp_tf(X)
                    self.assertFalse(torch.equal(X, X_tf))

                    # test equals
                    warp_tf2 = get_test_warp(
                        indices=indices, transform_on_train=False
                    ).to(**tkwargs)
                    self.assertTrue(warp_tf.equals(warp_tf2))
                    # test different transform_on_train
                    warp_tf2 = get_test_warp(indices=indices)
                    self.assertFalse(warp_tf.equals(warp_tf2))
                    # test different indices
                    warp_tf2 = get_test_warp(
                        indices=[0, 1], transform_on_train=False
                    ).to(**tkwargs)
                    self.assertFalse(warp_tf.equals(warp_tf2))

                    # test preprocess_transform
                    self.assertTrue(torch.equal(warp_tf.preprocess_transform(X), X))
                    warp_tf.transform_on_preprocess = True
                    self.assertTrue(torch.equal(warp_tf.preprocess_transform(X), X_tf))

                    # test _set_concentration
                    warp_tf._set_concentration(0, warp_tf.concentration0.shape)
                    warp_tf._set_concentration(1, warp_tf.concentration1.shape)

                    # test concentration prior
                    prior0 = LogNormalPrior(0.0, 0.75).to(**tkwargs)
                    prior1 = LogNormalPrior(0.0, 0.5).to(**tkwargs)
                    warp_tf = get_test_warp(
                        indices=[0, 1],
                        concentration0_prior=prior0,
                        concentration1_prior=prior1,
                    )
                    for i, (name, module, p, _, _) in enumerate(warp_tf.named_priors()):
                        self.assertEqual(name, f"concentration{i}_prior")
                        self.assertIsInstance(p, LogNormalPrior)
                        self.assertEqual(p.base_dist.scale, 0.75 if i == 0 else 0.5)

                # test gradients
                X = 1 + 5 * torch.rand(*batch_shape, 4, 3, **tkwargs)
                warp_tf = get_test_warp(indices=indices).to(**tkwargs)
                X_tf = warp_tf(X)
                X_tf.sum().backward()
                for grad in (warp_tf.concentration0.grad, warp_tf.concentration1.grad):
                    self.assertIsNotNone(grad)
                    self.assertFalse(torch.isnan(grad).any())
                    self.assertFalse(torch.isinf(grad).any())
                    self.assertFalse((grad == 0).all())
