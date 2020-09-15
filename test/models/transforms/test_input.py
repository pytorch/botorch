#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from copy import deepcopy

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputTransform,
    Normalize,
)
from botorch.utils.testing import BotorchTestCase


class NotSoAbstractInputTransform(InputTransform):
    def __init__(self, transform_on_train, transform_on_eval):
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval

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
            tf = ChainedInputTransform(
                transform_on_eval=False, stz_fixed=tf1, stz_learned=tf2
            )
            tf.eval()
            self.assertTrue(torch.equal(tf(X), X))

            # test not transformed on train
            tf = ChainedInputTransform(
                transform_on_train=False, stz_fixed=tf1, stz_learned=tf2
            )
            self.assertTrue(torch.equal(tf(X), X))

            # test __eq__
            other_tf = ChainedInputTransform(
                transform_on_train=False, stz_fixed=tf1, stz_learned=tf2
            )
            self.assertTrue(tf.equals(other_tf))
            # change order
            other_tf = ChainedInputTransform(
                transform_on_train=False, stz_learned=tf2, stz_fixed=tf1
            )
            self.assertFalse(tf.equals(other_tf))
