#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from string import ascii_lowercase

import torch
from botorch.utils.context_managers import (
    delattr_ctx,
    module_rollback_ctx,
    parameter_rollback_ctx,
    TensorCheckpoint,
    zero_grad_ctx,
)
from botorch.utils.testing import BotorchTestCase
from torch.nn import Module, Parameter


class TestContextManagers(BotorchTestCase):
    def setUp(self):
        super().setUp()
        module = self.module = Module()
        for i, name in enumerate(ascii_lowercase[:3], start=1):
            values = torch.rand(2).to(torch.float16)
            param = Parameter(values.to(torch.float64), requires_grad=bool(i % 2))
            module.register_parameter(name, param)

    def test_delattr_ctx(self):
        # Test temporary removal of attributes
        a = self.module.a
        b = self.module.b
        with delattr_ctx(self.module, "a", "b"):
            self.assertIsNone(getattr(self.module, "a", None))
            self.assertIsNone(getattr(self.module, "b", None))
            self.assertTrue(self.module.c is not None)

        # Test that removed attributes get restored
        self.assertTrue(self.module.a.equal(a))
        self.assertTrue(self.module.b.equal(b))

        with self.assertRaisesRegex(ValueError, "Attribute .* missing"):
            with delattr_ctx(self.module, "z", enforce_hasattr=True):
                pass  # pragma: no cover

    def test_parameter_rollback_ctx(self):
        # Test that only unfiltered parameters get rolled back
        a = self.module.a.detach().clone()
        b = self.module.b.detach().clone()
        c = self.module.c.detach().clone()
        parameters = dict(self.module.named_parameters())
        with parameter_rollback_ctx(parameters, dtype=torch.float16) as ckpt:
            for tnsr, _, __ in ckpt.values():  # test whether dtype is obeyed
                self.assertEqual(torch.float16, tnsr.dtype)

            self.module.a.data[...] = 0
            self.module.b.data[...] = 0
            self.module.c.data[...] = 0
            del ckpt["c"]  # test whether changes to checkpoint dict are respected

        self.assertTrue(self.module.a.equal(a))
        self.assertTrue(self.module.b.equal(b))
        self.assertTrue(self.module.c.eq(0).all())

        # Test rolling back to a user-provided checkpoint
        with parameter_rollback_ctx(
            parameters, checkpoint={"c": TensorCheckpoint(c, c.device, c.dtype)}
        ):
            pass
        self.assertTrue(self.module.c.equal(c))

    def test_module_rollback_ctx(self):
        # Test that only unfiltered objects get rolled back
        a = self.module.a.detach().clone()
        b = self.module.b.detach().clone()
        c = self.module.c.detach().clone()
        with module_rollback_ctx(
            self.module, lambda name: name == "a", dtype=torch.float16
        ) as ckpt:
            for tnsr, _, __ in ckpt.values():  # test whether dtype is obeyed
                self.assertEqual(torch.float16, tnsr.dtype)

            self.module.a.data[...] = 0
            self.module.b.data[...] = 0
            self.module.c.data[...] = 0

        self.assertTrue(self.module.a.equal(a))
        self.assertTrue(self.module.b.eq(0).all())
        self.assertTrue(self.module.c.eq(0).all())

        # Test that changes to checkpoint dict are reflected in rollback state
        with module_rollback_ctx(self.module) as ckpt:
            self.module.a.data[...] = 1
            self.module.b.data[...] = 1
            self.module.c.data[...] = 1
            del ckpt["a"]

        self.assertTrue(self.module.a.eq(1).all())
        self.assertTrue(self.module.b.eq(0).all())
        self.assertTrue(self.module.c.eq(0).all())

        # Test rolling back to a user-provided checkpoint
        checkpoint = {
            "a": TensorCheckpoint(a, a.device, a.dtype),
            "b": TensorCheckpoint(b, b.device, b.dtype),
            "c": TensorCheckpoint(c, c.device, c.dtype),
        }
        with module_rollback_ctx(module=self.module, checkpoint=checkpoint):
            pass
        self.assertTrue(self.module.a.equal(a))
        self.assertTrue(self.module.b.equal(b))
        self.assertTrue(self.module.c.equal(c))

        # Test that items in checkpoint get inserted into state_dict
        with delattr_ctx(self.module, "a"):
            with self.assertRaisesRegex(  # should fail when attempting to rollback
                RuntimeError, r'Unexpected key\(s\) in state_dict: "a"'
            ):
                with module_rollback_ctx(module=self.module, checkpoint=checkpoint):
                    pass

    def test_zero_grad_ctx(self):
        params = (Parameter(torch.rand(1)), Parameter(torch.rand(1)))
        sum(params).backward()
        with zero_grad_ctx(params, zero_on_enter=False, zero_on_exit=True):
            self.assertFalse(any(x.grad.eq(0).all() for x in params))
        self.assertTrue(all(x.grad.eq(0).all() for x in params))

        sum(params).backward()
        with zero_grad_ctx(params, zero_on_enter=True, zero_on_exit=False):
            self.assertTrue(all(x.grad.eq(0).all() for x in params))
            sum(params).backward()
        self.assertFalse(any(x.grad.eq(0).all() for x in params))
