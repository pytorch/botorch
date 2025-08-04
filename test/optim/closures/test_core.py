#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock

import numpy as np
import torch
from botorch.optim.closures.core import (
    FILL_VALUE,
    ForwardBackwardClosure,
    get_tensors_as_ndarray_1d,
    NdarrayOptimizationClosure,
)
from botorch.optim.utils import as_ndarray
from botorch.utils.testing import BotorchTestCase
from linear_operator.utils.errors import NanError, NotPSDError
from torch.nn import Module, Parameter


class ToyModule(Module):
    def __init__(self, w: Parameter, b: Parameter, x: Parameter, dummy: Parameter):
        r"""Toy module for unit testing."""
        super().__init__()
        self.w = w
        self.b = b
        self.x = x
        self.dummy = dummy

    def forward(self) -> torch.Tensor:
        return self.w * self.x + self.b

    @property
    def free_parameters(self) -> dict[str, torch.Tensor]:
        return {n: p for n, p in self.named_parameters() if p.requires_grad}


class TestForwardBackwardClosure(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        module = ToyModule(
            w=Parameter(torch.tensor(2.0)),
            b=Parameter(torch.tensor(3.0), requires_grad=False),
            x=Parameter(torch.tensor(4.0)),
            dummy=Parameter(torch.tensor(5.0)),
        ).to(self.device)
        self.modules = {}
        for dtype in ("float32", "float64"):
            self.modules[dtype] = module.to(dtype=getattr(torch, dtype))

    def test_main(self):
        for module in self.modules.values():
            closure = ForwardBackwardClosure(module, module.free_parameters)

            # Test __init__
            closure = ForwardBackwardClosure(module, module.free_parameters)
            self.assertEqual(module.free_parameters, closure.parameters)

            # Test return values
            value, (dw, dx, dd) = closure()
            self.assertTrue(value.equal(module()))
            self.assertTrue(dw.equal(module.x))
            self.assertTrue(dx.equal(module.w))
            self.assertEqual(dd, None)


class TestNdarrayOptimizationClosure(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        self.module = ToyModule(
            w=Parameter(torch.tensor(2.0)),
            b=Parameter(torch.tensor(3.0), requires_grad=False),
            x=Parameter(torch.tensor(4.0)),
            dummy=Parameter(torch.tensor(5.0)),
        ).to(self.device)

        self.wrappers = {}
        for dtype in ("float32", "float64"):
            module = self.module.to(dtype=getattr(torch, dtype))
            closure = ForwardBackwardClosure(module, module.free_parameters)
            wrapper = NdarrayOptimizationClosure(closure, closure.parameters)
            self.wrappers[dtype] = wrapper

    def test_main(self):
        for wrapper in self.wrappers.values():
            # Test setter/getter
            state = get_tensors_as_ndarray_1d(wrapper.closure.parameters)
            other = np.random.randn(*state.shape).astype(state.dtype)

            wrapper.state = other
            self.assertTrue(np.allclose(other, wrapper.state))

            index = 0
            for param in wrapper.closure.parameters.values():
                size = param.numel()
                self.assertTrue(
                    np.allclose(other[index : index + size], as_ndarray(param.view(-1)))
                )
                index += size

            wrapper.state = state
            self.assertTrue(np.allclose(state, wrapper.state))

            # Test __call__
            value, grads = wrapper(other)
            self.assertTrue(np.allclose(other, wrapper.state))
            self.assertIsInstance(value, np.ndarray)
            self.assertIsInstance(grads, np.ndarray)

            # Test return values
            value_tensor, grad_tensors = wrapper.closure()  # get raw Tensor equivalents
            self.assertTrue(np.allclose(value, as_ndarray(value_tensor)))
            index = 0
            for x, dx in zip(wrapper.parameters.values(), grad_tensors, strict=True):
                size = x.numel()
                grad = grads[index : index + size]
                if dx is None:
                    self.assertTrue((grad == FILL_VALUE).all())
                else:
                    self.assertTrue(np.allclose(grad, as_ndarray(dx)))
                index += size

            module = wrapper.closure.forward
            self.assertTrue(np.allclose(grads[0], as_ndarray(module.x)))
            self.assertTrue(np.allclose(grads[1], as_ndarray(module.w)))
            self.assertEqual(grads[2], FILL_VALUE)

            # Test persistence
            self.assertIs(
                wrapper._get_gradient_ndarray(), wrapper._get_gradient_ndarray()
            )

    def test_exceptions(self):
        for wrapper in self.wrappers.values():
            mock_closure = MagicMock(return_value=wrapper.closure())
            mock_wrapper = NdarrayOptimizationClosure(
                mock_closure, wrapper.closure.parameters
            )
            with self.assertRaisesRegex(NotPSDError, "foo"):
                mock_wrapper.closure.side_effect = NotPSDError("foo")
                mock_wrapper()

            for exception in (
                NanError("foo"),
                RuntimeError("singular"),
                RuntimeError("input is not positive-definite"),
            ):
                mock_wrapper.closure.side_effect = exception
                value, grads = mock_wrapper()
                self.assertTrue(np.isnan(value).all())
                self.assertTrue(np.isnan(grads).all())
