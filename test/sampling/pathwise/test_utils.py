#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest.mock import patch

import torch
from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.pathwise.utils import (
    append_transform,
    ChainedTransform,
    ConstantMulTransform,
    CosineTransform,
    get_input_transform,
    get_output_transform,
    get_train_inputs,
    get_train_targets,
    InverseLengthscaleTransform,
    is_finite_dimensional,
    kernel_instancecheck,
    ModuleDictMixin,
    ModuleListMixin,
    OutcomeUntransformer,
    prepend_transform,
    SineCosineTransform,
    sparse_block_diag,
    TransformedModuleMixin,
    untransform_shape,
)
from botorch.utils.context_managers import delattr_ctx
from botorch.utils.testing import BotorchTestCase
from gpytorch import kernels
from torch import Size, Tensor
from torch.nn import Module


class DummyModule(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class TestMixins(BotorchTestCase):
    """Test cases for the mixin classes in botorch.sampling.pathwise.utils.mixins.

    These tests verify that the mixins properly integrate with PyTorch's Module system
    and provide the expected container-like interfaces.
    """

    def test_module_dict_mixin(self):
        """Test ModuleDictMixin's dictionary-like interface and module registration.

        This test verifies that:
        1. The mixin properly initializes with Module
        2. Dictionary operations work as expected
        3. Modules are properly registered and tracked
        """

        class TestDict(Module, ModuleDictMixin[DummyModule]):
            def __init__(self):
                Module.__init__(self)  # Initialize Module first
                ModuleDictMixin.__init__(self, "modules")  # Then initialize mixin

            def forward(self, x: Tensor) -> Tensor:
                return x

        test_dict = TestDict()
        module = DummyModule()
        test_dict["test"] = module  # Test __setitem__
        self.assertIs(test_dict["test"], module)  # Test __getitem__
        self.assertEqual(len(test_dict), 1)  # Test __len__
        self.assertEqual(list(test_dict.keys()), ["test"])  # Test keys()
        self.assertEqual(list(test_dict.values()), [module])  # Test values()
        self.assertEqual(list(test_dict.items()), [("test", module)])  # Test items()
        test_dict.update({"other": DummyModule()})  # Test update()
        self.assertEqual(len(test_dict), 2)
        del test_dict["test"]  # Test __delitem__
        self.assertEqual(len(test_dict), 1)

    def test_module_list_mixin(self):
        """Test ModuleListMixin's list-like interface and module registration.

        This test verifies that:
        1. The mixin properly initializes with Module
        2. List operations work as expected
        3. Modules are properly registered and tracked
        """

        class TestList(Module, ModuleListMixin[DummyModule]):
            def __init__(self):
                Module.__init__(self)  # Initialize Module first
                ModuleListMixin.__init__(self, "modules")  # Then initialize mixin

            def forward(self, x: Tensor) -> Tensor:
                return x

            def append(self, module: DummyModule) -> None:
                self._modules_list.append(module)  # Use the actual ModuleList

        test_list = TestList()
        module = DummyModule()
        test_list.append(module)  # Test append
        self.assertIs(test_list[0], module)  # Test __getitem__
        self.assertEqual(len(test_list), 1)  # Test __len__
        test_list[0] = DummyModule()  # Test __setitem__
        self.assertIsNot(test_list[0], module)
        del test_list[0]  # Test __delitem__
        self.assertEqual(len(test_list), 0)

    def test_transformed_module_mixin(self):
        """Test TransformedModuleMixin's transform application functionality.

        This test verifies that:
        1. The mixin properly handles input and output transforms
        2. Transforms are applied in the correct order
        3. The module works without transforms
        """

        class TestModule(TransformedModuleMixin):
            def forward(self, x: Tensor) -> Tensor:
                return x

        module = TestModule()
        x = torch.randn(3)
        self.assertTrue(x.equal(module(x)))  # Test without transforms

        # Test input transform
        module.input_transform = lambda x: 2 * x
        self.assertTrue((2 * x).equal(module(x)))

        # Test output transform
        module.output_transform = lambda x: x + 1
        self.assertTrue((2 * x + 1).equal(module(x)))  # Test both transforms


class TestTransforms(BotorchTestCase):
    def test_inverse_lengthscale_transform(self):
        tkwargs = {"device": self.device, "dtype": torch.float64}
        kernel = kernels.MaternKernel(nu=2.5, ard_num_dims=3).to(**tkwargs)
        with self.assertRaisesRegex(RuntimeError, "does not implement `lengthscale`"):
            InverseLengthscaleTransform(kernels.ScaleKernel(kernel))

        x = torch.rand(3, 3, **tkwargs)
        transform = InverseLengthscaleTransform(kernel)
        self.assertTrue(transform(x).equal(kernel.lengthscale.reciprocal() * x))

    def test_constant_mul_transform(self):
        x = torch.randn(3)
        transform = ConstantMulTransform(torch.tensor(2.0))
        self.assertTrue((2 * x).equal(transform(x)))

    def test_cosine_transform(self):
        x = torch.randn(3)
        transform = CosineTransform()
        self.assertTrue(x.cos().equal(transform(x)))

    def test_sine_cosine_transform(self):
        x = torch.randn(3)
        transform = SineCosineTransform()
        self.assertTrue(torch.concat([x.sin(), x.cos()], dim=-1).equal(transform(x)))

    def test_outcome_untransformer(self):
        for untransformer in (
            OutcomeUntransformer(transform=Standardize(m=1), num_outputs=1),
            OutcomeUntransformer(transform=Standardize(m=2), num_outputs=2),
        ):
            with torch.random.fork_rng():
                torch.random.manual_seed(0)
                y = torch.rand(untransformer.num_outputs, 4, device=self.device)
            x = untransformer.transform(y.T)[0].T
            self.assertTrue(y.allclose(untransformer(x)))


class TestHelpers(BotorchTestCase):
    def test_kernel_instancecheck(self):
        base = kernels.RBFKernel()
        scale = kernels.ScaleKernel(base)
        self.assertTrue(kernel_instancecheck(base, kernels.RBFKernel))
        self.assertTrue(kernel_instancecheck(scale, kernels.RBFKernel))
        self.assertFalse(kernel_instancecheck(base, kernels.MaternKernel))
        self.assertTrue(
            kernel_instancecheck(scale, (kernels.RBFKernel, kernels.MaternKernel), any)
        )
        # Test all reducer - should be false (scale kernel is not both RBF & Matern)
        self.assertFalse(
            kernel_instancecheck(
                scale, (kernels.RBFKernel, kernels.MaternKernel), all, max_depth=0
            )
        )

    def test_is_finite_dimensional(self):
        self.assertFalse(is_finite_dimensional(kernels.RBFKernel()))
        self.assertFalse(is_finite_dimensional(kernels.MaternKernel()))
        self.assertTrue(is_finite_dimensional(kernels.LinearKernel()))
        self.assertFalse(
            is_finite_dimensional(kernels.ScaleKernel(kernels.RBFKernel()))
        )

    def test_sparse_block_diag(self):
        blocks = [torch.eye(2), 2 * torch.eye(3)]
        result = sparse_block_diag(blocks)
        self.assertTrue(result.is_sparse)
        self.assertEqual(result.shape, (5, 5))
        dense = result.to_dense()
        self.assertTrue(torch.all(dense[:2, :2] == torch.eye(2)))
        self.assertTrue(torch.all(dense[2:, 2:] == 2 * torch.eye(3)))
        self.assertTrue(torch.all(dense[:2, 2:] == 0))
        self.assertTrue(torch.all(dense[2:, :2] == 0))

    def test_transform_manipulation(self):
        class TestModule(TransformedModuleMixin):
            def forward(self, x: Tensor) -> Tensor:
                return x

        module = TestModule()
        transform1 = ConstantMulTransform(torch.tensor(2.0))
        transform2 = CosineTransform()

        # Test append_transform
        append_transform(module, "test_transform", transform1)
        self.assertIs(module.test_transform, transform1)
        append_transform(module, "test_transform", transform2)
        self.assertIsInstance(module.test_transform, ChainedTransform)

        # Test prepend_transform
        module = TestModule()
        prepend_transform(module, "test_transform", transform1)
        self.assertIs(module.test_transform, transform1)
        prepend_transform(module, "test_transform", transform2)
        self.assertIsInstance(module.test_transform, ChainedTransform)

    def test_untransform_shape(self):
        shape = Size([2, 3])
        transform = Standardize(m=1)
        self.assertEqual(untransform_shape(transform, shape), Size([2, 3]))
        self.assertEqual(untransform_shape(None, shape), shape)


class TestGetters(BotorchTestCase):
    def setUp(self):
        super().setUp()
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            train_X = torch.rand(5, 2)
            train_Y = torch.randn(5, 2)

        self.models = []
        for num_outputs in (1, 2):
            self.models.append(
                SingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y[:, :num_outputs],
                    input_transform=Normalize(d=2),
                    outcome_transform=Standardize(m=num_outputs),
                )
            )

            self.models.append(
                SingleTaskVariationalGP(
                    train_X=train_X,
                    train_Y=train_Y[:, :num_outputs],
                    input_transform=Normalize(d=2),
                    outcome_transform=Standardize(m=num_outputs),
                )
            )

    def test_get_input_transform(self):
        for model in self.models:
            self.assertIs(get_input_transform(model), model.input_transform)

    def test_get_output_transform(self):
        for model in self.models:
            transform = get_output_transform(model)
            self.assertIsInstance(transform, OutcomeUntransformer)
            self.assertIs(transform.transform, model.outcome_transform)

    def test_get_train_inputs(self):
        for model in self.models:
            model.train()
            X = (
                model.model.train_inputs[0]
                if isinstance(model, SingleTaskVariationalGP)
                else model.train_inputs[0]
            )
            Z = model.input_transform(X)
            train_inputs = get_train_inputs(model, transformed=False)
            self.assertIsInstance(train_inputs, tuple)
            self.assertEqual(len(train_inputs), 1)

            self.assertTrue(X.equal(get_train_inputs(model, transformed=False)[0]))
            self.assertTrue(Z.equal(get_train_inputs(model, transformed=True)[0]))

            model.eval()
            self.assertTrue(X.equal(get_train_inputs(model, transformed=False)[0]))
            self.assertTrue(Z.equal(get_train_inputs(model, transformed=True)[0]))
            with delattr_ctx(model, "input_transform"), patch.object(
                model, "_original_train_inputs", new=None
            ):
                self.assertTrue(Z.equal(get_train_inputs(model, transformed=False)[0]))
                self.assertTrue(Z.equal(get_train_inputs(model, transformed=True)[0]))

        with self.subTest("test_model_list"):
            model_list = ModelListGP(*self.models)
            input_list = get_train_inputs(model_list)
            self.assertIsInstance(input_list, list)
            self.assertEqual(len(input_list), len(self.models))
            for model, train_inputs in zip(model_list.models, input_list):
                for a, b in zip(train_inputs, get_train_inputs(model)):
                    self.assertTrue(a.equal(b))

    def test_get_train_targets(self):
        for model in self.models:
            model.train()
            if isinstance(model, SingleTaskVariationalGP):
                F = model.model.train_targets
                Y = model.outcome_transform.untransform(F)[0].squeeze(dim=0)
            else:
                F = model.train_targets
                Y = OutcomeUntransformer(model.outcome_transform, model.num_outputs)(F)

            self.assertTrue(F.equal(get_train_targets(model, transformed=True)))
            self.assertTrue(Y.equal(get_train_targets(model, transformed=False)))

            model.eval()
            self.assertTrue(F.equal(get_train_targets(model, transformed=True)))
            self.assertTrue(Y.equal(get_train_targets(model, transformed=False)))
            with delattr_ctx(model, "outcome_transform"):
                self.assertTrue(F.equal(get_train_targets(model, transformed=True)))
                self.assertTrue(F.equal(get_train_targets(model, transformed=False)))

        with self.subTest("test_model_list"):
            model_list = ModelListGP(*self.models)
            target_list = get_train_targets(model_list)
            self.assertIsInstance(target_list, list)
            self.assertEqual(len(target_list), len(self.models))
            for model, Y in zip(self.models, target_list):
                self.assertTrue(Y.equal(get_train_targets(model)))
