# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.models.kernels.constant import ConstantKernel
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import AdditiveKernel, MaternKernel, ProductKernel, ScaleKernel
from gpytorch.lazy import LazyEvaluatedKernelTensor
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor


class TestConstantKernel(BotorchTestCase):
    def test_kernel(self):
        n, d = 3, 5
        dtypes = [torch.float, torch.double]
        batch_shapes = [(), (2,), (7, 2)]
        for dtype, batch_shape in itertools.product(dtypes, batch_shapes):
            tkwargs = {"dtype": dtype, "device": self.device}
            places = 6 if dtype == torch.float else 12
            X = torch.rand(*batch_shape, n, d, **tkwargs)

            constant_kernel = ConstantKernel(batch_shape=batch_shape)
            KL = constant_kernel(X)
            self.assertIsInstance(KL, LazyEvaluatedKernelTensor)
            KM = KL.to_dense()
            self.assertIsInstance(KM, Tensor)
            self.assertEqual(KM.shape, (*batch_shape, n, n))
            self.assertEqual(KM.dtype, dtype)
            self.assertEqual(KM.device.type, self.device.type)
            # standard deviation is zero iff KM is constant
            self.assertAlmostEqual(KM.std().item(), 0, places=places)

            # testing last_dim_is_batch
            with self.subTest(last_dim_is_batch=True):
                KD = constant_kernel(X, last_dim_is_batch=True).to(device=self.device)
                self.assertIsInstance(KD, LazyEvaluatedKernelTensor)
                KM = KD.to_dense()
                self.assertIsInstance(KM, Tensor)
                self.assertEqual(KM.shape, (*batch_shape, d, n, n))
                self.assertAlmostEqual(KM.std().item(), 0, places=places)
                self.assertEqual(KM.dtype, dtype)
                self.assertEqual(KM.device.type, self.device.type)

            # testing diag
            with self.subTest(diag=True):
                KD = constant_kernel(X, diag=True)
                self.assertIsInstance(KD, Tensor)
                self.assertEqual(KD.shape, (*batch_shape, n))
                self.assertAlmostEqual(KD.std().item(), 0, places=places)
                self.assertEqual(KD.dtype, dtype)
                self.assertEqual(KD.device.type, self.device.type)

            # testing diag and last_dim_is_batch
            with self.subTest(diag=True, last_dim_is_batch=True):
                KD = constant_kernel(X, diag=True, last_dim_is_batch=True)
                self.assertIsInstance(KD, Tensor)
                self.assertEqual(KD.shape, (*batch_shape, d, n))
                self.assertAlmostEqual(KD.std().item(), 0, places=places)
                self.assertEqual(KD.dtype, dtype)
                self.assertEqual(KD.device.type, self.device.type)

            # testing AD
            with self.subTest(requires_grad=True):
                X.requires_grad = True
                constant_kernel(X).to_dense().sum().backward()
                self.assertIsNone(X.grad)  # constant kernel is not dependent on X

            # testing algebraic combinations with another kernel
            base_kernel = MaternKernel().to(device=self.device)

            with self.subTest(additive=True):
                sum_kernel = base_kernel + constant_kernel
                self.assertIsInstance(sum_kernel, AdditiveKernel)
                self.assertAllClose(
                    sum_kernel(X).to_dense(),
                    base_kernel(X).to_dense() + constant_kernel.constant.unsqueeze(-1),
                )

            # product with constant is equivalent to scale kernel
            with self.subTest(product=True):
                product_kernel = base_kernel * constant_kernel
                self.assertIsInstance(product_kernel, ProductKernel)

                scale_kernel = ScaleKernel(base_kernel, batch_shape=batch_shape)
                scale_kernel.to(device=self.device)
                self.assertAllClose(
                    scale_kernel(X).to_dense(), product_kernel(X).to_dense()
                )

            # setting constant
            pies = torch.full_like(constant_kernel.constant, torch.pi)
            constant_kernel.constant = pies
            self.assertAllClose(constant_kernel.constant, pies)

            # specifying prior
            constant_kernel = ConstantKernel(
                constant_prior=GammaPrior(concentration=2.4, rate=2.7)
            )

            with self.assertRaisesRegex(
                TypeError, "Expected gpytorch.priors.Prior but got"
            ):
                ConstantKernel(constant_prior=1)
