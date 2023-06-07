#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.kernels.orthogonal_additive_kernel import OrthogonalAdditiveKernel
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import MaternKernel
from gpytorch.lazy import LazyEvaluatedKernelTensor
from torch import nn, Tensor


class TestOrthogonalAdditiveKernel(BotorchTestCase):
    def test_kernel(self):
        n, d = 3, 5
        dtypes = [torch.float, torch.double]
        batch_shapes = [(), (2,), (7, 2)]
        for dtype in dtypes:
            tkwargs = {"dtype": dtype, "device": self.device}
            for batch_shape in batch_shapes:
                X = torch.rand(*batch_shape, n, d, **tkwargs)
                base_kernel = MaternKernel().to(device=self.device)
                oak = OrthogonalAdditiveKernel(
                    base_kernel,
                    dim=d,
                    second_order=False,
                    batch_shape=batch_shape,
                    **tkwargs,
                )
                KL = oak(X)
                self.assertIsInstance(KL, LazyEvaluatedKernelTensor)
                KM = KL.to_dense()
                self.assertIsInstance(KM, Tensor)
                self.assertEqual(KM.shape, (*batch_shape, n, n))
                self.assertEqual(KM.dtype, dtype)
                self.assertEqual(KM.device.type, self.device.type)
                # symmetry
                self.assertTrue(torch.allclose(KM, KM.transpose(-2, -1)))
                # positivity
                self.assertTrue(isposdef(KM))

                # testing differentiability
                X.requires_grad = True
                oak(X).to_dense().sum().backward()
                self.assertFalse(X.grad.isnan().any())
                self.assertFalse(X.grad.isinf().any())

                X_out_of_hypercube = torch.rand(n, d, **tkwargs) + 1
                with self.assertRaisesRegex(ValueError, r"x1.*hypercube"):
                    oak(X_out_of_hypercube, X).to_dense()

                with self.assertRaisesRegex(ValueError, r"x2.*hypercube"):
                    oak(X, X_out_of_hypercube).to_dense()

                with self.assertRaisesRegex(UnsupportedError, "does not support"):
                    oak.forward(x1=X, x2=X, last_dim_is_batch=True)

                oak_2nd = OrthogonalAdditiveKernel(
                    base_kernel,
                    dim=d,
                    second_order=True,
                    batch_shape=batch_shape,
                    **tkwargs,
                )
                KL2 = oak_2nd(X)
                self.assertIsInstance(KL2, LazyEvaluatedKernelTensor)
                KM2 = KL2.to_dense()
                self.assertIsInstance(KM2, Tensor)
                self.assertEqual(KM2.shape, (*batch_shape, n, n))
                # symmetry
                self.assertTrue(torch.allclose(KM2, KM2.transpose(-2, -1)))
                # positivity
                self.assertTrue(isposdef(KM2))
                self.assertEqual(KM2.dtype, dtype)
                self.assertEqual(KM2.device.type, self.device.type)

                # testing second order coefficient matrices are upper-triangular
                # and contain the transformed values in oak_2nd.raw_coeffs_2
                oak_2nd.raw_coeffs_2 = nn.Parameter(
                    torch.randn_like(oak_2nd.raw_coeffs_2)
                )
                C2 = oak_2nd.coeffs_2
                self.assertTrue(C2.shape == (*batch_shape, d, d))
                self.assertTrue((C2.tril() == 0).all())
                c2 = oak_2nd.coeff_constraint.transform(oak_2nd.raw_coeffs_2)
                i, j = torch.triu_indices(d, d, offset=1)
                self.assertTrue(torch.allclose(C2[..., i, j], c2))

                # second order effects change the correlation structure
                self.assertFalse(torch.allclose(KM, KM2))

                # check orthogonality of base kernels
                n_test = 7
                # inputs on which to evaluate orthogonality
                X_ortho = torch.rand(n_test, d, **tkwargs)
                # d x quad_deg x quad_deg
                K_ortho = oak._orthogonal_base_kernels(X_ortho, oak.z)

                # NOTE: at each random test input x_i and for each dimension d,
                # sum_j k_d(x_i, z_j) * w_j = 0.
                # Note that this implies the GP mean will be orthogonal as well:
                # mean(x) = sum_j k(x, x_j) alpha_j
                # so
                # sum_i mean(z_i) w_i
                # = sum_j alpha_j (sum_i k(z_i, x_j) w_i) // exchanging summations order
                # = sum_j alpha_j (0) // due to symmetry
                # = 0
                tol = 1e-5
                self.assertTrue(((K_ortho @ oak.w).squeeze(-1) < tol).all())


def isposdef(A: Tensor) -> bool:
    """Determines whether A is positive definite or not, by attempting a Cholesky
    decomposition. Expects batches of square matrices. Throws a RuntimeError otherwise.
    """
    _, info = torch.linalg.cholesky_ex(A)
    return not torch.any(info)
