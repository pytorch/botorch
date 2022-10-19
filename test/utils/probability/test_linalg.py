#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy

import torch
from botorch.utils.probability.linalg import PivotedCholesky
from botorch.utils.testing import BotorchTestCase


class TestPivotedCholesky(BotorchTestCase):
    def setUp(self):
        super().setUp()
        n = 5
        with torch.random.fork_rng():
            torch.random.manual_seed(0)

            matrix = torch.randn(2, n, n)
            matrix = matrix @ matrix.transpose(-1, -2)

            diag = matrix.diagonal(dim1=-2, dim2=-1).sqrt()
            idiag = diag.reciprocal().unsqueeze(-1)

            piv_chol = PivotedCholesky(
                step=0,
                tril=(idiag * matrix * idiag.transpose(-2, -1)).tril(),
                perm=torch.arange(n)[None].expand(len(matrix), n).contiguous(),
                diag=diag.clone(),
            )

        self.diag = diag
        self.matrix = matrix
        self.piv_chol = piv_chol

        self.piv_chol.update_()
        self.piv_chol.pivot_(torch.tensor([2, 3]))
        self.piv_chol.update_()

    def test_update_(self):
        # Construct permuted matrices A
        n = self.matrix.shape[-1]
        A = (1 / self.diag).unsqueeze(-1) * self.matrix * (1 / self.diag).unsqueeze(-2)
        A = A.gather(-1, self.piv_chol.perm.unsqueeze(-2).repeat(1, n, 1))
        A = A.gather(-2, self.piv_chol.perm.unsqueeze(-1).repeat(1, 1, n))

        # Test upper left block
        L = torch.linalg.cholesky(A[..., :2, :2])
        self.assertTrue(L.allclose(self.piv_chol.tril[..., :2, :2]))

        # Test lower left block
        beta = torch.linalg.solve_triangular(L, A[..., :2:, 2:], upper=False)
        self.assertTrue(
            beta.transpose(-1, -2).allclose(self.piv_chol.tril[..., 2:, :2])
        )

        # Test lower right block
        schur = A[..., 2:, 2:] - beta.transpose(-1, -2) @ beta
        self.assertTrue(schur.tril().allclose(self.piv_chol.tril[..., 2:, 2:]))

    def test_pivot_(self):
        piv_chol = deepcopy(self.piv_chol)
        self.assertEqual(piv_chol.perm.tolist(), [[0, 2, 1, 3, 4], [0, 3, 2, 1, 4]])

        piv_chol.pivot_(torch.tensor([2, 3]))
        self.assertEqual(piv_chol.perm.tolist(), [[0, 2, 1, 3, 4], [0, 3, 1, 2, 4]])
        self.assertTrue(piv_chol.tril[0].equal(self.piv_chol.tril[0]))

        A = self.piv_chol.tril[1]
        B = piv_chol.tril[1]
        self.assertTrue(A[2:4, :2].equal(B[2:4, :2].roll(1, 0)))
        self.assertTrue(A[4:, 2:4].equal(B[4:, 2:4].roll(1, 1)))

    def test_concat(self):
        A = self.piv_chol.expand(2, 2)
        B = self.piv_chol.expand(1, 2)
        B = B.concat(B, dim=0)
        for key in ("tril", "perm", "diag"):
            self.assertTrue(getattr(A, key).equal(getattr(B, key)))

    def test_clone(self):
        self.piv_chol.diag.requires_grad_(True)
        try:
            other = self.piv_chol.clone()
            for key in ("tril", "perm", "diag"):
                a = getattr(self.piv_chol, key)
                b = getattr(other, key)
                self.assertTrue(a.equal(b))
                self.assertFalse(a is b)

            other.diag.sum().backward()
            self.assertTrue(self.piv_chol.diag.grad.eq(1).all())
        finally:
            self.piv_chol.diag.requires_grad_(False)

    def test_detach(self):
        self.piv_chol.diag.requires_grad_(True)
        try:
            other = self.piv_chol.detach()
            for key in ("tril", "perm", "diag"):
                a = getattr(self.piv_chol, key)
                b = getattr(other, key)
                self.assertTrue(a.equal(b))
                self.assertFalse(a is b)

            with self.assertRaisesRegex(RuntimeError, "does not have a grad_fn"):
                other.diag.sum().backward()

        finally:
            self.piv_chol.diag.requires_grad_(False)

    def test_expand(self):
        other = self.piv_chol.expand(3, 2)
        for key in ("tril", "perm", "diag"):
            a = getattr(self.piv_chol, key)
            b = getattr(other, key)
            self.assertEqual(b.shape[: -a.ndim], (3,))
            self.assertTrue(b._base is a)
