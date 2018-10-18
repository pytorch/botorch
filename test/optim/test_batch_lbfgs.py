#! /usr/bin/env python3

import unittest

import torch
from botorch.optim.batch_lbfgs import (
    LBFGScompact,
    _batch_invert_triag,
    _batch_make_B_compact,
    _batch_make_E,
    _batch_make_gamma_S_Y,
    _batch_make_H_compact,
    _batch_make_L_R,
    _batch_make_M,
    batch_compact_lbfgs_updates,
)


class TestBatchInvertDiag(unittest.TestCase):
    def test_batch_invert_triag(self):
        r = 0.1 + torch.rand(2, 3, 1)
        R = torch.stack([torch.rand(3).diag() for _ in range(2)])
        R += r.bmm(r.transpose(-2, -1))
        # make upper diagonal
        R.mul_(torch.triu(torch.ones_like(R[0])))
        R_inv_naive = torch.stack([torch.inverse(r) for r in R])
        R_inv = _batch_invert_triag(R)
        self.assertTrue(torch.allclose(R_inv_naive, R_inv))


class TestBatchCompactLBFGSUpdates(unittest.TestCase):
    def setUp(self):
        self.n = 5
        self.num_batch = 2
        self.m_h = 3
        self.shape = torch.Size([self.num_batch, self.n, self.m_h])

    def _gen_test_Slist_Ylist(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.rand(
            self.num_batch, self.n, self.m_h + 1, device=device, requires_grad=True
        )
        (X ** 3).sum().backward()
        grad = X.grad
        Slist = [(X[:, :, i + 1] - X[:, :, i]).detach() for i in range(self.m_h)]
        Ylist = [(grad[:, :, i + 1] - grad[:, :, i]).detach() for i in range(self.m_h)]
        return Slist, Ylist

    def _gen_test_gamma_S_Y(self, cuda=False):
        Slist, Ylist = self._gen_test_Slist_Ylist(cuda=cuda)
        return _batch_make_gamma_S_Y(Slist, Ylist)

    def test_batch_make_gamma_S_Y(self, cuda=False):
        Slist, Ylist = self._gen_test_Slist_Ylist(cuda=cuda)
        gamma, S, Y = _batch_make_gamma_S_Y(Slist=Slist, Ylist=Ylist)

        s, y = Slist[-1], Ylist[-1]
        gamma_exp = torch.stack(
            [(s[b] * y[b]).sum() / (y[b] * y[b]).sum() for b in range(self.num_batch)]
        ).view(self.num_batch, 1, 1)
        self.assertTrue(all(torch.equal(S[..., i], Slist[i]) for i in range(self.m_h)))
        self.assertTrue(all(torch.equal(Y[..., i], Ylist[i]) for i in range(self.m_h)))
        self.assertTrue(torch.allclose(gamma, gamma_exp))

    def test_batch_make_gamma_S_Y_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_make_gamma_S_Y(cuda=True)

    def test_batch_make_L_R(self, cuda=False):
        gamma, S, Y = self._gen_test_gamma_S_Y(cuda=cuda)
        L, R = _batch_make_L_R(S=S, Y=Y)
        self.assertTrue(L.shape == torch.Size([self.num_batch, self.m_h, self.m_h]))
        self.assertTrue(R.shape == torch.Size([self.num_batch, self.m_h, self.m_h]))
        for b in range(self.num_batch):
            for i in range(self.m_h):
                for j in range(self.m_h):
                    if i > j:
                        l_exp = (S[b, :, i] * Y[b, :, j]).sum().item()
                        r_exp = 0.0
                    else:
                        l_exp = 0.0
                        r_exp = (S[b, :, i] * Y[b, :, j]).sum().item()
                    self.assertAlmostEqual(L[b, i, j].item(), l_exp, places=5)
                    self.assertAlmostEqual(R[b, i, j].item(), r_exp, places=5)

    def test_batch_make_L_R_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_make_L_R(cuda=True)

    def test_batch_make_M_E(self, cuda=False):
        gamma, S, Y = self._gen_test_gamma_S_Y(cuda=cuda)
        D_diag = torch.sum(S * Y, dim=1)
        L, R = _batch_make_L_R(S, Y)
        m = S.shape[-1]

        # check constuction of M
        M = _batch_make_M(S / gamma, S, D_diag, L)
        self.assertTrue(
            torch.allclose(M[:, :m, :m], torch.bmm((S / gamma).transpose(2, 1), S))
        )
        self.assertTrue(torch.equal(M[:, :m, m:], L))
        self.assertTrue(torch.equal(M.transpose(1, 2)[:, :m, m:], L))
        self.assertTrue(
            torch.equal(M[:, m:, m:], torch.stack([d.diag() for d in -D_diag]))
        )

        # check construction of E
        E = _batch_make_E(gamma * Y, Y, D_diag, R)
        Rinv = _batch_invert_triag(R)
        A = torch.bmm((gamma * Y).transpose(1, 2), Y)
        A.add_(torch.stack([d.diag() for d in D_diag]))
        self.assertTrue(
            torch.allclose(E[:, :m, :m], Rinv.transpose(1, 2).bmm(A).bmm(Rinv))
        )
        self.assertTrue(torch.equal(E[:, :m, m:], -Rinv.transpose(1, 2)))
        self.assertTrue(
            torch.equal(E.transpose(1, 2)[:, :m, m:], -Rinv.transpose(1, 2))
        )
        self.assertTrue(torch.all(E[:, m:, m:] == 0.0))

    def test_batch_make_M_E_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_make_M_E(cuda=True)

    def test_batch_make_B_H_compact(self, cuda=False):
        gamma, S, Y = self._gen_test_gamma_S_Y(cuda=cuda)
        D_diag = torch.sum(S * Y, dim=1)
        L, R = _batch_make_L_R(S, Y)

        N, M_LU, M_LU_pivots = _batch_make_B_compact(gamma, S, Y, D_diag, L)
        deltaS = S / gamma
        N_exp = torch.cat([deltaS, Y], dim=-1)
        M = _batch_make_M(deltaS, S, D_diag, L)
        M_LU_exp, M_LU_pivots_exp = torch.btrifact(M)
        self.assertTrue(torch.equal(N_exp, N))
        self.assertTrue(torch.allclose(M_LU_exp, M_LU))
        self.assertTrue(torch.equal(M_LU_pivots_exp, M_LU_pivots))

        F, E = _batch_make_H_compact(gamma, S, Y, D_diag, R)
        gammaY = gamma * Y
        F_exp = torch.cat([S, gammaY], dim=-1)
        E_exp = _batch_make_E(gammaY, Y, D_diag, R)
        self.assertTrue(torch.equal(F_exp, F))
        self.assertTrue(torch.equal(E_exp, E))

        # Ensure that H is in fact the inverse of B
        EYE = torch.eye(N.shape[1], device=N.device).repeat(N.shape[0], 1, 1)
        B = 1 / gamma * EYE - N.bmm(
            torch.btrisolve(N.transpose(1, 2), M_LU, M_LU_pivots)
        )
        H = gamma * EYE + F.bmm(E.bmm(F.transpose(1, 2)))

        self.assertTrue(torch.allclose(H.bmm(B), EYE, atol=1e-4))

    def test_batch_make_B_H_compact_cuda(self):
        if torch.cuda.is_available():
            self.test_batch_make_B_H_compact(cuda=True)

    def batch_compact_lbfgs_updates(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        S_test = torch.randn(2, 3, 4, device=device)
        Y_test = torch.randn(2, 3, 4, device=device)
        Slist = [S_test[:, :, i] for i in range(S_test.shape[-1])]
        Ylist = [Y_test[:, :, i] for i in range(Y_test.shape[-1])]

        result = batch_compact_lbfgs_updates(Slist=Slist, Ylist=Ylist)
        for attr in LBFGScompact._fields:
            self.assertIsNotNone(getattr(result, attr))

        result = batch_compact_lbfgs_updates(Slist=Slist, Ylist=Ylist, B=False)
        for attr in ("N", "M_LU", "M_LU_pivots"):
            self.assertIsNone(getattr(result, attr))

        result = batch_compact_lbfgs_updates(Slist=Slist, Ylist=Ylist, H=False)
        for attr in ("F", "E"):
            self.assertIsNone(getattr(result, attr))

    def batch_compact_lbfgs_updates_cuda(self):
        if torch.cuda.is_available():
            self.batch_compact_lbfgs_updates(cuda=True)

    def batch_compact_lbfgs_updates_no_local_curvature(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.rand(
            self.num_batch, self.n, self.m_h + 1, device=device, requires_grad=True
        )
        X.sum().backward()
        grad = X.grad
        Slist = [(X[:, :, i + 1] - X[:, :, i]).detach() for i in range(self.m_h)]
        Ylist = [(grad[:, :, i + 1] - grad[:, :, i]).detach() for i in range(self.m_h)]
        with self.assertRaises(ValueError):
            batch_compact_lbfgs_updates(Slist=Slist, Ylist=Ylist)

    def batch_compact_lbfgs_updates_no_local_curvature_cuda(self):
        if torch.cuda.is_available():
            self.batch_compact_lbfgs_updates_no_local_curvature(cuda=True)


if __name__ == "__main__":
    unittest.main()
