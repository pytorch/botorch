#!/usr/bin/env python3

from typing import List, NamedTuple, Optional, Tuple

import torch
from torch import Tensor


class LBFGScompact(NamedTuple):
    """Container for compact representation of batch LBFGS approximations

        Q = 1/gamma I + N M^{-1} N^T
        Q^{-1} = gamma I + F E F^T

    where Q and Q^{-1} are a `b x n x n` batch of p.s.d. matrices, N and F are
    `b x n x 2m` batches of tall matrices, and M and E are `b x 2m x 2m` batches
    of matrices.

    Attributes:
        gamma: A `b x 1 x 1` batch of constants
        N: A `b x n x 2m` batch of tall matrices
        M_LU: The data for the batch LU factorization of M
        M_LU_pivots: The pivots for the batch LU factorization of M
        F: A `b x n x 2m` batch of tall matrices
        E: A `b x 2m x 2m` batch of square matrices
        FE: A `b x n x 2m` batch of tall matrices (batch matrix product of F and E)
    """

    gamma: Optional[Tensor] = None
    N: Optional[Tensor] = None
    M_LU: Optional[Tensor] = None
    M_LU_pivots: Optional[Tensor] = None
    F: Optional[Tensor] = None
    E: Optional[Tensor] = None
    FE: Optional[Tensor] = None


def batch_compact_lbfgs_updates(
    Slist: List[Tensor], Ylist: List[Tensor], B: bool = True, H: bool = True
) -> LBFGScompact:
    """Batch update compact representation of LBFGS approximations.

    TODO: This currently takes in a list of tensors and then creates new tensors,
            doubling storage requirements - Find a better way to be efficient
            in re-using memory for the previous s/y.

    Args:
        Slist: A list of `m` (batched) `b x n` tensors, where
            `S[k] = X_k - X_{k-1}``
        Ylist: A list of `m` (batched) `b x n` tensors, where
            `Y[k] = \grad X_k - \grad X_{k-1}`
        B: If True, return approximation of Hessian
        H: If True, return approximation of the inverse of the Hessian

    Returns:
        LBFGScompact: A NamedTuple containing the compact represenation of
            the Hessian approximation and its inverse
    """
    if not (B or H):  # no need to compute anything
        return LBFGScompact()
    gamma, S, Y = _batch_make_gamma_S_Y(Slist, Ylist)
    D_diag = torch.sum(S * Y, dim=1)
    L, R = _batch_make_L_R(S, Y)
    if B:
        N, M_LU, M_LU_pivots = _batch_make_B_compact(gamma, S, Y, D_diag, L)
    else:
        N, M_LU, M_LU_pivots = None, None, None
    if H:
        F, E = _batch_make_H_compact(gamma, S, Y, D_diag, R)
        FE = F.bmm(E)
    else:
        F, E, FE = None, None, None
    return LBFGScompact(gamma, N, M_LU, M_LU_pivots, F, E, FE)


def _batch_make_gamma_S_Y(
    Slist: List[Tensor], Ylist: List[Tensor]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Construct gamma, S and Y Tensors from history"""
    s = Slist[-1].unsqueeze(-1)
    y = Ylist[-1].unsqueeze(-1)
    gamma = torch.bmm(y.transpose(2, 1), s) / torch.bmm(y.transpose(2, 1), y)
    if torch.any(torch.isnan(gamma)):
        raise RuntimeError(
            "Unable to compute a L-BFGS approximation of the Hessian. "
            "Your function may not exhibit any local curvature."
        )
    S = torch.stack(Slist, dim=-1)
    Y = torch.stack(Ylist, dim=-1)
    return gamma, S, Y


def _batch_make_L_R(S: Tensor, Y: Tensor) -> Tuple[Tensor, Tensor]:
    """Generate L and R in batch mode"""
    LR = torch.bmm(S.transpose(-2, -1), Y)
    upper_indexer = torch.triu(torch.ones_like(LR[0]))
    L = LR * (1 - upper_indexer)
    R = LR * upper_indexer
    return L, R


def _batch_make_B_compact(
    gamma: Tensor, S: Tensor, Y: Tensor, D_diag: Tensor, L: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Form the compact representation of the L-BFGS approximation of the Hessian"""
    deltaS = S / gamma
    N = torch.cat([deltaS, Y], dim=-1)
    M = _batch_make_M(deltaS, S, D_diag, L)
    M_LU, M_LU_pivots = torch.btrifact(M)
    return N, M_LU, M_LU_pivots


def _batch_make_H_compact(
    gamma: Tensor, S: Tensor, Y: Tensor, D_diag: Tensor, R: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Form the compact representation of the L-BFGS approximation of the inverse
    of the Hessian"""
    gammaY = gamma * Y
    F = torch.cat([S, gammaY], dim=-1)
    E = _batch_make_E(gammaY, Y, D_diag, R)
    return F, E


def _batch_make_M(deltaS: Tensor, S: Tensor, D_diag: Tensor, L: Tensor) -> Tensor:
    """Generate M in batch mode"""
    b, m = S.shape[0], S.shape[-1]
    M = torch.zeros(b, 2 * m, 2 * m, dtype=S.dtype, device=S.device)
    M[:, :m, :m] = torch.bmm(deltaS.transpose(2, 1), S)
    M[:, :m, m:] = L
    M[:, m:, :m] = L.transpose(-1, -2)
    d_idx = torch.arange(m, 2 * m, dtype=torch.long, device=S.device)
    M[:, d_idx, d_idx] = -D_diag
    return M


def _batch_make_E(gammaY: Tensor, Y: Tensor, D_diag: Tensor, R: Tensor) -> Tensor:
    """Generate E in batch mode"""
    b, m = Y.shape[0], Y.shape[-1]
    E = torch.zeros(b, 2 * m, 2 * m, dtype=Y.dtype, device=Y.device)
    A = gammaY.transpose(1, 2).bmm(Y)
    d_idx = torch.arange(m, dtype=torch.long, device=Y.device)
    A[:, d_idx, d_idx] += D_diag
    Rinv = _batch_invert_triag(R)
    E[:, :m, :m] = Rinv.transpose(1, 2).bmm(A).bmm(Rinv)
    E[:, :m, m:] = -Rinv.transpose(1, 2)
    E[:, m:, :m] = -Rinv
    return E


def _batch_invert_triag(M: Tensor) -> Tensor:
    """Efficiently batch-invert an (upper) triagonal martrix

    Args:
        M: A `b x n x n` batch of `n x n` upper triagonal matrices

    Returns:
        A `b x n x n` batch of inverses (these will also be upper triagonal)
    """
    M_pivots = (
        torch.arange(start=1, end=M.shape[-1] + 1, dtype=torch.int, device=M.device)
        .expand(M.shape[0], -1)
        .contiguous()
    )
    eye = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
    return torch.btrisolve(eye.expand(M.shape[0], -1, -1), M, M_pivots)
