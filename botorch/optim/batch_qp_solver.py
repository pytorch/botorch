#! /usr/bin/env python3

import logging
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from .batch_lbfgs import LBFGScompact


DEFAULT_MAX_ITER = 25
DEFAULT_NOT_IMPROVED_LIM = 3
DEFAULT_TOL_FLOAT = 1e-7
DEFAULT_TOL_DOUBLE = 1e-10
SUCCESS_MSG = "Optimization terminated successfully after {i} iterations."


class Constraints(NamedTuple):
    """Container for constraint matrices.

    Describes the feasible set: `{x : Gx <= h, Ax = b}`, with shapes as follows:

        x: `num_batch x n x 1`
        G: `num_batch x p x n`
        h: `num_batch x p x 1`
        A: `num_batch x m x n`
        b: `num_batch x m x 1`

    where `num_batch` is the batch size.

    Notes:
        - if the problem does not have (in-)equality constraints, zero-sized
            tensors can be used, e.g. `A = torch.zeros(num_batch, 0, n)` and
            `b = torch.zeros(num_batch, 0, 1)` in case of no equality constraints
        - while this supports different constraints per batch, the sizes of the
            respective tesnors need to be the same across batches
    """

    G: Tensor  # num_batch x p x n
    h: Tensor  # num_batch x p x 1
    A: Tensor  # num_batch x m x n
    b: Tensor  # num_batch x m x 1


class Variables(NamedTuple):
    """Container for variables

    Collects primal (x), slack (s), and dual (lmbda, nu) variables

        x: `num_batch x n x 1`
        s: `num_batch x p x 1`
        lmbda: `num_batch x p x 1`
        nu: `num_batch x m x 1`
    """

    x: Tensor  # num_batch x n x 1
    s: Tensor  # num_batch x p x 1
    lmbda: Tensor  # num_batch x p x 1
    nu: Tensor  # num_batch x m x 1


class Residuals(NamedTuple):
    """Container for residuals"""

    primal: Tensor  # b x 1 x 1
    dual: Tensor  # b x 1 x 1
    mu: Tensor  # b x 1 x 1


class OptimizationResults(NamedTuple):
    """Container for optimization results

    optimizer: A Variable tuple containing the optimizer (if successful)
    residuals: A Residuls tuple containing primal & dual residuals and duality gap
    success: True if the optimization terminated successfully
    message: An more detailed info message about the status of the optimization
    debug: A dictionary with debug information (only if `debug=True` in the
        `options` arg of batch_solve_lbfgs_qp)

    """

    optimizer: Variables
    residuals: Residuals
    success: bool
    message: str
    debug: Optional[Any] = None


logger = logging.getLogger(__name__)


def expand_constraint_tensors(
    G: Tensor, h: Tensor, A: Tensor, b: Tensor, num_batch: int
) -> Constraints:
    """Expand non-batched constraint matrices into batch versions.

    Args:
        G: A `p x n` tensor
        h: A `p x 1` tensor
        A: A `m x n` tensor
        b: A `m x 1` tensor

    Returns:
        A Constraints tuple with the following attributes:
            G: A `num_batch x p x n` tensor
            h: A `num_batch x p x 1` tensor
            A: A `num_batch x m x n` tensor
            b: A `num_batch x m x 1` tensor

    If p = 0 (m = 0), there are no inequality (equality) constraints
    """
    n, nineq, neq = G.shape[1], G.shape[0], A.shape[0]
    G_batched = G.expand(num_batch, nineq, n)
    h_batched = h.expand(num_batch, nineq).unsqueeze(-1)
    A_batched = A.expand(num_batch, neq, n)
    b_batched = b.expand(num_batch, neq).unsqueeze(-1)
    return Constraints(G=G_batched, h=h_batched, A=A_batched, b=b_batched)


def batch_solve_lbfgs_qp(
    lbfgs: LBFGScompact,
    q: Tensor,
    constraints: Constraints,
    options: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> OptimizationResults:
    """Solve a QP specified by low-rank approximation of the Hessian in batch mode

    Args:
        lbfgs: A LBFGScompact tuple with the compact representationso of the
            L-BFGS approximations of the Hessian (B) and its inverse (H)
        q: A `num_batch x n x 1` tensor (the q-vector in the objective)
        constraints: A Constraints tuple with matrices G, h, A, b
        options: A Dict with solver options

    Returns:
        An OptimizationResults tuple with the following fields:
            - optimizer
            - residuals
            - success
            - message

    Reference:
        J. Mattingley and S. Boyd. CVXGEN: a code generator for embedded convex
        optimization. Optimization and Engineering, 13(1):1–27, Mar 2012.
    """
    options: Dict[str, Any] = options or {}
    max_iter = options.get("max_iter", DEFAULT_MAX_ITER)
    best_res: Optional[Residuals] = None
    n_not_impr: int = 0
    nineq = constraints.G.shape[1]

    # if there are no constraints just return the anlytical solution
    if constraints.G.nelement() + constraints.A.nelement() == 0:
        return solve_unconstrained(lbfgs=lbfgs, q=q)

    Z0 = _gen_Z0(lbfgs=lbfgs, constraints=constraints)
    v = find_initial_condition(lbfgs=lbfgs, q=q, constraints=constraints, Z0=Z0)
    res = compute_residuals(variables=v, q=q, lbfgs=lbfgs, constraints=constraints)

    for i in range(max_iter):
        converged, best_res, n_not_impr = check_convergence(
            res=res,
            best_res=best_res,
            nineq=nineq,
            n_not_impr=n_not_impr,
            options=options,
        )
        _print_info(iter=i, verbose=verbose, residuals=best_res)
        if converged:
            return OptimizationResults(
                optimizer=v,
                residuals=res,
                success=True,
                message=SUCCESS_MSG.format(i=i),
            )
        # take a primal-dual step
        v, msg = step(
            lbfgs=lbfgs, q=q, constraints=constraints, variables=v, mu=res.mu, Z0=Z0
        )
        if msg is not None:
            return OptimizationResults(
                optimizer=v, residuals=res, success=False, message=msg
            )
        res = compute_residuals(variables=v, q=q, lbfgs=lbfgs, constraints=constraints)
    return OptimizationResults(
        optimizer=v,
        residuals=res,
        success=False,
        message="Exceeded maximum number of iterations",
    )


def find_initial_condition(
    lbfgs: LBFGScompact,
    q: Tensor,
    constraints: Constraints,
    Z0: Optional[Tensor] = None,
) -> Variables:
    """Compute initial condition for the primal-dual interior point solver.

    Args:
        lbfgs: A LBFGScompact tuple with the compact representationso of the
            L-BFGS approximations of the Hessian (B) and its inverse (H)
        q: A `num_batch x n x 1` tensor (the q-vector in the objective)
        constraints: A Constraints tuple with matrices G, h, A, b
        Z0: Intermediate matrix generated by _gen_Z0. If None, generate from scratch

    Reference:
        J. Mattingley and S. Boyd. CVXGEN: a code generator for embedded convex
        optimization. Optimization and Engineering, 13(1):1–27, Mar 2012.
    """
    G, h, A, b = constraints
    num_batch, nineq, neq = G.shape[0], G.shape[1], A.shape[1]

    if Z0 is None:
        Z0 = _gen_Z0(lbfgs=lbfgs, constraints=constraints)
    D_diag = torch.ones(num_batch, nineq, dtype=Z0.dtype, device=Z0.device)
    Z_LU, Z_LU_pivots, info = _gen_factor_Z(Z0=Z0, D_diag=D_diag, neq=neq, nineq=nineq)

    beta_x = -(lbfgs.gamma * q + lbfgs.FE.bmm(lbfgs.F.transpose(1, 2).bmm(q)))

    alpha_lmbda = h - G.bmm(beta_x)
    alpha_nu = b - A.bmm(beta_x)

    alpha_nu_lmbda = torch.cat([alpha_nu, alpha_lmbda], dim=1)
    beta_nu_lmbda = torch.btrisolve(-alpha_nu_lmbda, Z_LU, Z_LU_pivots)
    beta_lmbda = beta_nu_lmbda[:, neq:]
    beta_nu = nu_0 = beta_nu_lmbda[:, :neq]

    z = G.transpose(1, 2).bmm(beta_lmbda) + A.transpose(1, 2).bmm(beta_nu)
    x_0 = beta_x - (lbfgs.gamma * z + lbfgs.FE.bmm(lbfgs.F.transpose(1, 2).bmm(z)))

    if nineq == 0:
        s_0 = torch.empty(num_batch, 0, 1, dtype=x_0.dtype, device=x_0.device)
        lmbda_0 = torch.empty(num_batch, 0, 1, dtype=x_0.dtype, device=x_0.device)
    else:
        lmbda = G.bmm(x_0) - h
        alpha_p = lmbda.max(dim=1, keepdim=True)[0]
        s_0 = -lmbda + (1 + alpha_p) * (alpha_p >= 0).type_as(lmbda)
        alpha_d = (-lmbda).max(dim=1, keepdim=True)[0]
        lmbda_0 = lmbda + (1 + alpha_d) * (alpha_d >= 0).type_as(lmbda)

    return Variables(x=x_0, s=s_0, lmbda=lmbda_0, nu=nu_0)


def check_convergence(
    res: Residuals,
    best_res: Residuals,
    nineq: int,
    n_not_impr: int,
    options: Dict[str, Any],
) -> Tuple[bool, Residuals, int]:
    """Check convergence"""
    if best_res is not None:
        r = res.primal + res.dual + nineq * res.mu
        rb = best_res.primal + best_res.dual + nineq * best_res.mu
        impr = (r < rb).type_as(r)
        new_n_not_impr = n_not_impr + int(impr.sum().item() == 0)
        nbr: Dict[str, Tensor] = {
            k: impr * getattr(res, k) + (1 - impr) * getattr(best_res, k)
            for k in res._fields
        }
        new_best_res = Residuals(**nbr)
    else:
        new_best_res = res
        new_n_not_impr = 0
    r = new_best_res.primal + new_best_res.dual + nineq * new_best_res.mu
    default_tol = DEFAULT_TOL_FLOAT if r.dtype == torch.float else DEFAULT_TOL_DOUBLE
    converged = (
        new_n_not_impr == options.get("n_not_impr", DEFAULT_NOT_IMPROVED_LIM)
        or r.max().item() < options.get("tol", default_tol)
        or new_best_res.mu.min().item() > 1e32
    )
    return converged, new_best_res, new_n_not_impr


def step(
    lbfgs: LBFGScompact,
    q: Tensor,
    constraints: Constraints,
    variables: Variables,
    mu: Tensor,
    Z0: Optional[Tensor],
) -> Tuple[Variables, Optional[str]]:
    """Take a primal/dual step

    Internally, solves the KKT system to determine both affine scaling and
    correction-and-centering directions.

    Args:
        lbfgs: A LBFGScompact tuple with the compact representationso of the
            L-BFGS approximations of the Hessian (B) and its inverse (H)
        q : A num_batch x n x 1 batch of gradients
        constraints: A Constraints tuple with matrices G, h, A, b
        variables: A Variables tuple with primal and dual variables x, z, lmbda, nu
        Z0: A num_batch x (m+p) x (m+p) batch of tensors used in solving the KKT system

    Returns:
        variables: A Variables tuple containing the updated primal and dual variables
        message: None if step was successful, otherwise an error message
    """
    G, h, A, b = constraints
    num_batch, nineq, neq = G.shape[0], G.shape[1], A.shape[1]
    x, s, lmbda, nu = variables

    if Z0 is None:
        Z0 = _gen_Z0(lbfgs=lbfgs, constraints=constraints)
    D_diag = (s / lmbda).squeeze(-1)
    Z_LU, Z_LU_pivots, info = _gen_factor_Z(Z0=Z0, D_diag=D_diag, neq=neq, nineq=nineq)
    if info.abs().sum() != 0:
        msg = "LU factorization of Z matrix failed"
        logger.warning(msg)
        return variables, msg

    # TODO: We're recomputing a bunch of MVMs here that we've computed for the
    # residuals above. I'm sure we ca optimize this further.

    # compute affine scaling directions
    D_x_aff, D_s_aff, D_lmbda_aff, D_nu_aff = affine_scaling_directions(
        lbfgs=lbfgs,
        q=q,
        constraints=constraints,
        Z_LU=Z_LU,
        Z_LU_pivots=Z_LU_pivots,
        variables=variables,
    )

    # compute centering plus corrector directions
    D_x_cc, D_s_cc, D_lmbda_cc, D_nu_cc = centering_plus_corrector_directions(
        lbfgs=lbfgs,
        q=q,
        constraints=constraints,
        Z_LU=Z_LU,
        Z_LU_pivots=Z_LU_pivots,
        variables=variables,
        D_s_aff=D_s_aff,
        D_lmbda_aff=D_lmbda_aff,
        mu=mu,
    )

    # compute overall directions
    D_x = D_x_aff + D_x_cc
    D_s = D_s_aff + D_s_cc
    D_lmbda = D_lmbda_aff + D_lmbda_cc
    D_nu = D_nu_aff + D_nu_cc

    # Find appropriate step size
    if nineq == 0:
        alpha = torch.zeros(num_batch, 1, 1, dtype=D_x.dtype, device=D_x.device)
    else:
        alpha_s = _find_alpha(s, D_s)
        alpha_lmbda = _find_alpha(lmbda, D_lmbda)
        alpha = torch.min(
            torch.ones_like(alpha_s), 0.99 * torch.min(alpha_s, alpha_lmbda)
        )

    # update variables and return
    new_variables = Variables(
        x=x + alpha * D_x,
        s=s + alpha * D_s,
        lmbda=lmbda + alpha * D_lmbda,
        nu=nu + alpha * D_nu,
    )
    return new_variables, None


def affine_scaling_directions(
    lbfgs: LBFGScompact,
    q: Tensor,
    constraints: Constraints,
    Z_LU: Tensor,
    Z_LU_pivots: Tensor,
    variables: Variables,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute affine scaling directions.

    Args:
        lbfgs: A LBFGScompact tuple with the compact representationso of the
            L-BFGS approximations of the Hessian (B) and its inverse (H)
        q: A `num_batch x n x 1` tensor (the q-vector in the objective)
        constraints: A Constraints tuple with matrices G, h, A, b
        Z_LU: The batch LU factorization returned by torch.btrifact
        Z_LU_pivots: The pivots associated with Z_LU
        variables: A Variables tuple containing x, s, lmbda, nu

    Reference:
        J. Mattingley and S. Boyd. CVXGEN: a code generator for embedded convex
        optimization. Optimization and Engineering, 13(1):1–27, Mar 2012.
    """
    x, s, lmbda, nu = variables.x, variables.s, variables.lmbda, variables.nu
    G, h, A, b = constraints
    neq = A.shape[1]

    r_lmbda = -(G.bmm(x) + s - h)
    r_nu = -(A.bmm(x) - b)

    ztilde = A.transpose(1, 2).bmm(nu) + G.transpose(1, 2).bmm(lmbda) + q
    beta_x = -x - (
        lbfgs.gamma * ztilde + lbfgs.FE.bmm(lbfgs.F.transpose(1, 2).bmm(ztilde))
    )
    beta_s = -s

    alpha_lmbda = r_lmbda - (G.bmm(beta_x) + beta_s)
    alpha_nu = r_nu - A.bmm(beta_x)

    alpha_nu_lmbda = torch.cat([alpha_nu, alpha_lmbda], dim=1)
    beta_nu_lmbda = torch.btrisolve(-alpha_nu_lmbda, Z_LU, Z_LU_pivots)
    beta_nu = D_nu = beta_nu_lmbda[:, :neq]
    beta_lmbda = D_lmbda = beta_nu_lmbda[:, neq:]

    z = G.transpose(1, 2).bmm(beta_lmbda) + A.transpose(1, 2).bmm(beta_nu)
    D_x = beta_x - (lbfgs.gamma * z + lbfgs.FE.bmm(lbfgs.F.transpose(1, 2).bmm(z)))
    D_s = beta_s - (s / lmbda) * beta_lmbda

    return D_x, D_s, D_lmbda, D_nu


def centering_plus_corrector_directions(
    lbfgs: LBFGScompact,
    q: Tensor,
    constraints: Constraints,
    Z_LU: Tensor,
    Z_LU_pivots: Tensor,
    variables: Variables,
    D_s_aff: Tensor,
    D_lmbda_aff: Tensor,
    mu: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute centering-plus-corrector directions.

    Args:
        lbfgs: A LBFGScompact tuple with the compact representationso of the
            L-BFGS approximations of the Hessian (B) and its inverse (H)
        q: A `num_batch x n x 1` tensor (the q-vector in the objective)
        constraints: A Constraints tuple with matrices G, h, A, b
        Z_LU: The batch LU factorization returned by torch.btrifact
        Z_LU_pivots: The pivots associated with Z_LU
        variables: A Variables tuple containing x, s, lmbda, nu
        D_s_aff: The affine scaling directions for the slack variable s, a
            `num_batch x p x 1` tensor as returned by affine_scaling_directions
        D_lmbda_aff: The affine scaling directions for the dual variable lmbda,
            a `num_batch x p x 1` tensor as returned by affine_scaling_directions
        mu: The duality gap as a `num_batch x 1 x 1` tensor

    Reference:
        J. Mattingley and S. Boyd. CVXGEN: a code generator for embedded convex
        optimization. Optimization and Engineering, 13(1):1–27, Mar 2012.
    """
    s, lmbda = variables.s, variables.lmbda
    G, A = constraints.G, constraints.A
    num_batch, nineq, neq = G.shape[0], G.shape[1], A.shape[1]

    # compute centering-plus-corrector direction
    if nineq == 0:
        beta_s = torch.empty(num_batch, 0, 1, dtype=s.dtype, device=s.device)
    else:
        alpha_s = _find_alpha(s, D_s_aff).clamp_max_(1.0)
        alpha_lmbda = _find_alpha(lmbda, D_lmbda_aff).clamp_max_(1.0)
        alpha = torch.min(alpha_s, alpha_lmbda)
        sigma_num = (
            (s + alpha * D_s_aff).transpose(1, 2).bmm(lmbda + alpha * D_lmbda_aff)
        )
        sigma = (sigma_num / (mu * nineq)) ** 3
        beta_s = (sigma * mu - D_s_aff * D_lmbda_aff) / lmbda

    alpha_lmbda = -beta_s
    alpha_nu = torch.zeros(num_batch, neq, 1, dtype=s.dtype, device=s.device)

    alpha_nu_lmbda = torch.cat([alpha_nu, alpha_lmbda], dim=1)
    beta_nu_lmbda = torch.btrisolve(-alpha_nu_lmbda, Z_LU, Z_LU_pivots)
    beta_nu = D_nu = beta_nu_lmbda[:, :neq]
    beta_lmbda = D_lmbda = beta_nu_lmbda[:, neq:]

    z = G.transpose(1, 2).bmm(beta_lmbda) + A.transpose(1, 2).bmm(beta_nu)

    # beta_x is zero, so we can leave that out here
    D_x = -lbfgs.gamma * z - lbfgs.FE.bmm(lbfgs.F.transpose(1, 2).bmm(z))
    D_s = beta_s - (s / lmbda) * beta_lmbda

    return D_x, D_s, D_lmbda, D_nu


def compute_residuals(
    variables: Variables, q: Tensor, lbfgs: LBFGScompact, constraints: Constraints
) -> Residuals:
    G, h, A, b = constraints
    x, s, lmbda, nu = variables
    nineq = G.shape[1]

    # compute duality gap
    mu = s.transpose(1, 2).bmm(lmbda) / nineq

    # compute primal residuals
    primal_ineq = (G.bmm(x) + s - h).norm(p=2, dim=1, keepdim=True)
    primal_eq = (A.bmm(x) - b).norm(p=2, dim=1, keepdim=True)

    # compute dual residual
    if lbfgs.N is not None:
        Minv_NT_x = torch.btrisolve(
            lbfgs.N.transpose(1, 2).bmm(x), lbfgs.M_LU, lbfgs.M_LU_pivots
        )
        Qx = x / lbfgs.gamma + lbfgs.N.bmm(Minv_NT_x)
    else:
        Qx = x / lbfgs.gamma
    dual = (Qx + q + A.transpose(1, 2).bmm(nu) + G.transpose(1, 2).bmm(lmbda)).norm(
        p=2, dim=1, keepdim=True
    )

    return Residuals(primal=primal_ineq + primal_eq, dual=dual, mu=mu)


def solve_unconstrained(lbfgs: LBFGScompact, q: Tensor) -> OptimizationResults:
    """Compute the analytical solution to the unconstrained problem"""
    num_batch = q.shape[0]
    x = -(lbfgs.gamma * q + lbfgs.FE.bmm(lbfgs.F.transpose(1, 2).bmm(q)))
    empty_var = torch.empty(num_batch, 0, 1, dtype=q.dtype, device=q.device)
    zero_res = torch.zeros(num_batch, 1, 1, dtype=q.dtype, device=q.device)
    return OptimizationResults(
        optimizer=Variables(x=x, s=empty_var, lmbda=empty_var, nu=empty_var),
        residuals=Residuals(primal=zero_res, dual=zero_res, mu=zero_res),
        success=True,
        message="Unconstrained problem, found analytical solution",
    )


def _gen_Z0(lbfgs: LBFGScompact, constraints: Constraints) -> Tensor:
    """Generate a helper matrix used in solving the KKT system.

    Z0 does not change between iterations of the primal-dual algorithm for
    solving the QP for a given Hessian approximation, and hence needs to be
    constructed only onceself.

    TODO: Replace this with the partial factorization technique from qpth and
    run some benchmarks comparing the two.
    """
    G, A = constraints.G, constraints.A
    AF = A.bmm(lbfgs.F)
    GF = G.bmm(lbfgs.F)
    E_Ft_At = lbfgs.E.bmm(AF.transpose(1, 2))
    E_Ft_Gt = lbfgs.E.bmm(GF.transpose(1, 2))
    Z0_11 = lbfgs.gamma * A.bmm(A.transpose(1, 2)) + AF.bmm(E_Ft_At)
    Z0_12 = lbfgs.gamma * A.bmm(G.transpose(1, 2)) + AF.bmm(E_Ft_Gt)
    Z0_22 = lbfgs.gamma * G.bmm(G.transpose(1, 2)) + GF.bmm(E_Ft_Gt)
    Z0 = torch.cat(
        [
            torch.cat([Z0_11, Z0_12], dim=-1),
            torch.cat([Z0_12.transpose(1, 2), Z0_22], dim=-1),
        ],
        dim=1,
    )
    return Z0


def _gen_factor_Z(
    Z0: Tensor, D_diag: Tensor, neq: int, nineq: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate Z matrix from Z0 and LU-factorize it

        |Z_11  Z_12|   |Z0_11  Z0_12|    |0    0   |
        |Z_21  Z_22| = |Z0_21  Z0_22| +  |0  D_diag|

    Args:
        Z0: Intermediate matrix generated by _gen_Z0
        lbfgs: A LBFGScompact tuple with the compact representationso of the
            L-BFGS approximations of the Hessian (B) and its inverse (H)
        constraints: A Constraints tuple with matrices G, h, A, b
        neq: The number of linear equality constraints (i.e. A.shape[1])
        nineq: The number of linear inequality constraints (i.e. G.shape[1])

    Returns:
        Tuple of tensors returned by torch.btrifact representing the batch LU-
            factorization of the generated matrix Z

    This can be replaced by doing partial factorizations as qpth does. However,
    note that this will only yield siginifcant savings if neq >> nineq.
    """
    Z = Z0.clone()
    idx = torch.arange(neq, neq + nineq, dtype=torch.long, device=Z.device)
    Z[:, idx, idx] += D_diag
    Z_LU, Z_LU_pivots, info = torch.btrifact_with_info(Z)
    return Z_LU, Z_LU_pivots, info


def _find_alpha(v: Tensor, Delta_v: Tensor) -> Tensor:
    """Compute sup {alpha >=0 | v + alpha Delta_v >=0}

    Input:
        v: A `num_batch x k x 1` batch tensor
        Delta_v: A `num_batch x k x 1` batch tensor

    Returns:
        a: A `num_batch x 1 x 1` batch tensor s.t.
            a_i = sup {alpha >=0 | v_i + alpha Delta_v_i >=0}
    """
    neg = (Delta_v < 0).type_as(v)
    a = neg * (-v / Delta_v) + (1 - neg) * torch.ones_like(v)
    return a.min(dim=1, keepdim=True)[0].clamp_min_(0.0)


def _print_info(iter: int, verbose: bool, residuals: Residuals) -> None:
    if not verbose:
        return
    print(
        f"iteration: {iter}, "
        f"primal residual: {residuals.primal.mean():.5e}, "
        f"dual residual: {residuals.dual.mean():.5e}, "
        f"duality gap: {residuals.mu.mean():.5e}"
    )
