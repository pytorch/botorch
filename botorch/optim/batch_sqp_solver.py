#!/usr/bin/env python3

import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor

from .batch_lbfgs import LBFGScompact, batch_compact_lbfgs_updates
from .batch_qp_solver import Constraints, batch_solve_lbfgs_qp


logger = logging.getLogger()


class OptimizationResults(NamedTuple):
    """Results of the batched L-BFGS-SQP optimization problem."""

    Xopt: Tensor
    diagnostics: Dict[str, float]
    debug_traces: Dict[str, Any]


def batch_solve_lbfgs_sqp(
    f: Callable[[Tensor], Tensor],
    X0: Tensor,
    constraints: Constraints,
    options: Optional[Dict[str, Union[bool, float]]] = None,
    gradient: Optional[Callable[[Tensor], Tensor]] = None,
    hessian: Optional[Callable[[Tensor], Tensor]] = None,
) -> OptimizationResults:
    """Batched Sequential Quadratic Programming with L-BFGS Hessian approximations.

    Args:
        f (Callable): The function(s) to minimize. Maps a `r x n`-dimensional
            Tensor of inputs to an `r`-dimensional output Tensor, where `r` is
            the number of batches and `n` is number of variables. That is, f is
            of the form f(X) = [f_1(X[1,:]), f_2(X[2,:]), ..., f_r(X[r,:])].
            It is not required that f_i = f_j.
        X0 (Tensor): `r x n`-dimensional Tensor of initial conditions, with
            X0[l, :] the `l`-th initial condition. If X0 lives on the GPU, then
            the full optimization will be performed on the GPU as well.
        constraints (Constraints): The constraints to be used in the optimization.
            Can be different between batches as long as the shapes are the same.
        options (dict, optional): Dictionary of options for the optimization
            algorithm. TODO: Document all options & default values.
        gradient (Callable, optional): The gradient of f. If not provided, use
            pytorch's autograd functionality to compute gradients. Maps `r x n`-
            dimensional Tensors to `r x n`-dimensional Tensors.
        hessian (Callable, optional): The Hessian of f. If not provided, use
            L-BFGS updates to construct an approximation of the Hessian. Maps
            `r x n`-dimensional Tensors to `r x n x n`-dimensional Tensors, where
            hessian(X)[i,:,:] is the Hessian of f_i w.r.t. X[i,:].

    Returns:
        OptimizationResults: NamedTuple with the following fields:
            Xopt (Tensor): `r x n`-dimensional Tensor of optimizers, with
                X[l, :] the optimizer computed starting from the `l`-th initial
                condition X0[l, :].
            diagnostics (dict, optional): Dictionary of diagnostic information
                on the run of the optimzation algorithm.
            debug (dict, optional): Dictionary with additional debugging info.
                Only returned if `debug` is set to True in the options.

    """
    if gradient is not None or hessian is not None:
        raise NotImplementedError(
            "User-supplied gradient and Hessian not yet supported"
        )
    start_time = time.time()
    options = options or {}
    debug = options.get("debug", False)
    counter: Dict[str, float] = defaultdict(lambda: 0)

    num_batch = X0.shape[0]
    m_h = int(options.get("history_length", 50))

    if constraints.G.shape[0] != num_batch:
        raise ValueError(
            "Incompatible batch sizes: constraints={c}, X0={x}".format(
                c=constraints.G.shape[0], x=num_batch
            )
        )

    # this will hold the final results
    X = X0.detach().clone()

    # save a bunch of stuff when in debug mode
    debug_traces = defaultdict(list)

    # these hold L-BFGS components
    Slist: List[Tensor] = []
    Ylist: List[Tensor] = []
    lbfgs = None  # we initially don't have any curvature information

    # binary vector for indicating active (not yet converged elements)
    n_active = X.shape[0]
    Active = torch.ones(n_active, dtype=torch.uint8, device=X.device)

    # we'll only work with the non-converged batches
    Xa = X.clone().requires_grad_(True)

    # now take steps
    while True:
        # compute function value and gradient
        Xa, FXa, gradFXa = batch_get_zero_grad(
            f=lambda X: f(X, Active), X=Xa, counter=counter
        )

        # store current values for next iteration (L-BFGS, convergence checks)
        if counter["n_iter"] == 0:
            Xa_prev = Xa.clone()
            FXa_prev = FXa.clone()
            gradFXa_prev = gradFXa.clone()
        else:
            max_iter = options.get("max_iter", 50)
            if counter["n_iter"] >= max_iter:
                print(
                    "Maximum number of iterations ({max_iter}) exceeded. "
                    "Returning current best iterates."
                )
                break

        # check convergence of the batches
        active, any_converged = batch_check_convergence(
            gradFX=gradFXa,
            FX=FXa,
            # use large value for FX_prev in iteration 0 to avoid case distinction
            FX_prev=FXa_prev if counter["n_iter"] > 0 else FXa + 1e6,
            options=options,
        )

        # update the vector of active batches
        if any_converged:
            Active[Active] = active
            n_active = Active.sum().item()

        if n_active == 0:
            if options.get("verbose", False):
                print("Optimization successfully terminated for all batches.")
            break

        if options.get("verbose", False):
            print(
                "Iteration {i}, batches active: {n_active}/{num_batch}".format(
                    i=counter["n_iter"] + 1,
                    n_active=n_active,
                    num_batch=num_batch,
                    options=options,
                )
            )

        # subset to active batches if any converged
        if any_converged:
            Xa = Xa[active]
            FXa = FXa[active]
            gradFXa = gradFXa[active]
            constraints = Constraints(
                G=constraints.G[active],
                h=constraints.h[active],
                A=constraints.A[active],
                b=constraints.b[active],
            )
            Slist = [s[active] for s in Slist]
            Ylist = [y[active] for y in Ylist]

        # update L-BFGS components and Hessian approximations
        if counter["n_iter"] > 0:
            t_lbfgs = time.time()
            lbfgs, lbfgs_diagnostics = _batch_update_lfbfgs_components(
                Slist=Slist,
                Ylist=Ylist,
                X=Xa,
                gradFX=gradFXa,
                X_prev=Xa_prev[active],
                gradFX_prev=gradFXa_prev[active],
                m_h=m_h,
            )
            counter["t_lbfgs"] += time.time() - t_lbfgs

        # save current iterate and gradient for use in next loop
        Xa_prev = Xa.clone()
        FXa_prev = FXa.clone()
        gradFXa_prev = gradFXa.clone()

        # compute step (feasible by construction)
        step, step_diagnostics = projected_quasi_newton_step(
            f=f,
            X=Xa,
            FX=FXa,
            gradFX=gradFXa,
            lbfgs=lbfgs,
            constraints=constraints,
            Active=Active,
            n_iter=counter["n_iter"],
            options=options,
        )

        # update diagnostics
        for k, v in step_diagnostics.items():
            counter[k] += v

        # take the step
        X_next = Xa + step
        X[Active] = X_next
        Xa = X_next.clone().requires_grad_(True)
        counter["n_iter"] += 1

        if debug:
            debug_traces["active_batches"].append(Active.clone())
            debug_traces["X"].append(X.clone())
            debug_traces["fval"].append(FXa.clone())
            debug_traces["step_length"].append(step.clone())

    return OptimizationResults(
        Xopt=X,
        diagnostics={"wall_time": time.time() - start_time, **counter},
        debug_traces=dict(debug_traces),
    )


def batch_get_zero_grad(
    f: Callable[[Tensor], Tensor], X: Tensor, counter: Dict[str, float]
) -> Tensor:
    """Get the gradient of fX w.r.t. X and zero it in a batched fashion"""
    t_f = time.time()
    FX = f(X)
    counter["t_f"] += time.time() - t_f
    t_grad = time.time()
    torch.autograd.backward([*FX])
    gradFX = X.grad.clone()
    X.grad.zero_()
    counter["t_grad"] += time.time() - t_grad
    return X.detach(), FX.detach(), gradFX


def batch_check_convergence(
    gradFX: Tensor, FX: Tensor, FX_prev: Tensor, options: Dict[str, float]
) -> Tensor:
    """Batch-check convergence (grad. norm, abs. and rel. reductions in f)"""
    # norm of the gradient
    large_grad = gradFX.norm(p=2, dim=-2).squeeze(-1) > options.get("grad_eps", 1e-8)
    # absolute reduction in function value
    deltaF = FX_prev - FX
    large_abs_red = deltaF > options.get("abs_red_eps", 1e-8)
    # relative reduction in function value
    fmin_eps = torch.full_like(FX, options.get("fmin_eps", 1e-7))
    rel_red = deltaF / torch.max(FX.abs(), fmin_eps)
    large_rel_red = rel_red > options.get("rel_red_eps", 1e-9)
    # active if gradient or reduction (abs or rel) is small
    active = large_grad & (large_abs_red | large_rel_red)
    any_converged = active.sum().item() < active.shape[0]
    return active, any_converged


def projected_quasi_newton_step(
    f: Callable[[Tensor], Tensor],
    X: Tensor,
    FX: Tensor,
    gradFX: Tensor,
    lbfgs: LBFGScompact,
    constraints: Constraints,
    Active: torch.Tensor,
    n_iter: int,
    options: Dict[str, float],
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute a (feasible) step using backtracking line search.
    TODO: Improve line search method.
    """
    # compute descent direction
    t_D = time.time()
    if lbfgs is None or n_iter < options.get("gradient_steps", 3):
        D = _get_initial_descent_direction(X, gradFX, constraints, options)
    else:
        D = _get_descent_direction(X, gradFX, lbfgs, constraints, options)
    t_D = time.time() - t_D

    # perform line search (any alpha <=1 will be feasible by construction)
    alpha, ls_diag = _batch_backtracking_ls(f, X, FX, gradFX, D, Active, options)

    # take the step (again, there should be a better way to do this...)
    step = alpha * D
    diagnostics = {"n_iter_ls": ls_diag["n_iter"], "t_D": t_D, "t_ls": ls_diag["dt"]}
    return step, diagnostics


def _get_descent_direction(
    X: Tensor,
    gradFX: Tensor,
    lbfgs: LBFGScompact,
    constraints: Constraints,
    options: Dict[str, float],
) -> Tensor:
    """Get descent direction by solving a Quadratic Program."""
    t_constraints = _adjust_constraints(constraints=constraints, X=X)
    res = batch_solve_lbfgs_qp(
        lbfgs=lbfgs,
        q=gradFX,
        constraints=t_constraints,
        options=options,
        verbose=options.get("qp_verbose", False),
    )
    if not res.success:
        raise RuntimeError("Failed to get descent direction")
    Z_star = res.optimizer.x
    # X_star = Z_star + X, D = X_star - X, return D => return Z_star
    return Z_star


def _adjust_constraints(constraints: Constraints, X: Tensor) -> Constraints:
    """ """
    return Constraints(
        G=constraints.G,
        h=constraints.h - constraints.G.bmm(X),
        A=constraints.A,
        b=constraints.b - constraints.A.bmm(X),
    )


def _get_initial_descent_direction(
    X: Tensor, gradFX: Tensor, constraints: Constraints, options: Dict[str, float]
) -> Tensor:
    """Get descent direction from gradient alone."""
    D = -gradFX / gradFX.norm(p=2, dim=1, keepdim=True)
    eta_0 = options.get("eta_0", 1.0)
    # if the problem is unconstrained there is no need to project anything
    if constraints.G.nelement() + constraints.A.nelement() == 0:
        return eta_0 * D
    # we solve a QP to project on the feasible set
    num_batch, n = constraints.G.shape[0], constraints.G.shape[-1]
    lbfgs = _gen_scaled_identity_H_lbfgs(
        gamma=0.5, num_batch=num_batch, n=n, dtype=X.dtype, device=X.device
    )
    q = -2 * (eta_0 * D + X)
    res = batch_solve_lbfgs_qp(
        lbfgs=lbfgs,
        q=q,
        constraints=constraints,
        options=options,
        verbose=options.get("qp_verbose", False),
    )
    if not res.success:
        raise RuntimeError("Failed to get initial descent direction")
    return res.optimizer.x - X


def _batch_backtracking_ls(
    f: Callable[[Tensor], Tensor],
    X: Tensor,
    FX: Tensor,
    gradFX: Tensor,
    D: Tensor,
    Active: Tensor,
    options: Dict[str, float],
) -> Tuple[Tensor, Dict[str, float]]:
    """Batched backtracking line search to achieve sufficient decrease in f.
    TODO: Implement bisection algorithm for satisfying weak Wolfe conditions
    (see https://sites.math.washington.edu/~burke/crs/408/notes/nlp/line.pdf,
    implemented in https://github.com/hjmshi/PyTorch-LBFGS)

    """
    t0 = time.time()
    beta = options.get("beta", 0.5)
    nu = options.get("nu", 1e-4)
    alpha = torch.ones(X.shape[0], 1, 1, dtype=X.dtype, device=X.device)
    ActF = Active.clone()
    ActF[ActF] = A = _batch_check_armijo(
        lambda X: f(X, ActF), X, FX, gradFX, D, alpha, nu
    )
    n_iter = 0
    while A.sum() > 0:
        n_iter += 1
        alpha[A] *= beta
        ActF[ActF] = A[A] = _batch_check_armijo(
            lambda X: f(X, ActF), X[A], FX[A], gradFX[A], D[A], alpha[A], nu
        )
    diagnostics = {"n_iter": n_iter, "dt": time.time() - t0}
    return alpha, diagnostics


def _batch_check_armijo(
    f: Callable[[Tensor], Tensor],
    X: Tensor,
    FX: Tensor,
    gradFX: Tensor,
    D: Tensor,
    alpha: Tensor,
    nu: float,
) -> Tensor:
    """Check Armijo condition for sufficient descent (batch mode).
    This function maps a (r x n x 1) tensor to a (r x 1 x 1) tensor
    """
    return f(X + alpha * D) > FX + nu * (alpha * D.transpose(1, 2).bmm(gradFX)).view(-1)


def _batch_update_lfbfgs_components(
    Slist: List[Tensor],
    Ylist: List[Tensor],
    X: Tensor,
    gradFX: Tensor,
    X_prev: Tensor,
    gradFX_prev: Tensor,
    m_h: int,
) -> Tuple[Optional[LBFGScompact], Dict[str, float]]:
    """Update L-BFGS-B components efficiently in batch mode."""
    t0 = time.time()
    # compute quantities required for L-BFGS updates
    s = (X - X_prev).squeeze(-1)
    y = (gradFX - gradFX_prev).squeeze(-1)
    if len(Slist) != len(Ylist):
        raise ValueError("Histories must be of the same length!")
    if len(Slist) >= m_h:
        del Slist[0], Ylist[0]
    Slist.append(s)
    Ylist.append(y)
    # peform the update (we only need H, the approx. of the inverse)
    lbfgs = batch_compact_lbfgs_updates(Slist=Slist, Ylist=Ylist, B=False, H=True)
    diagnostics = {"t_lbfgbs": time.time() - t0}
    return lbfgs, diagnostics


def _gen_scaled_identity_H_lbfgs(
    gamma: float, num_batch: int, n: int, dtype=torch.dtype, device=torch.device
) -> LBFGScompact:
    """Generate a trivial L-BFGS approximation of a scaled identity matrix

    There is no need for a history > 1 in this case.
    """
    return LBFGScompact(
        gamma=torch.full((num_batch, 1, 1), gamma, dtype=dtype, device=device),
        F=torch.zeros(num_batch, n, 1, dtype=dtype, device=device),
        E=torch.zeros(num_batch, 1, 1, dtype=dtype, device=device),
        FE=torch.zeros(num_batch, n, 1, dtype=dtype, device=device),
    )
