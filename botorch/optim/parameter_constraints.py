#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utility functions for constrained optimization.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.exceptions.errors import CandidateGenerationError, UnsupportedError
from scipy.optimize import Bounds
from torch import Tensor


ScipyConstraintDict = Dict[
    str, Union[str, Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]
]
NLC_TOL = -1e-6


def make_scipy_bounds(
    X: Tensor,
    lower_bounds: Optional[Union[float, Tensor]] = None,
    upper_bounds: Optional[Union[float, Tensor]] = None,
) -> Optional[Bounds]:
    r"""Creates a scipy Bounds object for optimziation

    Args:
        X: `... x d` tensor
        lower_bounds: Lower bounds on each column (last dimension) of `X`. If
            this is a single float, then all columns have the same bound.
        upper_bounds: Lower bounds on each column (last dimension) of `X`. If
            this is a single float, then all columns have the same bound.

    Returns:
        A scipy `Bounds` object if either lower_bounds or upper_bounds is not
        None, and None otherwise.

    Example:
        >>> X = torch.rand(5, 2)
        >>> scipy_bounds = make_scipy_bounds(X, 0.1, 0.8)
    """
    if lower_bounds is None and upper_bounds is None:
        return None

    def _expand(bounds: Union[float, Tensor], X: Tensor, lower: bool) -> Tensor:
        if bounds is None:
            ebounds = torch.full_like(X, float("-inf" if lower else "inf"))
        else:
            if not torch.is_tensor(bounds):
                bounds = torch.tensor(bounds)
            ebounds = bounds.expand_as(X)
        return _arrayify(ebounds).flatten()

    lb = _expand(bounds=lower_bounds, X=X, lower=True)
    ub = _expand(bounds=upper_bounds, X=X, lower=False)
    return Bounds(lb=lb, ub=ub, keep_feasible=True)


def make_scipy_linear_constraints(
    shapeX: torch.Size,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
) -> List[ScipyConstraintDict]:
    r"""Generate scipy constraints from torch representation.

    Args:
        shapeX: The shape of the torch.Tensor to optimize over (i.e. `b x q x d`)
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`, where
            `indices` is a single-dimensional index tensor (long dtype) containing
            indices into the last dimension of `X`, `coefficients` is a
            single-dimensional tensor of coefficients of the same length, and
            rhs is a scalar.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) == rhs` (with `indices`
            and `coefficients` of the same form as in `inequality_constraints`).

    Returns:
        A list of dictionaries containing callables for constraint function
        values and Jacobians and a string indicating the associated constraint
        type ("eq", "ineq"), as expected by `scipy.minimize`.

    This function assumes that constraints are the same for each input batch,
    and broadcasts the constraints accordingly to the input batch shape. This
    function does support constraints across elements of a q-batch if the
    indices are a 2-d Tensor.

    Example:
        The following will enforce that `x[1] + 0.5 x[3] >= -0.1` for each `x`
        in both elements of the q-batch, and each of the 3 t-batches:

        >>> constraints = make_scipy_linear_constraints(
        >>>     torch.Size([3, 2, 4]),
        >>>     [(torch.tensor([1, 3]), torch.tensor([1.0, 0.5]), -0.1)],
        >>> )

        The following will enforce that `x[0, 1] + 0.5 x[1, 3] >= -0.1` where
        x[0, :] is the first element of the q-batch and x[1, :] is the second
        element of the q-batch, for each of the 3 t-batches:

        >>> constraints = make_scipy_linear_constraints(
        >>>     torch.size([3, 2, 4])
        >>>     [(torch.tensor([[0, 1], [1, 3]), torch.tensor([1.0, 0.5]), -0.1)],
        >>> )
    """
    constraints = []
    if inequality_constraints is not None:
        for indcs, coeffs, rhs in inequality_constraints:
            constraints += _make_linear_constraints(
                indices=indcs, coefficients=coeffs, rhs=rhs, shapeX=shapeX, eq=False
            )
    if equality_constraints is not None:
        for indcs, coeffs, rhs in equality_constraints:
            constraints += _make_linear_constraints(
                indices=indcs, coefficients=coeffs, rhs=rhs, shapeX=shapeX, eq=True
            )
    return constraints


def eval_lin_constraint(
    x: np.ndarray, flat_idxr: List[int], coeffs: np.ndarray, rhs: float
) -> np.float64:
    r"""Evaluate a single linear constraint.

    Args:
        x: The input array.
        flat_idxr: The indices in `x` to consider.
        coeffs: The coefficients corresponding to the indices.
        rhs: The right-hand-side of the constraint.

    Returns:
        The evaluted constraint: `\sum_i (coeffs[i] * x[i]) - rhs`
    """
    return np.sum(x[flat_idxr] * coeffs, -1) - rhs


def lin_constraint_jac(
    x: np.ndarray, flat_idxr: List[int], coeffs: np.ndarray, n: int
) -> np.ndarray:
    r"""Return the Jacobian associated with a linear constraint.

    Args:
        x: The input array.
        flat_idxr: The indices for the elements of x that appear in the constraint.
        coeffs: The coefficients corresponding to the indices.
        n: number of elements

    Returns:
        The Jacobian.
    """
    # TODO: Use sparse representation (not sure if scipy optim supports that)
    jac = np.zeros(n)
    jac[flat_idxr] = coeffs
    return jac


def _arrayify(X: Tensor) -> np.ndarray:
    r"""Convert a torch.Tensor (any dtype or device) to a numpy (double) array.

    Args:
        X: The input tensor.

    Returns:
        A numpy array of double dtype with the same shape and data as `X`.
    """
    return X.cpu().detach().contiguous().double().clone().numpy()


def _make_linear_constraints(
    indices: Tensor,
    coefficients: Tensor,
    rhs: float,
    shapeX: torch.Size,
    eq: bool = False,
) -> List[ScipyConstraintDict]:
    r"""Create linear constraints to be used by `scipy.minimize`.

    Encodes constraints of the form
    `\sum_i (coefficients[i] * X[..., indices[i]]) ? rhs`
    where `?` can be designated either as `>=` by setting `eq=False`, or as
    `=` by setting `eq=True`.

    If indices is one-dimensional, the constraints are broadcasted across
    all elements of the q-batch. If indices is two-dimensional, then
    constraints are applied across elements of a q-batch. In either case,
    constraints are created for all t-batches.

    Args:
        indices: A tensor of shape `c` or `c x 2`, where c is the number of terms
            in the constraint. If single-dimensional, contains the indices of
            the dimensions of the feature space that occur in the linear
            constraint. If two-dimensional, contains pairs of indices of the
            q-batch (0) and the feature space (1) that occur in the linear
            constraint.
        coefficients: A single-dimensional tensor of coefficients with the same
            number of elements as `indices`.
        rhs: The right hand side of the constraint.
        shapeX: The shape of the torch tensor to construct the constraints for
            (i.e. `b x q x d`). Must have three dimensions.
        eq: If True, return an equality constraint, o/w return an inequality
            constraint (indicated by "eq" / "ineq" value of the `type` key).

    Returns:
        A list of constraint dictionaries with the following keys

        - "type": Indicates the type of the constraint ("eq" if `eq=True`, "ineq" o/w)
        - "fun": A callable evaluating the constraint value on `x`, a flattened
            version of the input tensor `X`, returning a scalar.
        - "jac": A callable evaluating the constraint's Jacobian on `x`, a flattened
            version of the input tensor `X`, returning a numpy array.
    """
    if len(shapeX) != 3:
        raise UnsupportedError("`shapeX` must be `b x q x d`")
    q, d = shapeX[-2:]
    n = shapeX.numel()
    constraints: List[ScipyConstraintDict] = []
    coeffs = _arrayify(coefficients)
    ctype = "eq" if eq else "ineq"
    if indices.dim() > 2:
        raise UnsupportedError(
            "Linear constraints supported only on individual candidates and "
            "across q-batches, not across general batch shapes."
        )
    elif indices.dim() == 2:
        # indices has two dimensions (potential constraints across q-batch elements)
        if indices[:, 0].max() > q - 1:
            raise RuntimeError(f"Index out of bounds for {q}-batch")
        if indices[:, 1].max() > d - 1:
            raise RuntimeError(f"Index out of bounds for {d}-dim parameter tensor")

        offsets = [shapeX[i:].numel() for i in range(1, len(shapeX))]
        # rule is [i, j, k] is at
        # i * offsets[0] + j * offsets[1] + k
        for i in range(shapeX[0]):
            idxr = []
            for a in indices:
                b = a.tolist()
                idxr.append(i * offsets[0] + b[0] * offsets[1] + b[1])
            fun = partial(
                eval_lin_constraint, flat_idxr=idxr, coeffs=coeffs, rhs=float(rhs)
            )
            jac = partial(lin_constraint_jac, flat_idxr=idxr, coeffs=coeffs, n=n)
            constraints.append({"type": ctype, "fun": fun, "jac": jac})
    elif indices.dim() == 1:
        # indices is one-dim - broadcast constraints across q-batches and t-batches
        if indices.max() > d - 1:
            raise RuntimeError(f"Index out of bounds for {d}-dim parameter tensor")
        offsets = [shapeX[i:].numel() for i in range(1, len(shapeX))]
        for i in range(shapeX[0]):
            for j in range(shapeX[1]):
                idxr = (i * offsets[0] + j * offsets[1] + indices).tolist()
                fun = partial(
                    eval_lin_constraint, flat_idxr=idxr, coeffs=coeffs, rhs=float(rhs)
                )
                jac = partial(lin_constraint_jac, flat_idxr=idxr, coeffs=coeffs, n=n)
                constraints.append({"type": ctype, "fun": fun, "jac": jac})
    else:
        raise ValueError("`indices` must be at least one-dimensional")
    return constraints


def _generate_unfixed_lin_constraints(
    constraints: Optional[List[Tuple[Tensor, Tensor, float]]],
    fixed_features: Dict[int, float],
    dimension: int,
    eq: bool,
) -> Optional[List[Tuple[Tensor, Tensor, float]]]:

    # If constraints is None or an empty list, then return itself
    if not constraints:
        return constraints

    # replace_index generates the new indices for the unfixed dimensions
    # after eliminating the fixed dimensions.
    # Example: dimension = 5, ff.keys() = [1, 3], replace_index = {0: 0, 2: 1, 4: 2}
    unfixed_keys = sorted(set(range(dimension)) - set(fixed_features))
    unfixed_keys = torch.tensor(unfixed_keys).to(constraints[0][0])
    replace_index = torch.arange(dimension - len(fixed_features)).to(constraints[0][0])

    new_constraints = []
    # parse constraints one-by-one
    for constraint_id, (indices, coefficients, rhs) in enumerate(constraints):
        new_rhs = rhs
        new_indices = []
        new_coefficients = []
        # the following unsqueeze is done to facilitate a simpler for-loop.
        indices_2dim = indices if indices.ndim == 2 else indices.unsqueeze(-1)
        for coefficient, index in zip(coefficients, indices_2dim):
            ffval_or_None = fixed_features.get(index[-1].item())
            # if ffval_or_None is None, then the index is not fixed
            if ffval_or_None is None:
                new_indices.append(index)
                new_coefficients.append(coefficient)
            # otherwise, we "remove" the constraints corresponding to that index
            else:
                new_rhs = new_rhs - coefficient.item() * ffval_or_None

        # all indices were fixed, so the constraint is gone.
        if len(new_indices) == 0:
            if (eq and new_rhs != 0) or (not eq and new_rhs > 0):
                prefix = "Eq" if eq else "Ineq"
                raise CandidateGenerationError(
                    f"{prefix}uality constraint {constraint_id} not met "
                    "with fixed_features."
                )
        else:
            # However, one key transformation has to be noted.
            # new_indices is with respect to the older (fuller) domain, and so it will
            # have to be converted using replace_index.
            new_indices = torch.stack(new_indices, dim=0)
            # generate new index location after the removal of fixed_features indices
            new_indices_dim_d = new_indices[:, -1].unsqueeze(-1)
            new_indices_dim_d = replace_index[
                torch.nonzero(new_indices_dim_d == unfixed_keys, as_tuple=True)[1]
            ]
            new_indices[:, -1] = new_indices_dim_d
            # squeeze(-1) is a no-op if dim -1 is not singleton
            new_indices.squeeze_(-1)
            # convert new_coefficients to Tensor
            new_coefficients = torch.stack(new_coefficients)
            new_constraints.append((new_indices, new_coefficients, new_rhs))
    return new_constraints


def _make_f_and_grad_nonlinear_inequality_constraints(
    f_np_wrapper: Callable, nlc: Callable
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """
    Create callables for objective + grad for the nonlinear inequality constraints.
    The Scipy interface requires specifying separate callables and we use caching to
    avoid evaluating the same input twice. This caching onlh works if
    the returned functions are evaluated on the same input in immediate
    sequence (i.e., calling `f_obj(X_1)`, `f_grad(X_1)` will result in a
    single forward pass, while `f_obj(X_1)`, `f_grad(X_2)`, `f_obj(X_1)`
    will result in three forward passes).
    """

    def f_obj_and_grad(x):
        obj, grad = f_np_wrapper(x, f=nlc)
        return obj, grad

    cache = {"X": None, "obj": None, "grad": None}

    def f_obj(X):
        X_c = cache["X"]
        if X_c is None or not np.array_equal(X_c, X):
            cache["X"] = X.copy()
            cache["obj"], cache["grad"] = f_obj_and_grad(X)
        return cache["obj"]

    def f_grad(X):
        X_c = cache["X"]
        if X_c is None or not np.array_equal(X_c, X):
            cache["X"] = X.copy()
            cache["obj"], cache["grad"] = f_obj_and_grad(X)
        return cache["grad"]

    return f_obj, f_grad


def make_scipy_nonlinear_inequality_constraints(
    nonlinear_inequality_constraints: List[Callable],
    f_np_wrapper: Callable,
    x0: Tensor,
) -> List[Dict]:
    r"""Generate Scipy nonlinear inequality constraints from callables.

    Args:
        nonlinear_inequality_constraints: List of callables for the nonlinear
            inequality constraints. Each callable represents a constraint of the
            form >= 0 and takes a torch tensor of size (p x q x dim) and returns a
            torch tensor of size (p x q).
        f_np_wrapper: A wrapper function that given a constraint evaluates the value
             and gradient (using autograd) of a numpy input and returns both the
             objective and the gradient.
        x0: The starting point for SLSQP. We return this starting point in (rare)
            cases where SLSQP fails and thus require it to be feasible.

    Returns:
        A list of dictionaries containing callables for constraint function
        values and Jacobians and a string indicating the associated constraint
        type ("eq", "ineq"), as expected by `scipy.minimize`.
    """
    if not isinstance(nonlinear_inequality_constraints, list):
        raise ValueError(
            "`nonlinear_inequality_constraints` must be a list of callables, "
            f"got {type(nonlinear_inequality_constraints)}."
        )

    scipy_nonlinear_inequality_constraints = []
    for nlc in nonlinear_inequality_constraints:
        if _arrayify(nlc(x0)).item() < NLC_TOL:
            raise ValueError(
                "`batch_initial_conditions` must satisfy the non-linear inequality "
                "constraints."
            )
        f_obj, f_grad = _make_f_and_grad_nonlinear_inequality_constraints(
            f_np_wrapper=f_np_wrapper, nlc=nlc
        )
        scipy_nonlinear_inequality_constraints.append(
            {
                "type": "ineq",
                "fun": f_obj,
                "jac": f_grad,
            }
        )
    return scipy_nonlinear_inequality_constraints
