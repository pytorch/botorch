#!/usr/bin/env python3

from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.optimize import Bounds
from torch import Tensor


ScipyConstraintDict = Dict[
    str, Union[str, Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]
]


def make_scipy_bounds(
    lower_bounds: Optional[Union[float, Tensor]],
    upper_bounds: Optional[Union[float, Tensor]],
    X: Tensor,
) -> Optional[Bounds]:
    """Creates a scipy Bounds object for optimziation

    Args:
        lower_bounds: Lower bounds on each column (last dimension) of X. If this
            is a single float, then all columns have the same bound.
        upper_bounds: Lower bounds on each column (last dimension) of X. If this
            is a single float, then all columns have the same bound.
        X: `... x d` tensor

    Returns
        A scipy Bounds object if either lower_bounds or upper_bounds is not None,
            and None otherwise.
    """
    if lower_bounds is None and upper_bounds is None:
        return None

    def _expand(bounds: Tensor, X: Tensor) -> Tensor:
        if bounds is None:
            ebounds = torch.full_like(X, float("inf"))
        else:
            if not torch.is_tensor(bounds):
                bounds = torch.tensor(bounds)
            ebounds = bounds.expand_as(X)
        return _arrayify(ebounds).flatten()

    lb = _expand(lower_bounds, X)
    ub = _expand(upper_bounds, X)
    return Bounds(lb=lb, ub=ub, keep_feasible=True)


def make_scipy_linear_constraints(
    shapeX: torch.Size,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
) -> Tuple[ScipyConstraintDict]:
    """Generate scipy constraints from torch represenation

    Args:
        shapeX: The shape of the torch tensor to optimze over (i.e. `b x q x d`)
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`

    Returns:
        A list of dictionaries with callables for function value and Jacobian
            as expected by scipy.minimize, together with their associated
            constraint types ("eq", "ineq")
    """
    constraints = []
    if inequality_constraints is not None:
        for indcs, coeffs, rhs in inequality_constraints:
            c = _make_lin_constraint(
                indices=indcs, coefficients=coeffs, rhs=rhs, shapeX=shapeX
            )
            c["type"] = "ineq"
            constraints.append(c)
    if equality_constraints is not None:
        for indcs, coeffs, rhs in equality_constraints:
            c = _make_lin_constraint(
                indices=indcs, coefficients=coeffs, rhs=rhs, shapeX=shapeX
            )
            c["type"] = "eq"
            constraints.append(c)
    return tuple(constraints)


def eval_lin_constraint(
    x: np.ndarray, flat_idxr: List[int], coeffs: np.ndarray, rhs: float
) -> float:
    """Evaluate a single linear constraint"""
    return np.sum(x[flat_idxr] * coeffs, -1) - rhs


def lin_constraint_jac(
    x: np.ndarray, flat_idxr: List[int], coeffs: np.ndarray, n: int
) -> np.ndarray:
    """Return the jacobian associated with a linear constraint"""
    # TODO: Use sparse representation (not sure if scipy optim supports that)
    jac = np.zeros(n)
    jac[flat_idxr] = coeffs
    return jac


def _arrayify(X: Tensor) -> np.ndarray:
    return X.cpu().detach().contiguous().double().clone().numpy()


def _make_flat_indexer(indices: Tensor, shape: torch.Size) -> List[int]:
    """Convert a list of multi-dimensional index tensors for a multi-dimensional
    tensor X to the one-dimensional indexer for the corresponding flattened X
    """
    multipliers = [shape[i:].numel() for i in range(1, len(shape))] + [1]
    multipliers = torch.tensor(multipliers, device=indices.device)
    return (indices * multipliers).sum(-1).tolist()


def _make_lin_constraint(
    indices: Tensor, coefficients: Tensor, rhs: float, shapeX: torch.Size
) -> ScipyConstraintDict:
    """Create a linear constraint to be used by scipy.minimize

    Implements a constraint of the form
        `\sum_i (X[indices[i]] * coefficients[i]) ? rhs`
    where `?` can be designated either as `>=` by setting `type="ineq"`, or as
    `=` by setting `type="eq"` in the returned dictionary.

    Args:
        indices:
        coefficients:
        rhs: The right hand side of the constraint.
        shapeX: The shape of the torch tensor to optimze over (i.e. `b x q x d`)

    Returns:
        A dictionary with keys "fun" and "jac", each with the appropritely
        constructed callable on a single-dimensional input `x` (the flattened,
        numpyified version of the optimization variable X). This does not contain
        the "type" key indicating the constraint type ("eq" or "ineq").
    """
    d = shapeX[-1]
    if indices.max() >= d:
        raise RuntimeError(f"Index out of bounds for {d}-dim parameter tensor")
    flat_idxr = _make_flat_indexer(indices=indices, shape=shapeX)
    coeffs = _arrayify(coefficients)
    # TODO: Check signs
    fun = partial(eval_lin_constraint, flat_idxr=flat_idxr, coeffs=coeffs, rhs=rhs)
    jac = partial(
        lin_constraint_jac, flat_idxr=flat_idxr, coeffs=coeffs, n=shapeX.numel()
    )
    return {"fun": fun, "jac": jac}
