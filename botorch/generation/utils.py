#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import typing  # noqa F401
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from botorch.optim.parameter_constraints import _arrayify
from botorch.optim.utils import fix_features
from torch import Tensor
from torch.nn import Module


def _flip_sub_unique(x: Tensor, k: int) -> Tensor:
    """Get the first k unique elements of a single-dimensional tensor, traversing the
    tensor from the back.

    Args:
        x: A single-dimensional tensor
        k: the number of elements to return

    Returns:
        A tensor with min(k, |x|) elements.

    Example:
        >>> x = torch.tensor([1, 6, 4, 3, 6, 3])
        >>> y = _flip_sub_unique(x, 3)  # tensor([3, 6, 4])
        >>> y = _flip_sub_unique(x, 4)  # tensor([3, 6, 4, 1])
        >>> y = _flip_sub_unique(x, 10)  # tensor([3, 6, 4, 1])

    NOTE: This should really be done in C++ to speed up the loop. Also, we would like
    to make this work for arbitrary batch shapes, I'm sure this can be sped up.
    """
    n = len(x)
    i = 0
    out = set()
    idcs = torch.empty(k, dtype=torch.long)
    for j, xi in enumerate(x.flip(0).tolist()):
        if xi not in out:
            out.add(xi)
            idcs[i] = n - 1 - j
            i += 1
        if len(out) >= k:
            break
    return x[idcs[: len(out)]]


def get_candidate_optim_objective(
    initial_conditions: Tensor,
    acquisition_function: Module,
    with_grad: bool,
    fixed_features: Optional[Dict[int, Optional[float]]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
) -> Union[Callable[[np.ndarray], float, np.ndarray], Callable[[np.ndarray], float]]:
    r"""Construct the objective for candidate optimization.

    Note: the post-processing function is only applied if `with_grad=False`.

    Args:
        initial_conditions: Starting points for optimization.
        acquisition_function: Acquisition function to be used.
        with_grad: A boolean indicating whether to return gradients
        fixed_features: This is a dictionary of feature indices to values, where
            all generated candidates will have features fixed to these values.
            If the dictionary value is None, then that feature will just be
            fixed to the clamped value and not optimized. Assumes values to be
            compatible with lower_bounds and upper_bounds!
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).

    Returns:
        A callable objective function for candidate optimization.

    """
    shapeX = initial_conditions.shape
    if with_grad:

        def f(x):
            X = (
                torch.from_numpy(x)
                .to(initial_conditions)
                .view(shapeX)
                .contiguous()
                .requires_grad_(True)
            )
            X_processed = post_processing_func(X)
            X_fix = fix_features(X=X_processed, fixed_features=fixed_features)
            loss = -acquisition_function(X_fix).sum()
            # compute gradient w.r.t. the inputs (does not accumulate in leaves)
            gradf = _arrayify(torch.autograd.grad(loss, X)[0].contiguous().view(-1))
            fval = loss.item()
            return fval, gradf

    else:

        def f(x):
            with torch.no_grad():
                X = torch.from_numpy(x).to(initial_conditions).view(shapeX).contiguous()
                X_processed = post_processing_func(X)
                X_fix = fix_features(X=X_processed, fixed_features=fixed_features)
                loss = -acquisition_function(X_fix).sum()
                fval = loss.item()
            return fval

    return f
