#!/usr/bin/env python3

from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor


def get_outcome_constraint_transforms(
    outcome_constraints: Optional[Tuple[Tensor, Tensor]],
) -> Optional[List[Callable[[Tensor], Tensor]]]:
    r"""
    Create outcome constraint callables from outcome constraint tensors

    Args:
        outcome_constraints: A tuple of `(A, b)`. For `k` outcome constraints
            and `m` outputs at `f(x)``, `A` is `k x m` and `b` is `k x 1` such
            that `A f(x) <= b`.

    Returns:
        A list of callables, each mapping a Tensor of size `b x q x t` to a
        Tensor of size `b x q`, where `t` is the number of outputs (tasks) of
        the model. Negative values imply feasibility. The callables support
        broadcasting (e.g. for calling on a tensor of shape
        `mc_samples x b x q x t`).

    """
    if outcome_constraints is None:
        return None
    A, b = outcome_constraints

    def _oc(a: Tensor, rhs: Tensor, Y: Tensor) -> Tensor:
        """
        Evaluate constraints.

        Note: einsum multiples Y by a and sums over the `t`-dimension. Einsum
            is ~2x faster than using `(Y * a.view(1, 1, -1)).sum(dim-1)`.

        Args:
            a: `t`-dim tensor of weights for the outcomes
            rhs: Singleton tensor containing the outcome constraint value
            Y: `... x b x q x t` tensor of function values

        Returns:
            Tensor: `... x b x q`-dim tensor where negative values imply feasibility
        """
        lhs = torch.einsum("...t, t", [Y, a])
        return lhs - rhs

    return [partial(_oc, a, rhs) for a, rhs in zip(A, b)]
