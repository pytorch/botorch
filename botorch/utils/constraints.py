#!/usr/bin/env python3

r"""
Helpers for handling outcome constraints.
"""

from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor


def get_outcome_constraint_transforms(
    outcome_constraints: Optional[Tuple[Tensor, Tensor]],
) -> Optional[List[Callable[[Tensor], Tensor]]]:
    r"""Create outcome constraint callables from outcome constraint tensors.

    Args:
        outcome_constraints: A tuple of `(A, b)`. For `k` outcome constraints
            and `o` outputs at `f(x)``, `A` is `k x o` and `b` is `k x 1` such
            that `A f(x) <= b`.

    Returns:
        A list of callables, each mapping a Tensor of size `b x q x o` to a
        tensor of size `b x q`, where `o` is the number of outputs of the model.
        Negative values imply feasibility. The callables support broadcasting
        (e.g. for calling on a tensor of shape `mc_samples x b x q x o`).

    Example:
        >>> # constrain `f(x)[0] <= 0`
        >>> A = torch.tensor([[1., 0.]])
        >>> b = torch.tensor([[0.]])
        >>> outcome_constraints = get_outcome_constraint_transforms((A, b))
    """
    if outcome_constraints is None:
        return None
    A, b = outcome_constraints

    def _oc(a: Tensor, rhs: Tensor, Y: Tensor) -> Tensor:
        r"""Evaluate constraints.

        Note: einsum multiples Y by a and sums over the `o`-dimension. Einsum
            is ~2x faster than using `(Y * a.view(1, 1, -1)).sum(dim-1)`.

        Args:
            a: `o`-dim tensor of weights for the outcomes
            rhs: Singleton tensor containing the outcome constraint value
            Y: `... x b x q x o` tensor of function values

        Returns:
            A `... x b x q`-dim tensor where negative values imply feasibility
        """
        lhs = torch.einsum("...o, o", [Y, a])
        return lhs - rhs

    return [partial(_oc, a, rhs) for a, rhs in zip(A, b)]
