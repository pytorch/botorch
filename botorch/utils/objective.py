#!/usr/bin/env python3

from typing import Callable, Optional

import torch
from torch import Tensor


def get_objective_weights_transform(
    weights: Optional[Tensor]
) -> Callable[[Tensor], Tensor]:
    """Greate a linear objective callable froma set of weights.

    Create a callable mapping a Tensor of size `b x q x t` to a Tensor of size
    `b x q`, where `t` is the number of outputs (tasks) of the model using
    scalarization via the objective weights. This callable supports broadcasting
    (e.g. for calling on a tensor of shape `mc_samples x b x q x t`). For `t = 1`,
    the objective weight is used to determine the optimization direction.

    Args:
        weights: a 1-dimensional Tensor containing a weight for each task.
            If not provided, the identity mapping is used.

    Returns:
        Callable[Tensor, Tensor]: transform function using the objective weights

    """
    # if no weights provided, just extract the single output
    if weights is None:
        return lambda Y: Y.squeeze(-1)
    # TODO: replace with einsum once pytorch performance issues are resolved
    return lambda Y: torch.sum(Y * weights.view(1, 1, -1), dim=-1)
