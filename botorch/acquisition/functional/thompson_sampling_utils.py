#!/usr/bin/env python3

from typing import Callable, List, Optional

import torch
from torch import Tensor

from ...models import Model
from ...utils import squeeze_last_dim
from ..batch_utils import batch_mode_transform
from .batch_acquisition import apply_constraints_nonnegative_soft_


@batch_mode_transform
def discrete_thompson_sample(
    X: Tensor,
    model: Model,
    objective: Callable[[Tensor], Tensor] = squeeze_last_dim,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    mc_samples: int = 5000,
    eta: float = 1e-3,
) -> Tensor:
    """Samples from the model posterior at X and computes the fraction of time
        each point is the winner according to the objective and constraints.

        Note that if the winner is tied, an arbitrary winner will be allocated credit.

    Args:
        X: A `b x q x d` Tensor with `b` t-batches of `q` design points
            each. If X is two-dimensional, assume `b = 1`.
        model: A fitted Model
        objective: A callable mapping a Tensor of size `mc_samples x b x q x t`
            to a Tensor of size `mc_samples x b x q`, where `t` is the number of
            outputs (tasks) of the model. If omitted, use the identity map
            (applicable to single-task models only).
            Assumed to be non-negative when the constraints are used!
        constraints: A list of callables, each mapping a Tensor of size
            `mc_samples x b x q x t` to a Tensor of size `mc_samples x b x q`,
            where negative values imply feasibility. Only relevant for multi-task
            models (`t` > 1).
        mc_samples: The number of Monte-Carlo samples to draw from the model
            posterior.
        eta: The temperature parameter of the softmax function used in approximating
            the constraints. As `eta -> 0`, the exact (discontinuous) constraint
            is recovered.

    Returns:
        Tensor: `b x q` tensor Z which for Z[i,j] contains the fraction of GP posterior
        samples for which X[i,j,:] had the best outcome.
    """
    posterior = model.posterior(X)
    samples = posterior.rsample(torch.Size([mc_samples]))
    # Shape of samples is mc_samples x b x q x t
    obj = objective(samples)

    # TODO: Change this to apply_constraints_ in the future (T40798532).
    apply_constraints_nonnegative_soft_(
        obj=obj, constraints=constraints, samples=samples, eta=eta
    )
    # Shape of obj is mc_samples x b x q
    best_indices = torch.max(obj, dim=-1)[1]

    b, q = obj.shape[-2], obj.shape[-1]
    # Shape of best_indices is mc_samples x b
    # TODO: remove contiguous() when https://github.com/pytorch/pytorch/issues/15058
    # is resolved.
    count = torch.stack(
        [torch.bincount(best_indices[:, i].contiguous(), minlength=q) for i in range(b)]
    )
    # Shape of count is b x q
    return count.type_as(obj) / mc_samples
