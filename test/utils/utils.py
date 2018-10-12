#! /usr/bin/env python3

import torch
from torch import Tensor


def approx_equal(t1: Tensor, t2: Tensor, epsilon: float = 1e-4) -> bool:
    """Determine if two tensors are approximately equal

    Args:
        t1: tensor
        t2: tensor

    Returns:
        True if max |t1 - t2| < epsilon, Foalse otherwise
    """
    if t1.shape != t2.shape:
        raise RuntimeError(
            "Shape mismatch between t1 ({s1}) and t2 ({s2})".format(
                s1=t1.shape, s2=t2.shape
            )
        )
    return torch.max((t1 - t2).abs()).item() < epsilon
