#!/usr/bin/env python3

import torch
from torch import Tensor


def squeeze_last_dim(Y: Tensor) -> Tensor:
    """Squeeze the last dimension of a Tensor."""
    return Y.squeeze(-1)


def standardize(X: Tensor) -> Tensor:
    """Standardize a tensor by dim=0.

    Args:
        X: tensor `n x (d)`

    Returns:
        Tensor: standardized X
    """
    X_std = X.std(dim=0)
    X_std = X_std.where(X_std >= 1e-9, torch.full_like(X_std, 1.0))
    return (X - X.mean(dim=0)) / X_std
