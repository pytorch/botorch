#!/usr/bin/env python3

import torch
from torch import Tensor


ALPHA = [1.0, 1.2, 3.0, 3.2]

A = [
    [10, 3, 17, 3.5, 1.7, 8],
    [0.05, 10, 17, 0.1, 8, 14],
    [3, 3.5, 1.7, 10, 17, 8],
    [17, 8, 0.05, 10, 0.1, 14],
]

P = [
    [1312, 1696, 5569, 124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091, 381],
]

GLOBAL_MINIMIZER = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
GLOBAL_MINIMUM = -3.32237


def hartmann6(X: Tensor) -> Tensor:
    """Hartmann6 synthetic test function supporting batch evaluation.

    H(x) = - sum_{i=1}^4 ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij)**2 )

    H has a 6 local minima and a global minimum at
    z = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
    with H(z) = -3.32237

    Args:
        X (Tensor): A Tensor of size 6 or k x 6 (k batch evaluations)

    """
    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    inner_sum = torch.sum(X.new(A) * (X.unsqueeze(1) - 0.0001 * X.new(P)) ** 2, dim=2)
    result = -torch.sum(X.new(ALPHA) * torch.exp(-inner_sum), dim=1)
    return result if batch else result.squeeze(0)
