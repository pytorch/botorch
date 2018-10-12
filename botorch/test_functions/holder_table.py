#!/usr/bin/env python3

import math

import torch
from torch import Tensor


GLOBAL_MINIMIZERS = [
    [8.0550, 9.6646],
    [-8.0550, -9.6646],
    [-8.0550, 9.6646],
    [8.0550, -9.6646],
]
GLOBAL_MINIMUM = -19.2085


def holder_table(X: Tensor):
    """Holder Table synthetic test function supporting batch evaluation.

    Two-dimensional function that is typically evaluated on [0, 10]^2.

    H(x) = - | sin(x_1) * cos(x_2) * exp(| 1 - ||x|| / pi |) |

    H has 4 global minima with H(z_i) = -19.2085 at
    z_1 = ( 8.05502,  9.66459)
    z_2 = (-8.05502, -9.66459)
    z_3 = (-8.05502,  9.66459)
    z_4 = ( 8.05502, -9.66459)

    Args:
        X (Tensor): A Tensor of size 2 or k x 2 (k batch evaluations)

    """
    batch = X.ndimension() > 1
    X = X if batch else X.unsqueeze(0)
    term = torch.abs(1 - torch.norm(X, dim=1) / math.pi)
    result = -torch.abs(torch.sin(X[:, 0]) * torch.cos(X[:, 1]) * torch.exp(term))
    return result if batch else result.squeeze(0)
