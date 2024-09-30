import torch
from linear_operator.operators import LinearOperator


def permute_solve(A: LinearOperator, b: LinearOperator) -> LinearOperator:
    r"""Solve the batched linear system AX = b, where b is a batched column
    vector. The solve is carried out after permuting the largest batch
    dimension of b to the final position, which results in a more efficient
    matrix-matrix solve.

    This ideally should be handled upstream (in GPyTorch, linear_operator or
    PyTorch), after which any uses of this method can be replaced with
    `A.solve(b)`.

    Args:
        A: LinearOperator of shape (n, n)
        b: LinearOperator of shape (..., n, 1)

    Returns:
        LinearOperator of shape (..., n, 1)
    """
    # permute dimensions to move largest batch dimension to the end (more efficient
    # than unsqueezing)
    largest_batch_dim = max(enumerate(b.shape[:-1]), key=lambda t: t[0])
    perm = list(range(b.ndim))
    perm.remove(largest_batch_dim)
    perm.append(largest_batch_dim)
    b_p = b.permute(*perm)

    # solve
    x_p = A.solve(b_p)

    # Undo permutation
    inverse_perm = torch.argsort(torch.tensor(perm))
    x = x_p.permute(*inverse_perm).unsqueeze(-1)

    return x
