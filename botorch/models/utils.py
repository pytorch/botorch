#! /usr/bin/env python3

r"""
Utiltiy functions for models.
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor


def _make_X_full(X: Tensor, output_indices: List[int], tf: int) -> Tensor:
    r"""Helper to construct input tensor with task indices.

    Args:
        X: The raw input tensor (without task information).
        output_indices: The output indices to generate (passed in via `posterior`).
        tf: The task feature index.

    Returns:
        Tensor: The full input tensor for the multi-task model, including task
            indices.
    """
    index_shape = X.shape[:-1] + torch.Size([1])
    indexers = (
        torch.full(index_shape, fill_value=i, device=X.device, dtype=X.dtype)
        for i in output_indices
    )
    X_l, X_r = X[..., :tf], X[..., tf:]
    return torch.cat(
        [torch.cat([X_l, indexer, X_r], dim=-1) for indexer in indexers], dim=0
    )


def multioutput_to_batch_mode_transform(
    train_X: Tensor,
    train_Y: Tensor,
    num_outputs: int,
    train_Yvar: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    r"""Transforms training inputs for a multi-output model.

    Used for multi-output models that internally are represented by a
    batched single output model, where each output is modeled as an
    independent batch.

    Args:
        train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
            features.
        train_Y: A `n x (o)` or `batch_shape x n x (o)` (batch mode) tensor of
            training observations.
        num_outputs: number of outputs
        train_Yvar: A `batch_shape x n x (o)`
            tensor of observed measurement noise.

    Returns:
        3-element tuple containing

        - A `(o) x batch_shape x n x d` tensor of training features.
        - A `(o) x batch_shape x n` tensor of training observations.
        - A `(o) x batch_shape x n` tensor observed measurement noise.
    """
    input_batch_shape = train_X.shape[:-2]
    if num_outputs > 1:
        # make train_Y `o x batch_shape x n`
        train_Y = train_Y.permute(-1, *range(train_Y.dim() - 1))
        # expand train_X to `o x batch_shape x n x d`
        train_X = train_X.unsqueeze(0).expand(
            torch.Size([num_outputs] + [-1] * train_X.dim())
        )
        if train_Yvar is not None:
            # make train_Yvar `o x batch_shape x n`
            train_Yvar = train_Yvar.permute(-1, *range(train_Yvar.dim() - 1))
    elif train_Y.dim() > 1:
        #  single output, make train_Y `batch_shape x n`
        target_shape = input_batch_shape + torch.Size([-1])
        train_Y = train_Y.view(target_shape)
        if train_Yvar is not None:
            # make train_Yvar `batch_shape x n`
            train_Yvar = train_Yvar.view(target_shape)
    return train_X, train_Y, train_Yvar


def add_output_dim(X: Tensor, original_batch_shape: torch.Size) -> Tuple[Tensor, int]:
    r"""Inserts the output dimension at the correct location. The trailing batch dimensions
        of X must match the original batch dimensions of the training inputs, but
        can also include extra batch dimensions.

    Args:
        X: A `(new_batch_shape) x (original_batch_shape) x n x d` tensor of features.
        original_batch_shape: the batch shape of the model's training inputs.

    Returns:
        2-element tuple containing

        - A `(new_batch_shape) x o x (original_batch_shape) x n x d` tensor of
        features.
        - The index corresponding to the output dimension.
    """
    num_original_batch_dims = len(original_batch_shape)
    if X.shape[-(num_original_batch_dims + 2) : -2] != original_batch_shape:
        raise ValueError(
            "The trailing batch dimensions of X must match the batch dimensions of the"
            " training inputs."
        )
    # insert `t` dimension
    output_dim_idx = len(X.shape) - (num_original_batch_dims + 2)
    X = X.unsqueeze(output_dim_idx)
    return X, output_dim_idx
