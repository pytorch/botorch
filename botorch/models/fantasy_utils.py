#!/usr/bin/env python3

from copy import deepcopy
from typing import Dict, Optional, Tuple

import gpytorch
import torch
from torch import Tensor

from .gpytorch import GPyTorchModel


def _get_fantasy_state(
    model: GPyTorchModel,
    X: Tensor,
    num_samples: int,
    base_samples: Optional[Tensor] = None,
) -> GPyTorchModel:
    """Initializes a batched fantasized GP from a given GPyTorch model

    Args:
        model: The input GPyTorch model (must not operate in batch mode)
        X: A `k x p` tensor containing the `k` points at which to fantasize
        num_samples: The number of fantasies
        base_samples:  A Tensor of N(0, 1) random variables used for
            deterministic optimization

    Returns:
        The fantasy model as a GP in batch mode
    """
    # save model parameters
    state_dict = model.state_dict()

    if base_samples is not None:
        num_samples = base_samples.shape[0]

    # generate fantasies from model posterior at new q-batch
    with gpytorch.settings.fast_pred_var():
        posterior = model.posterior(X, observation_noise=True)
        fantasies = posterior.rsample(
            sample_shape=torch.Size([num_samples]), base_samples=base_samples
        )

    train_targets = model.train_targets
    fantasy_shape: Tuple[int, ...]
    # Includes number of tasks if multi-task GP
    if train_targets.ndimension() > 1:
        fantasy_shape = (num_samples, -1, train_targets.shape[-1])
    else:
        fantasy_shape = (num_samples, -1)
    p = model.train_inputs[0].shape[-1]

    # create new training data tensors
    train_X = torch.cat([model.train_inputs[0], X]).expand(num_samples, -1, p)
    train_Y = torch.cat([model.train_targets.expand(*fantasy_shape), fantasies], dim=1)

    return state_dict, train_X, train_Y


def _load_fantasy_state_dict(
    model: GPyTorchModel, state_dict: Dict[str, Tensor]
) -> GPyTorchModel:
    state_dict = deepcopy(state_dict)
    # load the (shared) hyperparameters, make sure to adjust size appropriately
    for k, v in model.named_parameters():
        state_dict[k] = state_dict[k].expand_as(v)
    for k, v in model.named_buffers():
        state_dict[k] = state_dict[k].expand_as(v)
    model.load_state_dict(state_dict)
    model.eval()
    return model
