#!/usr/bin/env python3

from copy import deepcopy
from typing import Optional, Tuple

import torch
from botorch.utils import manual_seed
from torch import Tensor

from .gpytorch import GPyTorchModel


def initialize_batch_fantasy_GP(
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
    state_dict = deepcopy(model.state_dict())

    if base_samples is not None:
        num_samples = base_samples.shape[0]

    train_targets = model.train_targets
    fantasy_shape: Tuple[int, ...]
    # Includes number of tasks if multi-task GP
    if train_targets.ndimension() > 1:
        fantasy_shape = (num_samples, -1, train_targets.shape[-1])
    else:
        fantasy_shape = (num_samples, -1)
    p = model.train_inputs[0].shape[-1]

    # generate fantasies from model posterior at new q-batch
    posterior = model.posterior(X, observation_noise=True)
    fantasies = posterior.rsample(
        sample_shape=torch.Size([num_samples]), base_samples=base_samples
    )

    # create new training data tensors
    train_x = torch.cat([model.train_inputs[0], X]).expand(num_samples, -1, p)
    train_y = torch.cat([model.train_targets.expand(*fantasy_shape), fantasies], dim=1)

    # instantiate the fantasy model(s)
    likelihood = deepcopy(model.likelihood).eval()
    fantasy_model = model.__class__(train_x, train_y, likelihood)

    # load the (shared) hyperparameters, make sure to adjust size appropriately
    for k, v in fantasy_model.named_parameters():
        state_dict[k] = state_dict[k].expand_as(v)
    for k, v in fantasy_model.named_buffers():
        state_dict[k] = state_dict[k].expand_as(v)
    fantasy_model.load_state_dict(state_dict)
    fantasy_model.eval()

    return fantasy_model
