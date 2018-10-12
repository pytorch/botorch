#!/usr/bin/env python3

from typing import Dict, List, Optional, Union

import torch
from torch import Tensor
from botorch.utils import check_convergence
from gpytorch import Module
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Optimizer


def fit_model(
    gp_model: Module,
    likelihood: Likelihood,
    train_x: Tensor,
    train_y: Tensor,
    optimizer: Optimizer = torch.optim.Adam,
    options: Optional[Dict[str, Union[float, str]]] = None,
    max_iter: int = 50,
    verbose: bool = True,
) -> Module:
    """Fit hyperparameters of a gpytorch model.

    Args:
        gp_model: A gpytorch GP model
        likelihood: A gpytorch likelihood
        train_x: An n x p Tensor of training features
        train_y: An n x 1 Tensor of training lables (n x t) for multi-task GPs
        optimizer: A pytorch Optimzer from the torch.optim module
        options: A dictionary of options to be passed to the optimizer and / or
            the convergence check
        max_iter: The maximum number of optimization steps
        verbose: If True, print information about the fitting to stdout

    Returns:
        The fitted gp_model

    """
    model = gp_model(train_x.detach(), train_y.detach(), likelihood)
    if train_x.is_cuda:
        model.to(device=train_x.device)
    options = options or {}
    model.train()
    likelihood.train()
    optimizer = optimizer(
        [{"params": model.parameters()}],
        lr=options.get("lr", 0.05),  # TODO: be A LOT smarter about this
    )
    mll = ExactMarginalLogLikelihood(likelihood, model)
    param_trajectory: Dict[str, List[Tensor]] = {
        name: [] for name, param in model.named_parameters()
    }
    loss_trajectory: List[float] = []
    i = 0
    converged = False
    while not converged:
        i += 1
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        loss_trajectory.append(loss.item())
        for name, param in model.named_parameters():
            param_trajectory[name].append(param.detach().clone())
        if verbose:
            print("Iter: {} - MLL: {:.3f}".format(i, -loss.item()))
        optimizer.step()
        converged = check_convergence(
            loss_trajectory=loss_trajectory,
            param_trajectory=param_trajectory,
            options=options,
            max_iter=max_iter,
        )

    model.eval()
    likelihood.eval()
    return model
