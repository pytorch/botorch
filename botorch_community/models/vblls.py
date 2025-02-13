#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains an implemenation of a Variational Bayesian Last Layer (VBLL) model that can be used within BoTorch.

References:

[1] P. Brunzema, M. Jordahn, J. Willes, S. Trimpe, J. Snoek, J. Harrison.
    Bayesian Optimization via Contrinual Variational Last Layer Training.
    International Conference on Learning Representations, 2025.

Contributor: brunzema
"""

import os
from typing import Optional, Dict
from abc import ABC, abstractmethod

import random

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from gpytorch.distributions import MultivariateNormal

from botorch.posteriors import Posterior
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior

from botorch_community.posteriors.bll_posterior import BLLPosterior

import vbll


torch.set_default_dtype(torch.float64)


class VBLLNetwork(nn.Module):
    """
    A model with a Variational Bayesian Linear Last (VBLL) layer.

    Args:
        in_features (int, optional):
            Number of input features. Defaults to 2.
        hidden_features (int, optional):
            Number of hidden units per layer. Defaults to 50.
        out_features (int, optional):
            Number of output features. Defaults to 1.
        num_layers (int, optional):
            Number of hidden layers in the MLP. Defaults to 3.
        prior_scale (float, optional):
            Scaling factor for the prior distribution in the Bayesian last layer. Defaults to 1.0.
        wishart_scale (float, optional):
            Scaling factor for the Wishart prior in the Bayesian last layer. Defaults to 0.01.
        kl_scale (float, optional):
            Weighting factor for the Kullback-Leibler (KL) divergence term in the loss. Defaults to 1.0.
        backbone (nn.Module, optional):
            A predefined feature extractor to be used before the MLP layers. If None,
            a default MLP structure is used. Defaults to None.
        activation (nn.Module, optional):
            Activation function applied between hidden layers. Defaults to `nn.ELU()`.

    Notes:
        - If a `backbone` module is provided, it is applied before the fully connected layers. If not, we use a default MLP structure.
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 50,
        out_features: int = 1,
        num_layers: int = 3,
        prior_scale: float = 1.0,
        wishart_scale: float = 0.01,
        kl_scale: float = 1.0,
        backbone: nn.Module = None,
        activation: nn.Module = nn.ELU(),
        device=None,
    ):
        super(VBLLNetwork, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.activation = activation
        self.kl_scale = kl_scale

        if backbone is None:
            self.backbone = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                self.activation,
                *[
                    nn.Sequential(
                        nn.Linear(hidden_features, hidden_features), self.activation
                    )
                    for _ in range(num_layers)
                ],
            )
        else:
            self.backbone = backbone

        # could be changed to other regression layers in vbll package
        self.head = vbll.Regression(
            hidden_features,
            out_features,
            regularization_weight=1.0,  # will be adjusted dynamically at each iteration based on the number of data points
            prior_scale=prior_scale,
            wishart_scale=wishart_scale,
            parameterization="dense_precision",
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        return self.head(x)

    def sample_posterior_function(self, sample_shape: Optional[torch.Size] = None):
        """
        Samples a posterior function by drawing parameters from the model's learned distribution.

        Args:
            sample_shape (Optional[torch.Size], optional):
                The desired shape for the sampled parameters. If None, a single sample is drawn.
                Defaults to None.

        Returns:
            Callable[[Tensor], Tensor]:
                A function that takes an input tensor `x` and returns the corresponding
                model output tensor. The function applies the backbone transformation
                and computes the final output using the sampled parameters.

        Notes:
            - If `sample_shape` is None, a single set of parameters is sampled.
            - If `sample_shape` is provided, multiple parameter samples are drawn, and the function
              will return a batched output where the first dimension corresponds to different samples.
        """
        if sample_shape is None:
            sampled_params = self.head.W().rsample().to(self.device)
        else:
            sampled_params = self.head.W().rsample(sample_shape).to(self.device)

        def sampled_parametric_function(x: Tensor) -> Tensor:
            x = self.backbone(x)

            if sample_shape is None:
                return (sampled_params @ x[..., None]).squeeze(-1)

            x_expanded = x.unsqueeze(0).expand(sampled_params.shape[0], -1, -1)
            output = torch.matmul(sampled_params, x_expanded.transpose(-1, -2))
            return output

        return sampled_parametric_function


class AbstractBLLModel(Model, ABC):
    def __init__(self):
        super().__init__()
        self.model = None
        self.old_model = None  # Used for continual learning

    @property
    def num_outputs(self) -> int:
        return self.model.num_outputs

    @property
    def num_inputs(self):
        return self.model.num_inputs

    @property
    def device(self):
        return self.model.device

    @staticmethod
    def sigmoid_retrain_schedule(
        time_horizon, epoch, transition_window_ratio, location
    ):
        # Compute stretch dynamically
        transition_window = time_horizon * transition_window_ratio
        stretch = (
            2 * torch.log(torch.tensor(9.0)) / transition_window
        )  # Ensures 10%-90% transition within the transition_window, see [1]

        # Sigmoid function
        probability = 1 / (1 + torch.exp(-stretch * (location - epoch)))
        return torch.rand(1).item() < probability

    def fit(
        self,
        train_X: Tensor,
        train_y: Tensor,
        optimization_settings: Dict = None,
        continual_learning_settings: Dict = None,
        old_model_params: Dict = None,
    ):
        """
        Fits the model to the given training data. Note that for continual learning, we assume that the last point in the training data is the new point.

        Args:
            train_X (Tensor):
                The input training data, expected to be a PyTorch tensor of shape (num_samples, num_features).

            train_y (Tensor):
                The target values for training, expected to be a PyTorch tensor of shape (num_samples, num_outputs).

            optimization_settings (dict, optional):
                A dictionary containing optimization-related settings. If a key is missing, default values will be used.
                Available settings:
                    - "num_epochs" (int, default=100): The maximum number of training epochs.
                    - "patience" (int, default=10): Number of epochs to wait before early stopping.
                    - "freeze_backbone" (bool, default=False): If True, the backbone of the model is frozen.
                    - "batch_size" (int, default=32): Batch size for the training.
                    - "optimizer" (torch.optim.Optimizer, default=torch.optim.AdamW): Optimizer for training.
                    - "wd" (float, default=1e-4): Weight decay (L2 regularization) coefficient.
                    - "clip_val" (float, default=1.0): Gradient clipping threshold.

            continual_learning_settings (dict, optional):
                A dictionary specifying continual learning (CL) configurations. If a key is missing, default values will be used.
                Available settings:
                    - "use_cl" (bool or str, default=False): If False, CL is disabled. Can be set to:
                        - `"event_trigger"`: Enables CL based on an event-triggering mechanism.
                        - `"sigmoid_schedule"`: Enables CL using a sigmoid-based schedule.
                    - "event_trigger" (dict, required if `use_cl == "event_trigger"`):
                        - "test_threshold" (float, default=0.0): Threshold on the log likelihood of the new point.
                    - "sigmoid_schedule" (dict, required if `use_cl == "sigmoid_schedule"`):
                        - "transition_window_ratio" (float, default=0.5): Ratio of transition window to time horizon.
                        - "location" (float or None, default=None): Center of the transition window. Defaults to `time_horizon / 2` if None.
                        - "time_horizon" (int, default=100): Total time horizon for the sigmoid schedule.
                        - "current_iteration" (int, default=0): Current iteration of the optimization.

        Returns:
            None: The function trains the model in place and does not return a value.

        Notes:
            - If continual learning (CL) is enabled, the function dynamically determines whether to retrain the model
              based on the specified method (`event_trigger` or `sigmoid_schedule`). TODO: Add full functionality.
        """

        # Default settings
        default_opt_settings = {
            "num_epochs": 10_000,
            "freeze_backbone": False,
            "patience": 100,
            "batch_size": 32,
            "optimizer": torch.optim.AdamW,
            "lr": 1e-3,
            "wd": 1e-4,
            "clip_val": 1.0,
        }

        default_cont_learning_settings = {
            "use_cl": False,  # Can be False, "event_trigger", or "sigmoid_schedule"
            "event_trigger": {
                "test_threshold": 0.0,  # Threshold on the log-likelihood for event triggering
            },
            "sigmoid_schedule": {
                "transition_window_ratio": 0.5,  # Ratio of the transition window to time horizon
                "location": None,  # Defaults to time_horizon / 2 if None
                "time_horizon": 100,  # Total time horizon for the optimization
                "current_iteration": 0,  # Current iteration of the optimization
            },
        }

        # Merge defaults with provided settings
        if optimization_settings is None:
            optimization_settings = default_opt_settings
        else:
            optimization_settings = {**default_opt_settings, **optimization_settings}

        # Merge defaults with provided settings
        if continual_learning_settings is None:
            cl_settings = default_cont_learning_settings
        else:
            cl_settings = {
                **default_cont_learning_settings,
                **continual_learning_settings,
            }

        # Make dataloader based on train_X, train_y
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset = [[train_X[i], train_y[i]] for i, _ in enumerate(train_X)]

        dataloader = DataLoader(
            dataset, shuffle=True, batch_size=optimization_settings["batch_size"]
        )

        # default is to always fully train
        full_training = True

        """
        # renormalize the data based on the mean and std of the data expext the last point
        mean = train_y[:-1].mean(dim=0)
        if len(train_y) > 2:
            std = train_y[:-1].std(dim=0)
        else:
            std = torch.ones_like(mean)

        # renorm^2 suuper ugly
        x_new = train_X[-1]
        y_new = (train_y[-1] * train_y.std(dim=0) + train_y.mean(dim=0) - mean) / std
        out = self.old_model(x_new.to(device))
        """
        log_likelihood = 0  # -out.val_loss_fn(y_new.to(device)).sum().item()

        if cl_settings["use_cl"]:
            method = cl_settings["use_cl"]

            if not old_model_params:
                raise ValueError(
                    "No old model parameters provided for continual learning."
                )
            else:
                self.old_model.load_state_dict(old_model_params)

            if method == "event_trigger":
                full_training = (
                    log_likelihood < cl_settings["event_trigger"]["test_threshold"]
                )

            elif method == "sigmoid_schedule":
                # Sigmoid-based probability
                transition_window_ratio = cl_settings["sigmoid_schedule"][
                    "transition_window_ratio"
                ]
                location = cl_settings["sigmoid_schedule"]["location"]
                time_horizon = cl_settings["sigmoid_schedule"]["time_horizon"]
                iteration = cl_settings["sigmoid_schedule"]["iteration"]

                if location is None:
                    location = (
                        time_horizon / 2
                    )  # Default to middle of time horizon if not provided

                full_training = self.sigmoid_retrain_schedule(
                    time_horizon, iteration, transition_window_ratio, location
                )

            else:
                raise ValueError(f"Unknown continual learning method: {method}")

        if not full_training:
            self.model.to(device)
            self.model.load_state_dict(self.old_model.state_dict())

            with torch.no_grad():
                x, y = x_new.to(device), y_new.to(device)
                out = self.model(x.reshape(1, -1))
                loss = out.train_loss_fn(y.reshape(1, -1), recursive_update=True)

            print("Recurive update of the last layer.")

        else:
            self.model.to(device)
            self.set_regularization_weight(self.model.kl_scale / len(train_y))
            param_list = [
                {
                    "params": self.model.head.parameters(),
                    "weight_decay": 0.0,
                },
            ]

            # freeze backbone
            if not optimization_settings["freeze_backbone"]:
                param_list.append(
                    {
                        "params": self.model.backbone.parameters(),
                        "weight_decay": optimization_settings["wd"],
                    }
                )

            optimizer = optimization_settings["optimizer"](
                lr=optimization_settings["lr"], params=param_list
            )

            best_loss = float("inf")
            epochs_no_improve = 0
            early_stop = False
            best_model_state = None  # To store the best model parameters

            for epoch in range(optimization_settings["num_epochs"] + 1):
                # early stopping
                if early_stop:
                    break

                self.model.train()
                running_loss = []

                for train_step, (x, y) in enumerate(dataloader):
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    out = self.model(x)
                    loss = out.train_loss_fn(y)  # vbll layer will calculate the loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), optimization_settings["clip_val"]
                    )
                    optimizer.step()
                    running_loss.append(loss.item())

                # Calculate average loss over the epoch
                avg_loss = sum(running_loss[-len(dataloader) :]) / len(dataloader)

                # Early stopping logic
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_state = self.model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= optimization_settings["patience"]:
                    early_stop = True

            # load best model
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                print("Early stopping at epoch ", epoch, " with loss ", best_loss)

    def set_regularization_weight(self, new_weight: float):
        self.model.head.regularization_weight = new_weight

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise=False,
        posterior_transform=None,
    ) -> Posterior:
        if len(X.shape) < 3:
            B, D = X.shape
            Q = 1
            batched = False
        else:
            B, Q, D = X.shape
            X = X.reshape(B * Q, D)
            batched = True

        K = self.num_outputs
        posterior = self.model(X).predictive  #

        # Extract mean and variance
        mean = posterior.mean.squeeze(-1)
        variance = posterior.variance.squeeze(-1)
        cov = torch.diag_embed(variance)

        # TODO: may need some further reshaping for batch size > 1
        # Mean in `(batch_shape, q*k)`
        # mean = mean.reshape(B, Q * K)

        # # Cov is `(batch_shape, q*k, q*k)`
        # cov += 1e-4 * torch.eye(B * Q * K).to(X)
        # cov = cov.reshape(B, Q, K, B, Q, K)
        # cov = torch.einsum('bqkbrl->bqkrl', cov)  # (B, Q, K, Q, K)
        # cov = cov.reshape(B, Q * K, Q * K)

        dist = MultivariateNormal(mean, cov)
        post_pred = GPyTorchPosterior(dist)

        if batched:
            return BLLPosterior(post_pred, self, X, self.num_outputs)
        else:
            return post_pred

    @abstractmethod
    def sample(self, sample_shape: Optional[torch.Size] = None):
        raise NotImplementedError


class VBLLModel(AbstractBLLModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = VBLLNetwork(*args, **kwargs)
        self.old_model = VBLLNetwork(*args, **kwargs)  # used for continual learning

    def sample(self, sample_shape: Optional[torch.Size] = None):
        return self.model.sample_posterior_function(sample_shape)

    def __str__(self):
        return self.model.__str__()
