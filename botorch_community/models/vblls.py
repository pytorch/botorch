#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains an implemenation of a Variational Bayesian Last Layer (VBLL) model that can be used within BoTorch
for Bayesian optimization.

References:

[1] P. Brunzema, M. Jordahn, J. Willes, S. Trimpe, J. Snoek, J. Harrison.
    Bayesian Optimization via Contrinual Variational Last Layer Training.
    International Conference on Learning Representations, 2025.

Contributor: brunzema
"""

from typing import Dict, Optional, Type, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from gpytorch.distributions import MultivariateNormal

from botorch.posteriors import Posterior
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior

from botorch_community.posteriors.bll_posterior import BLLPosterior


torch.set_default_dtype(torch.float64)


class SampleModel(nn.Module):
    def __init__(self, backbone: nn.Module, sampled_params: Tensor):
        super().__init__()
        self.backbone = backbone
        self.sampled_params = sampled_params

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)

        if self.sampled_params.dim() == 2:
            return (self.sampled_params @ x[..., None]).squeeze(-1)

        x_expanded = x.unsqueeze(0).expand(self.sampled_params.shape[0], -1, -1)
        return torch.matmul(self.sampled_params, x_expanded.transpose(-1, -2))


class VBLLNetwork(nn.Module):
    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 64,
        out_features: int = 1,
        num_layers: int = 3,
        prior_scale: float = 1.0,
        wishart_scale: float = 0.01,
        kl_scale: float = 1.0,
        backbone: nn.Module = None,
        activation: nn.Module = nn.ELU(),
        device=None,
    ):
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
            - If a `backbone` module is provided, it is applied before the variational last layer. If not, we use a default MLP structure.
        """
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

        # could be changed to other vbll regression layers
        self.head = Regression(
            hidden_features,
            out_features,
            regularization_weight=1.0,  # will be adjusted dynamically at each iteration based on the number of data points
            prior_scale=prior_scale,
            wishart_scale=wishart_scale,
            parameterization="dense",
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        return self.head(x)

    def sample_posterior_function(
        self, sample_shape: Optional[torch.Size] = None
    ) -> nn.Module:
        """
        Samples a posterior function by drawing parameters from the model's learned distribution.

        Args:
            sample_shape (Optional[torch.Size], optional):
                The desired shape for the sampled parameters. If None, a single sample is drawn.
                Defaults to None.

        Returns:
            nn.Module[[Tensor], Tensor]:
                A nn.Module that takes an input tensor `x` and returns the corresponding
                model output tensor. The function applies the backbone transformation
                and computes the final output using the sampled parameters.

        Notes:
            - If `sample_shape` is None, a single set of parameters is sampled.
            - If `sample_shape` is provided, multiple parameter samples are drawn, and the function
              will return a batched output where the first dimension corresponds to different samples.
        """
        sampled_params = (
            self.head.W().rsample(sample_shape).to(self.device)
            if sample_shape
            else self.head.W().rsample().to(self.device)
        )
        return SampleModel(self.backbone, sampled_params)


def _get_optimizer(
    optimizer_class: Type[Optimizer],
    model_parameters,
    lr: float = 1e-3,
    **kwargs,
) -> Optimizer:
    """
    Creates and returns an optimizer.

    Args:
        optimizer_class (Type[Optimizer]): The optimizer class (e.g., torch.optim.AdamW).
        model_parameters: Parameters to be optimized.
        lr (float): Learning rate.
        **kwargs: Additional arguments to be passed to the optimizer.

    Returns:
        Optimizer: The initialized optimizer.
    """
    return optimizer_class(model_parameters, lr=lr, **kwargs)


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

    @property
    def backbone(self):
        return self.model.backbone

    def fit(
        self,
        train_X: Tensor,
        train_y: Tensor,
        optimization_settings: Optional[Dict] = None,
        initialization_params: Optional[Dict] = None,
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

            initialization_params (dict, optional):
                A dictionary containing the initial parameters of the model for feature reuse.
                If None, the optimization will start from from the random initialization in the __init__ method.

        Returns:
            None: The function trains the model in place and does not return a value.
        """

        # Default settings
        default_opt_settings = {
            "num_epochs": 10_000,
            "freeze_backbone": False,
            "patience": 100,
            "batch_size": 32,
            "optimizer": torch.optim.AdamW,  # Now uses a class, not an instance
            "lr": 1e-3,
            "wd": 1e-4,
            "clip_val": 1.0,
            "optimizer_kwargs": {},  # Extra optimizer-specific args (e.g., betas for Adam)
        }

        # Merge defaults with provided settings
        optimization_settings = (
            default_opt_settings
            if optimization_settings is None
            else {**default_opt_settings, **optimization_settings}
        )

        # Make dataloader based on train_X, train_y
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset = [[train_X[i], train_y[i]] for i, _ in enumerate(train_X)]

        dataloader = DataLoader(
            dataset, shuffle=True, batch_size=optimization_settings["batch_size"]
        )

        if initialization_params is not None:
            self.model.load_state_dict(initialization_params)

        self.model.to(device)
        self.set_reg_weight(self.model.kl_scale / len(train_y))
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

        # Extract settings
        optimizer_class = optimization_settings["optimizer"]
        optimizer_kwargs = optimization_settings.get("optimizer_kwargs", {})

        # Initialize optimizer using helper function
        optimizer = _get_optimizer(
            optimizer_class,
            model_parameters=param_list,
            lr=optimization_settings["lr"],
            **optimizer_kwargs,
        )

        best_loss = float("inf")
        epochs_no_improve = 0
        early_stop = False
        best_model_state = None  # To store the best model parameters

        self.model.train()

        for epoch in range(1, optimization_settings["num_epochs"] + 1):
            # early stopping
            if early_stop:
                break

            running_loss = []

            for train_step, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = out.train_loss_fn(y)  # vbll layer will calculate the loss

                loss.backward()

                if optimization_settings["clip_val"] is not None:
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

    def set_reg_weight(self, new_weight: float):
        self.model.head.regularization_weight = new_weight

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise=None,
        posterior_transform=None,
    ) -> Posterior:
        # Determine if the input is batched
        batched = X.dim() == 3

        if not batched:
            N, D = X.shape
            B = 1
        else:
            B, N, D = X.shape
            X = X.reshape(B * N, D)

        posterior = self.model(X).predictive

        # Extract mean and variance
        mean = posterior.mean.squeeze()
        variance = posterior.variance.squeeze()
        cov = torch.diag_embed(variance)

        K = self.num_outputs
        mean = mean.reshape(B, N * K)

        # Cov must be `(B, Q*K, Q*K)`
        cov = cov.reshape(B, N, K, B, N, K)
        cov = torch.einsum("bqkbrl->bqkrl", cov)  # (B, Q, K, Q, K)
        cov = cov.reshape(B, N * K, N * K)

        # Remove fake batch dimension if not batched
        if not batched:
            mean = mean.squeeze(0)
            cov = cov.squeeze(0)

        # pass as MultivariateNormal to GPyTorchPosterior
        mvn_dist = MultivariateNormal(mean, cov)
        post_pred = GPyTorchPosterior(mvn_dist)
        return BLLPosterior(post_pred, self, X, self.num_outputs)

    @abstractmethod
    def sample(self, sample_shape: Optional[torch.Size] = None):
        raise NotImplementedError


class VBLLModel(AbstractBLLModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = VBLLNetwork(*args, **kwargs)

    def sample(self, sample_shape: Optional[torch.Size] = None):
        return self.model.sample_posterior_function(sample_shape)

    def __str__(self):
        return self.model.__str__()


#######################################################################################################################
# The following code is from the repository vbll (https://github.com/VectorInstitute/vbll) which is under the MIT license
# The code is from the paper "Variational Bayesian Last Layers" by Harrison et al., ICLR 2024
#######################################################################################################################


# following functions/classes are from https://github.com/VectorInstitute/vbll/blob/main/vbll/utils/distributions.py
def get_parameterization(p):
    if p in cov_param_dict:
        return cov_param_dict[p]
    else:
        raise ValueError("Must specify a valid covariance parameterization.")


def tp(M):
    return M.transpose(-1, -2)


def sym(M):
    return (M + tp(M)) / 2.0


# Credit to https://github.com/brentyi/fannypack/blob/2888aa5d969824ac1e1a528264674ece3f4703f9/fannypack/utils/_math.py
def cholesky_inverse(u: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """Alternative to `torch.cholesky_inverse()`, with support for batch dimensions.

    Relevant issue tracker: https://github.com/pytorch/pytorch/issues/7500

    Args:
        u (torch.Tensor): Triangular Cholesky factor. Shape should be `(*, N, N)`.
        upper (bool, optional): Whether to consider the Cholesky factor as a lower or
            upper triangular matrix.

    Returns:
        torch.Tensor:
    """
    if u.dim() == 2 and not u.requires_grad:
        return torch.cholesky_inverse(u, upper=upper)
    return torch.cholesky_solve(torch.eye(u.size(-1)).expand(u.size()), u, upper=upper)


class Normal(torch.distributions.Normal):
    def __init__(self, loc, chol):
        super(Normal, self).__init__(loc, chol)

    @property
    def mean(self):
        return self.loc

    @property
    def var(self):
        return self.scale**2

    @property
    def chol_covariance(self):
        return torch.diag_embed(self.scale)

    @property
    def covariance_diagonal(self):
        return self.var

    @property
    def covariance(self):
        return torch.diag_embed(self.var)

    @property
    def precision(self):
        return torch.diag_embed(1.0 / self.var)

    @property
    def logdet_covariance(self):
        return 2 * torch.log(self.scale).sum(-1)

    @property
    def logdet_precision(self):
        return -2 * torch.log(self.scale).sum(-1)

    @property
    def trace_covariance(self):
        return self.var.sum(-1)

    @property
    def trace_precision(self):
        return (1.0 / self.var).sum(-1)

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (self.var.unsqueeze(-1) * (b**2)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((b**2) / self.var.unsqueeze(-1)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __add__(self, inp):
        if isinstance(inp, Normal):
            new_cov = self.var + inp.var
            return Normal(
                self.mean + inp.mean, torch.sqrt(torch.clip(new_cov, min=1e-12))
            )
        elif isinstance(inp, torch.Tensor):
            return Normal(self.mean + inp, self.scale)
        else:
            raise NotImplementedError(
                "Distribution addition only implemented for diag covs"
            )

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def squeeze(self, idx):
        return Normal(self.loc.squeeze(idx), self.scale.squeeze(idx))


class DenseNormal(torch.distributions.MultivariateNormal):
    def __init__(self, loc, cholesky):
        super(DenseNormal, self).__init__(loc, scale_tril=cholesky)

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        return self.scale_tril

    @property
    def covariance(self):
        return self.scale_tril @ tp(self.scale_tril)

    @property
    def inverse_covariance(self):
        warnings.warn(
            "Direct matrix inverse for dense covariances is O(N^3), consider using eg inverse weighted inner product"
        )
        return tp(torch.linalg.inv(self.scale_tril)) @ torch.linalg.inv(self.scale_tril)

    @property
    def logdet_covariance(self):
        return 2.0 * torch.diagonal(self.scale_tril, dim1=-2, dim2=-1).log().sum(-1)

    @property
    def trace_covariance(self):
        return (self.scale_tril**2).sum(-1).sum(-1)  # compute as frob norm squared

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((tp(self.scale_tril) @ b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (torch.linalg.solve(self.scale_tril, b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def squeeze(self, idx):
        return DenseNormal(self.loc.squeeze(idx), self.scale_tril.squeeze(idx))


class LowRankNormal(torch.distributions.LowRankMultivariateNormal):
    def __init__(self, loc, cov_factor, diag):
        super(LowRankNormal, self).__init__(loc, cov_factor=cov_factor, cov_diag=diag)

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        raise NotImplementedError()

    @property
    def covariance(self):
        return self.cov_factor @ tp(self.cov_factor) + torch.diag_embed(self.cov_diag)

    @property
    def inverse_covariance(self):
        raise NotImplementedError()

    @property
    def logdet_covariance(self):
        # Apply Matrix determinant lemma
        term1 = torch.log(self.cov_diag).sum(-1)
        arg1 = tp(self.cov_factor) @ (self.cov_factor / self.cov_diag.unsqueeze(-1))
        term2 = torch.linalg.det(arg1 + torch.eye(arg1.shape[-1])).log()
        return term1 + term2

    @property
    def trace_covariance(self):
        # trace of sum is sum of traces
        trace_diag = self.cov_diag.sum(-1)
        trace_lowrank = (self.cov_factor**2).sum(-1).sum(-1)
        return trace_diag + trace_lowrank

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        diag_term = (self.cov_diag.unsqueeze(-1) * (b**2)).sum(-2)
        factor_term = ((tp(self.cov_factor) @ b) ** 2).sum(-2)
        prod = diag_term + factor_term
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        raise NotImplementedError()

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def squeeze(self, idx):
        return LowRankNormal(
            self.loc.squeeze(idx),
            self.cov_factor.squeeze(idx),
            self.cov_diag.squeeze(idx),
        )


class DenseNormalPrec(torch.distributions.MultivariateNormal):
    """A DenseNormal parameterized by the mean and the cholesky decomp of the precision matrix.

    This function also includes a recursive_update function which performs a recursive
    linear regression update with effecient cholesky factor updates.
    """

    def __init__(self, loc, cholesky, validate_args=False):
        prec = cholesky @ tp(cholesky)
        super(DenseNormalPrec, self).__init__(
            loc, precision_matrix=prec, validate_args=validate_args
        )
        self.tril = cholesky

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        raise NotImplementedError()

    @property
    def covariance(self):
        warnings.warn(
            "Direct matrix inverse for dense covariances is O(N^3), consider using eg inverse weighted inner product"
        )
        return cholesky_inverse(self.tril)

    @property
    def inverse_covariance(self):
        return self.precision_matrix

    @property
    def logdet_covariance(self):
        return -2.0 * torch.diagonal(self.tril, dim1=-2, dim2=-1).log().sum(-1)

    @property
    def trace_covariance(self):
        return (
            (torch.inverse(self.tril) ** 2).sum(-1).sum(-1)
        )  # compute as frob norm squared

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (torch.linalg.solve(self.tril, b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((tp(self.tril) @ b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def squeeze(self, idx):
        return DenseNormalPrec(self.loc.squeeze(idx), self.tril.squeeze(idx))


cov_param_dict = {
    "dense": DenseNormal,
    "dense_precision": DenseNormalPrec,
    "diagonal": Normal,
    "lowrank": LowRankNormal,
}


# following functions/classes are from https://github.com/VectorInstitute/vbll/blob/main/vbll/layers/regression.py
def gaussian_kl(p, q_scale):
    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean**2).sum(-1).sum(-1) / q_scale
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)
    return 0.5 * (mse_term + trace_term + logdet_term)  # currently exclude constant


@dataclass
class VBLLReturn:
    predictive: Normal | DenseNormal | torch.distributions.studentT.StudentT
    train_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    val_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ood_scores: None | Callable[[torch.Tensor], torch.Tensor] = None


class Regression(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        regularization_weight,
        parameterization="dense",
        prior_scale=1.0,
        wishart_scale=1e-2,
        cov_rank=None,
        dof=1.0,
    ):
        """
        Variational Bayesian Linear Regression

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        regularization_weight : float
            Weight on regularization term in ELBO
        parameterization : str
            Parameterization of covariance matrix. Currently supports {'dense', 'diagonal', 'lowrank', 'dense_precision'}
        prior_scale : float
            Scale of prior covariance matrix
        wishart_scale : float
            Scale of Wishart prior on noise covariance
        dof : float
            Degrees of freedom of Wishart prior on noise covariance
        """
        super(Regression, self).__init__()

        self.wishart_scale = wishart_scale
        self.dof = (dof + out_features + 1.0) / 2.0
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_scale = prior_scale * (1.0 / in_features)

        # noise distribution
        self.noise_mean = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.noise_logdiag = nn.Parameter(
            torch.randn(out_features) * (np.log(wishart_scale))
        )

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features))

        if parameterization == "diagonal":
            self.W_logdiag = nn.Parameter(
                torch.randn(out_features, in_features) - 0.5 * np.log(in_features)
            )
        elif parameterization == "dense":
            self.W_logdiag = nn.Parameter(
                torch.randn(out_features, in_features) - 0.5 * np.log(in_features)
            )
            self.W_offdiag = nn.Parameter(
                torch.randn(out_features, in_features, in_features) / in_features
            )
        elif parameterization == "dense_precision":
            self.W_logdiag = nn.Parameter(
                torch.randn(out_features, in_features) + 0.5 * np.log(in_features)
            )
            self.W_offdiag = nn.Parameter(
                torch.randn(out_features, in_features, in_features) * 0.0
            )
        elif parameterization == "lowrank":
            self.W_logdiag = nn.Parameter(
                torch.randn(out_features, in_features) - 0.5 * np.log(in_features)
            )
            self.W_offdiag = nn.Parameter(
                torch.randn(out_features, in_features, cov_rank) / in_features
            )

    def W(self):
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist == Normal:
            cov = self.W_dist(self.W_mean, cov_diag)
        elif (self.W_dist == DenseNormal) or (self.W_dist == DenseNormalPrec):
            tril = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            cov = self.W_dist(self.W_mean, tril)
        elif self.W_dist == LowRankNormal:
            cov = self.W_dist(self.W_mean, self.W_offdiag, cov_diag)

        return cov

    def noise(self):
        return Normal(self.noise_mean, torch.exp(self.noise_logdiag))

    def forward(self, x):
        out = VBLLReturn(
            self.predictive(x), self._get_train_loss_fn(x), self._get_val_loss_fn(x)
        )
        return out

    def predictive(self, x):
        return (self.W() @ x[..., None]).squeeze(-1) + self.noise()

    def _get_train_loss_fn(self, x):
        def loss_fn(y):
            # construct predictive density N(W @ phi, Sigma)
            W = self.W()
            noise = self.noise()
            pred_density = Normal((W.mean @ x[..., None]).squeeze(-1), noise.scale)
            pred_likelihood = pred_density.log_prob(y)

            trace_term = 0.5 * (
                (W.covariance_weighted_inner_prod(x.unsqueeze(-2)[..., None]))
                * noise.trace_precision
            )

            kl_term = gaussian_kl(W, self.prior_scale)
            wishart_term = (
                self.dof * noise.logdet_precision
                - 0.5 * self.wishart_scale * noise.trace_precision
            )
            total_elbo = torch.mean(pred_likelihood - trace_term)
            total_elbo += self.regularization_weight * (wishart_term - kl_term)
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            # compute log likelihood under variational posterior via marginalization
            logprob = self.predictive(x).log_prob(y).sum(-1)  # sum over output dims
            return -logprob.mean(0)  # mean over batch dim

        return loss_fn
