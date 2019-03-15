#! /usr/bin/env python3

"""
GP model for problems with known constant observation noise level.
"""


from typing import Optional

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.lazy import DiagLazyTensor, LazyTensor
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.means import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.module import Module
from gpytorch.priors import GammaPrior
from torch import Tensor

from .gpytorch import GPyTorchModel


class ConstantNoise(Module):
    """Constant noise model.

    Args:
        noise: A `(b1 x ... x bk)` tensor of noise levels, one associated
            with each t-batch. In the non-batched case, this is simply a
            scalar tensor.
    """

    def __init__(self, noise: Tensor) -> None:
        super().__init__()
        self.register_buffer("_noise_levels", noise.unsqueeze(-1))

    @property
    def noise(self):
        return self._noise_levels

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    def _set_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        self.initialize(_noise_levels=value.unsqueeze(-1))

    def forward(self, x: Tensor, shape: Optional[torch.Size] = None) -> LazyTensor:
        if not torch.is_tensor(x):
            # TODO: Ensure consistent behavior in train and eval modes (gpytorch)
            x = x[0]
        noise_diag = self._noise_levels.expand(x.shape[:-1])
        return DiagLazyTensor(noise_diag)


class ConstantNoiseGP(ExactGP, GPyTorchModel):
    """A model using a fixed noise level that allows fitting on different fidelity data.

    Args:
        train_X: A `(b1 x ... x bk) x n x d`-dimensional tensor of input features.
        train_Y: A `(b1 x ... x bk) x n`-dimensional tensor of output observations.
        train_Y_se: A float or `(b1 x ... x bk)`-dimensional tensor of observed
            measurement noise (constant across data points for each batch).
    """

    def __init__(self, train_X: Tensor, train_Y: Tensor, train_Y_se: Tensor) -> None:
        if train_X.ndimension() == 1:
            batch_size, ard_num_dims = 1, None
        if train_X.ndimension() == 2:
            batch_size, ard_num_dims = 1, train_X.shape[-1]
        elif train_X.ndimension() == 3:
            batch_size, ard_num_dims = train_X.shape[0], train_X.shape[-1]
        else:
            raise ValueError(f"Unsupported shape {train_X.shape} for train_X.")

        if not train_Y.shape[:-1] == train_Y_se.shape:
            raise ValueError(
                "train_Y and train_Y_se must have the same (t-)batch shape"
            )
        noise_covar = ConstantNoise(noise=train_Y_se ** 2)
        likelihood = _GaussianLikelihoodBase(noise_covar=noise_covar)
        super().__init__(
            train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        self.mean_module = ConstantMean(batch_size=batch_size)
        self.covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_size=batch_size,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )

    @property
    def num_outputs(self) -> int:
        return 1

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def reinitialize(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Y_se: Optional[Tensor] = None,
        keep_params: bool = True,
    ) -> None:
        """Reinitialize model and the likelihood given new data.

        This does not refit the model.
        If device/dtype of the new training data are different from that of the
        model, then the model is moved to the new device/dtype.

        Args:
            train_X: A tensor of new training data
            train_Y: A tensor of new training observations
            train_y_se: A tensor of new training noise observations
            keep_params: If True, keep the parameter values (speeds up refitting
                on similar data)
        """
        if train_Y_se is None:
            raise RuntimeError("ConstantNoiseGP requires observation noise")
        if keep_params:
            noise = train_Y_se.mean(dim=-1, keepdim=True) ** 2
            self.likelihood.noise_covar = ConstantNoise(noise=noise)
            self.set_train_data(inputs=train_X, targets=train_Y, strict=False)
        else:
            self.__init__(train_X=train_X, train_Y=train_Y, train_Y_se=train_Y_se)
        # move to new device / dtype if necessary
        self.to(device=train_X.device, dtype=train_X.dtype)
