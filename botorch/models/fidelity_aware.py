#! /usr/bin/env python3

from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.lazy import DiagLazyTensor, LazyTensor
from gpytorch.likelihoods import GaussianLikelihood, _GaussianLikelihoodBase
from gpytorch.means import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.module import Module
from gpytorch.priors import GammaPrior
from torch import Tensor
from torch.nn.functional import softplus

from .fantasy_utils import _get_fantasy_state, _load_fantasy_state_dict
from .gp_regression import SingleTaskGP
from .gpytorch import GPyTorchModel


class FidelityAwareHeteroskedasticNoise(Module):
    """A noise model that allows scaling the noise by a fidelity level

    Args:
        noise_model: A pytorch module mapping a `(b) x n x d`-dimensional tensor
            to a MultivariateNormal distribution, the mean of which is used as
            the out-of sample prediction of the noise level.
        x_idxr: A LongTensor with `d` elements, used for indexing the elements
            of a tensor with last dimension of size `d + d'` to be fed into the
            noise model.
        phi_idxr: A LongTensor with `d` elements, used for indexing the noise-only
            parameter(s). This is the complement of `x_idxr`.
        noise_tranform (optional): A parameter transform applied to the output
            of the noise_model.
        phi_func (optional): A callable mapping the noise-only feature(s) to
            normalized fidelities.

    For "actual" parameters `x` and fidelity parameters `phi`, the noise level
    returned is `noise_transform(noise_model(x)) / phi_func(phi)`

    """

    def __init__(
        self,
        noise_model: Module,
        x_idxr: Tensor,
        phi_idxr: Tensor,
        noise_transform: Callable[[Tensor], Tensor] = torch.exp,
        phi_func: Callable[[Tensor], Tensor] = lambda phi: phi,
    ) -> None:
        super().__init__()
        self.noise_model = noise_model
        self._x_idxr, self._phi_idxr = x_idxr, phi_idxr
        self._noise_transform = noise_transform
        self._phi_func = phi_func

    def forward(
        self, features: Tensor, shape: Optional[torch.Size] = None
    ) -> LazyTensor:
        if not torch.is_tensor(features):
            # TODO: Ensure consistent behavior in train and eval modes
            features = features[0]
        x, phi = features[..., self._x_idxr], features[..., self._phi_idxr]
        output = self.noise_model(x)
        noise_diag = self._noise_transform(output.mean) / self._phi_func(phi.squeeze())
        return DiagLazyTensor(noise_diag)


class FidelityAwareSingleTaskGP(ExactGP, GPyTorchModel):
    """A model supporting observation noise scaling by a provided fidelity level.

    Args:
        train_X: A `(b) x n x (d + d')`-dimensional tensor of input features,
            where `d` features alter both the normalized observation noise and
            the latent function, and `d'` noise-only features only re-scale the
            observation noise. The second set of features are strictly for fidelity.
        train_Y: A `(b) x n`-dimensional tensor of output observations.
        train_Y_se: A `(b) x n`-dimensional tensor of observed measurement noise.
            This should not be normalized.
        phi_idcs: The index or indices corresponding to the noise-only features.
        phi_func: A callable mapping the noise-only features to the fidelity level.
            The variance of the measurement noise is assumed to behave as
            `~ 1 / phi_func(phi)`. For the typical use case where `phi` is the
            sample size of i.i.d. observations, `phi_func` is the identity.

    This model internally utilizes a heteroskedastic noise model of the normalized
    observation noise over all non-noise-only features. When evaluating, the
    normalized observation noise predicted at the provided features is scaled
    approprirately by the value provided by `phi_func` evaluated on the noise-only
    features.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Y_se: Tensor,
        phi_idcs: Union[int, Iterable[int]] = -1,
        phi_func: Callable[[Tensor], Tensor] = lambda phi: phi,
    ) -> None:
        self._phi_idcs = phi_idcs
        x_idxr, phi_idxr = _make_phi_indexers(X=train_X, phi_idcs=phi_idcs)
        self._x_idxr, self._phi_idxr = x_idxr, phi_idxr

        train_x = train_X[..., x_idxr]
        train_phi = train_X[..., phi_idxr]
        if train_x.ndimension() == 1:
            batch_size, ard_num_dims = 1, None
        if train_x.ndimension() == 2:
            batch_size, ard_num_dims = 1, train_x.shape[-1]
        elif train_x.ndimension() == 3:
            batch_size, ard_num_dims = train_x.shape[0], train_x.shape[-1]
        else:
            raise ValueError(f"Unsupported shape {train_X.shape} for train_X.")

        # Set up noise model
        train_y_log_var = (
            2 * train_Y_se.log() + phi_func(train_phi.view_as(train_Y_se)).log()
        )
        noise_likelihood = GaussianLikelihood(
            batch_size=batch_size, param_transform=softplus
        )
        noise_model = SingleTaskGP(
            train_X=train_x, train_Y=train_y_log_var, likelihood=noise_likelihood
        )
        noise_covar = FidelityAwareHeteroskedasticNoise(
            noise_model=noise_model, x_idxr=x_idxr, phi_idxr=phi_idxr, phi_func=phi_func
        )
        likelihood = _GaussianLikelihoodBase(noise_covar)
        super().__init__(
            train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        self.mean_module = ConstantMean(batch_size=batch_size)
        self.covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                lengthscale_prior=GammaPrior(2.0, 5.0),
                param_transform=softplus,
            ),
            batch_size=batch_size,
            param_transform=softplus,
            outputscale_prior=GammaPrior(1.1, 0.05),
        )

    def forward(self, z: Tensor) -> MultivariateNormal:
        x = z[..., self._x_idxr]
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def fantasize(
        self, X: Tensor, num_samples: int, base_samples: Optional[Tensor] = None
    ) -> "FidelityAwareSingleTaskGP":
        state_dict, train_X, train_Y = _get_fantasy_state(
            model=self, X=X, num_samples=num_samples, base_samples=base_samples
        )
        noise_covar = self.likelihood.noise_covar
        train_Y_log_var = noise_covar.noise_model.train_targets
        # for now use mean prediction for noise
        noise_fantasies = noise_covar(X).diag()
        # TODO: Use variance of noise predictions for fantasizing
        tYlv = torch.cat(
            [
                train_Y_log_var.expand(num_samples, -1),
                noise_fantasies.expand(num_samples, -1),
            ],
            dim=-1,
        )
        train_Y_se = torch.exp(0.5 * tYlv)
        fantasy_model = FidelityAwareSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Y_se=train_Y_se,
            phi_idcs=self._phi_idcs,
            phi_func=noise_covar._phi_func,
        )
        return _load_fantasy_state_dict(model=fantasy_model, state_dict=state_dict)

    def reinitialize(
        self, train_X: Tensor, train_Y: Tensor, train_Y_se: Optional[Tensor] = None
    ) -> None:
        """
        Reinitialize model and the likelihood.

        Note: this does not refit the model.
        """
        assert train_Y_se is not None
        self.__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Y_se=train_Y_se,
            phi_idcs=self._phi_idcs,
            phi_func=self.likelihood.noise_covar._phi_func,
        )


def _make_phi_indexers(
    X: Tensor, phi_idcs: Union[int, Iterable[int]]
) -> Tuple[Tensor, Tensor]:
    """Make utility indexers for indexing into a full tensor X.

    Args:
        X: The full tensor, including non-noise features.
        phi_idcs: The index (or indices) corresponding to the noise-only features
            in `X`. These always pertain to the last dimension of `X`.

    Returns:
        Tensor: A LongTensor `x_idxr` with all indices corresponding to the
            non-noise-only features in `X`. These can be accessed from `X` using
            `X[..., x_idxr]`
        Tensor: A LongTensor `phi_idxr` with all indices corresponding to the
            noise-only features in `X` (i.e. the complement of `x_idxr`). These
            can be accessed from `X` using `X[..., phi_idxr]`
    """
    all_idcs = torch.arange(X.shape[-1], device=X.device)
    phi_idxr = all_idcs[phi_idcs].view(-1)
    x_idcs = list(set(all_idcs.tolist()) - set(phi_idxr.tolist()))
    x_idxr = torch.tensor(x_idcs).type_as(phi_idxr)
    return x_idxr, phi_idxr
