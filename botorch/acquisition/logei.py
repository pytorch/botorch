# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Batch implementations of the LogEI family of improvements-based acquisition functions.
"""


from __future__ import annotations

from functools import partial

from typing import Callable, List, Optional, TypeVar, Union

import torch
from botorch.acquisition.monte_carlo import SampleReducingMCAcquisitionFunction
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.exceptions.errors import BotorchError
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.safe_math import (
    fatmax,
    log_fatplus,
    log_softplus,
    logmeanexp,
    smooth_amax,
)
from torch import Tensor

"""
NOTE: On the default temperature parameters:

tau_relu: It is generally important to set `tau_relu` to be very small, in particular,
smaller than the expected improvement value. Otherwise, the optimization can stagnate.
By setting `tau_relu=1e-6` by default, stagnation is exceedingly unlikely to occur due
to the smooth ReLU approximation for practical applications of BO.
IDEA: We could consider shrinking `tau_relu` with the progression of the optimization.

tau_max: This is only relevant for the batch (`q > 1`) case, and `tau_max=1e-2` is
sufficient to get a good approximation to the maximum improvement in the batch of
candidates. If `fat=False`, the smooth approximation to the maximum can saturate
numerically. It is therefore recommended to use `fat=True` when optimizing batches
of `q > 1` points.
"""
TAU_RELU = 1e-6
TAU_MAX = 1e-2
FloatOrTensor = TypeVar("FloatOrTensor", float, Tensor)


class LogImprovementMCAcquisitionFunction(SampleReducingMCAcquisitionFunction):
    r"""
    Abstract base class for Monte-Carlo-based batch LogEI acquisition functions.

    :meta private:
    """

    _log: bool = True

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        eta: Union[Tensor, float] = 1e-3,
        fat: bool = True,
        tau_max: float = TAU_MAX,
    ) -> None:
        r"""
        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated using `get_sampler`.
                NOTE: For posteriors that do not support base samples,
                a sampler compatible with intended use case must be provided.
                See `ForkedRNGSampler` and `StochasticSampler` as examples.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated.
            constraints: A list of constraint callables which map a Tensor of posterior
                samples of dimension `sample_shape x batch-shape x q x m`-dim to a
                `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
                are satisfied if `constraint(samples) < 0`.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. See the docs of
                `compute_(log_)constraint_indicator` for more details on this parameter.
            fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
                approximation to the ReLU.
            tau_max: Temperature parameter controlling the sharpness of the
                approximation to the `max` operator over the `q` candidate points.
        """
        if isinstance(objective, ConstrainedMCObjective):
            raise BotorchError(
                "Log-Improvement should not be used with `ConstrainedMCObjective`."
                "Please pass the `constraints` directly to the constructor of the "
                "acquisition function."
            )
        q_reduction = partial(fatmax if fat else smooth_amax, tau=tau_max)
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            sample_reduction=logmeanexp,
            q_reduction=q_reduction,
            constraints=constraints,
            eta=eta,
            fat=fat,
        )
        self.tau_max = tau_max


class qLogExpectedImprovement(LogImprovementMCAcquisitionFunction):
    r"""MC-based batch Log Expected Improvement.

    This computes qLogEI by
    (1) sampling the joint posterior over q points,
    (2) evaluating the smoothed log improvement over the current best for each sample,
    (3) smoothly maximizing over q, and
    (4) averaging over the samples in log space.

    `qLogEI(X) ~ log(qEI(X)) = log(E(max(max Y - best_f, 0)))`,

    where `Y ~ f(X)`, and `X = (x_1,...,x_q)`.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qLogEI = qLogExpectedImprovement(model, best_f, sampler)
        >>> qei = qLogEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        eta: Union[Tensor, float] = 1e-3,
        fat: bool = True,
        tau_max: float = TAU_MAX,
        tau_relu: float = TAU_RELU,
    ) -> None:
        r"""q-Log Expected Improvement.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can be
                a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into `X` upon forward call. Copied and set to have no
                gradient.
            constraints: A list of constraint callables which map a Tensor of posterior
                samples of dimension `sample_shape x batch-shape x q x m`-dim to a
                `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
                are satisfied if `constraint(samples) < 0`.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. See the docs of
                `compute_(log_)smoothed_constraint_indicator` for details.
            fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
                approximation to the ReLU.
            tau_max: Temperature parameter controlling the sharpness of the smooth
                approximations to max.
            tau_relu: Temperature parameter controlling the sharpness of the smooth
                approximations to ReLU.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
            tau_max=check_tau(tau_max, name="tau_max"),
            fat=fat,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.tau_relu = check_tau(tau_relu, name="tau_relu")

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qLogExpectedImprovement on the candidate set `X`.

        Args:
            obj: `mc_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `mc_shape x batch_shape x q`-dim Tensor of expected improvement values.
        """
        li = _log_improvement(
            Y=obj,
            best_f=self.best_f,
            tau=self.tau_relu,
            fat=self._fat,
        )
        return li


"""
###################################### utils ##########################################
"""


def _log_improvement(
    Y: Tensor,
    best_f: Tensor,
    tau: Union[float, Tensor],
    fat: bool,
) -> Tensor:
    """Computes the logarithm of the softplus-smoothed improvement, i.e.
    `log_softplus(Y - best_f, beta=(1 / tau))`.
    Note that softplus is an approximation to the regular ReLU objective whose maximum
    pointwise approximation error is linear with respect to tau as tau goes to zero.

    Args:
        obj: `mc_samples x batch_shape x q`-dim Tensor of output samples.
        best_f: Best previously observed objective value(s), broadcastable with `obj`.
        tau: Temperature parameter for smooth approximation of ReLU.
            as `tau -> 0`, maximum pointwise approximation error is linear w.r.t. `tau`.
        fat: Toggles the logarithmic / linear asymptotic behavior of the
            smooth approximation to ReLU.

    Returns:
        A `mc_samples x batch_shape x q`-dim Tensor of improvement values.
    """
    log_soft_clamp = log_fatplus if fat else log_softplus
    Z = Y - best_f.to(Y)
    return log_soft_clamp(Z, tau=tau)  # ~ ((Y - best_f) / Y_std).clamp(0)


def check_tau(tau: FloatOrTensor, name: str) -> FloatOrTensor:
    """Checks the validity of the tau arguments of the functions below, and returns
    `tau` if it is valid."""
    if isinstance(tau, Tensor) and tau.numel() != 1:
        raise ValueError(name + f" is not a scalar: {tau.numel() = }.")
    if not (tau > 0):
        raise ValueError(name + f" is non-positive: {tau = }.")
    return tau
