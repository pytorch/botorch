#!/usr/bin/env python3

"""
Wraps the batch acquisition functions defined in botorch.acquisition.functional
into BatchAcquisitionFunction gpytorch modules.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import torch
from torch import Tensor

from ..models.model import Model
from ..utils.sampling import construct_base_samples
from ..utils.transforms import squeeze_last_dim
from .batch_utils import batch_mode_instance_method, match_batch_shape
from .functional.batch_acquisition import (
    batch_expected_improvement,
    batch_noisy_expected_improvement,
    batch_probability_of_improvement,
    batch_simple_regret,
    batch_upper_confidence_bound,
)
from .modules import AcquisitionFunction


class BatchAcquisitionFunction(AcquisitionFunction, ABC):
    """Abstract base class for batch acquisition functions."""

    def __init__(
        self,
        model: Model,
        mc_samples: int = 500,
        X_pending: Optional[Tensor] = None,
        seed: Optional[int] = None,
        qmc: Optional[bool] = True,
    ) -> None:
        super().__init__(model=model)
        self.mc_samples = mc_samples
        self.X_pending = X_pending
        if self.X_pending is not None:
            self.X_pending.requires_grad_(False)
        self.seed = seed
        self.qmc = qmc
        self.base_samples = None

    @property
    def _base_samples_q_batch_size(self) -> Optional[int]:
        """Size of base_sample's q-batch.

        Returns:
            The q-batch size of `base_samples` if `base_samples` are set,
                and `None` otherwise.
        """
        if self.base_samples is not None:
            return self.base_samples.shape[-2]

    def _set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        """Set pending points.

        Args:
            X_pending: A `b x m x d`-dim tensor of pending points.
        """
        self.X_pending = X_pending
        # Ensure we regenerate base_samples, which is stateful.
        self.base_samples = None

    @abstractmethod
    def _forward(self, X: Tensor) -> Tensor:
        """Takes in a `b x q x d` Tensor of `b` t-batches with `q` `d`-dim
        design points each, and returns a one-dimensional Tensor with `b`
        elements."""
        pass

    def forward(self, X: Tensor) -> Tensor:
        """Takes in a `(b) x q x d` X Tensor of `b` t-batches with `q` `d`-dim
        design points each, expands and concatenates `self.X_pending` and
        returns a one-dimensional Tensor with `b` elements."""
        if self.X_pending is not None:
            # Some batch acquisition functions like qKG without discretization
            # rely upon the order of points in this torch.cat. It must remain
            # [X_pending, X], not [X, X_pending] for this code to work properly.
            X = torch.cat([match_batch_shape(self.X_pending, X), X], dim=-2)
        # We need to construct base_samples if
        # (i) we do QMC without a fixed seed
        # (ii) if the seed is fixed but the shape of the base samples is
        #      incompatible with the input - this happens if (a) we haven't
        #      constructed any base_samples yet, or (b) the q-size of the input
        #      has changed between evaluations.
        if (self.qmc and self.seed is None) or (
            self.seed is not None and self._base_samples_q_batch_size != X.shape[-2]
        ):
            self._construct_base_samples(X)
        return self._forward(X)

    @batch_mode_instance_method
    def _construct_base_samples(self, X: Tensor) -> None:
        """Construct base samples (for QMC and/or fixed seed).

        Args:
            X: A `batch_shape x q x d`-dim Tensor of features.
        """
        output_shape = torch.Size([X.shape[-2], self.model.num_outputs])
        # TODO: Support batch evaluation of batached models
        batch_shape = torch.Size([1 for _ in X.shape[:-2]])  # collapse batch dimensions
        base_samples = construct_base_samples(
            batch_shape=batch_shape,
            output_shape=output_shape,
            sample_shape=torch.Size([self.mc_samples]),
            qmc=self.qmc,
            seed=self.seed,
            device=X.device,
            dtype=X.dtype,
        )
        self.base_samples = base_samples


class qExpectedImprovement(BatchAcquisitionFunction):
    """q-Expected Improvement with constraints."""

    def __init__(
        self,
        model: Model,
        best_f: float,
        objective: Callable[[Tensor], Tensor] = squeeze_last_dim,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        infeasible_cost: float = 0.0,
        mc_samples: int = 500,
        X_pending: Optional[Tensor] = None,
        qmc: Optional[bool] = True,
        seed: Optional[int] = None,
    ) -> None:
        """q-Expected Improvement with constraints.

        Args:
            model: A fitted model.
            best_f: The best (feasible) function value observed so far (assumed
                noiseless).
            objective: A callable mapping a Tensor of size `b x q x t` to a
                Tensor of size `b x q`, where `t` is the number of outputs of
                the model. This callable must support broadcasting.
                If omitted, squeeze the output dimension (applicable to single-
                output models only).
            constraints: A list of callables, each mapping a Tensor of size
                `b x q x t` to a Tensor of size `b x q`, where negative values
                imply feasibility. This callable must support broadcasting.
                Only relevant for multi-output models (`t` > 1).
            infeasible_cost: The infeasibility cost `M`. Should be set s.t.
                `-M < min_x obj(x)`.
            mc_samples: The number of (quasi-) Monte-Carlo samples to use for
                approximating the expectation.
            X_pending: A `m x d`-dim Tensor with `m` design points that are
                pending for evaluation.
            qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
            seed: If provided, perform deterministic optimization (i.e. the
                function to optimize is fixed and not stochastic).
        """
        super().__init__(
            model=model, mc_samples=mc_samples, X_pending=X_pending, qmc=qmc, seed=seed
        )
        self.best_f = best_f
        self.objective = objective
        self.constraints = constraints
        self.infeasible_cost = infeasible_cost

    def _forward(self, X: Tensor) -> Tensor:
        """Evaluate q-EI at design X.

        Args:
            X: A `b x q x d` Tensor with `b` t-batches of `q` design points each.

        Returns:
            Tensor: The q-EI value of the design X for each of the `b` t-batches.
        """
        return batch_expected_improvement(
            X=X,
            model=self.model,
            best_f=self.best_f,
            objective=self.objective,
            constraints=self.constraints,
            M=self.infeasible_cost,
            mc_samples=self.mc_samples,
            base_samples=self.base_samples,
        )


class qNoisyExpectedImprovement(BatchAcquisitionFunction):
    """q-Noisy Expected Improvement with constraints."""

    def __init__(
        self,
        model: Model,
        X_observed: Tensor,
        objective: Callable[[Tensor], Tensor] = squeeze_last_dim,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        infeasible_cost: float = 0.0,
        mc_samples: int = 500,
        X_pending: Optional[Tensor] = None,
        qmc: Optional[bool] = True,
        seed: Optional[int] = None,
    ) -> None:
        """q-Noisy Expected Improvement with constraints.

        Args:
            model: A fitted model.
            X_observed: A `q' x d`-dim Tensor of `q'` design points that have
                already been observed and would be considered as the best design
                point.
            objective: A callable mapping a Tensor of size `b x q x t` to a
                Tensor of size `b x q`, where `t` is the number of outputs of
                the model. This callable must support broadcasting. If omitted,
                squeeze the output dimension (applicable to single-output models
                only).
            constraints: A list of callables, each mapping a Tensor of size
                `b x q x t` to a Tensor of size `b x q`, where negative values
                imply feasibility. This callable must support broadcasting.
                Only relevant for multi-output models (`t` > 1).
            infeasible_cost: The infeasibility cost `M`. Should be set s.t.
                `-M < min_x obj(x)`.
            mc_samples: The number of (quasi-) Monte-Carlo samples to use for
                approximating the expectation.
            X_pending: A `m x d`-dim Tensor with `m` design points that are
                pending for evaluation.
            qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
            seed: If provided, perform deterministic optimization (i.e. the
                function to optimize is fixed and not stochastic).
        """
        super().__init__(
            model=model, mc_samples=mc_samples, X_pending=X_pending, qmc=qmc, seed=seed
        )
        # TODO: get X_observed from model?
        self.X_observed = X_observed
        self.objective = objective
        self.constraints = constraints
        self.infeasible_cost = infeasible_cost

    @batch_mode_instance_method
    def _construct_base_samples(self, X: Tensor) -> None:
        """Construct base samples (for QMC and/or fixed seed).

        Args:
            X: A `batch_shape x q x d`-dim Tensor of features.
        """
        # need to construct samples for the observed points as well
        n = X.shape[-2] + self.X_observed.shape[-2]
        output_shape = torch.Size([n, self.model.num_outputs])
        # TODO: Support batch evaluation of batached models
        batch_shape = torch.Size([1 for _ in X.shape[:-2]])  # collapse batch dimensions
        base_samples = construct_base_samples(
            batch_shape=batch_shape,
            output_shape=output_shape,
            sample_shape=torch.Size([self.mc_samples]),
            qmc=self.qmc,
            seed=self.seed,
            device=X.device,
            dtype=X.dtype,
        )
        self.base_samples = base_samples

    def _forward(self, X: Tensor) -> Tensor:
        """Evaluate q-NEI at design X.

        Args:
            X: A `b x q x d` Tensor with `b` t-batches of `q` design points each.

        Returns:
            The q-NoisyEI value of the design X for each of the `b` t-batches.
        """
        return batch_noisy_expected_improvement(
            X=X,
            model=self.model,
            X_observed=self.X_observed,
            objective=self.objective,
            constraints=self.constraints,
            M=self.infeasible_cost,
            mc_samples=self.mc_samples,
            base_samples=self.base_samples,
        )


class qProbabilityOfImprovement(BatchAcquisitionFunction):
    """q-Probability of Improvement."""

    def __init__(
        self,
        model: Model,
        best_f: float,
        mc_samples: int = 500,
        X_pending: Optional[Tensor] = None,
        qmc: Optional[bool] = True,
        seed: Optional[int] = None,
    ) -> None:
        """TODO: Revise!

        Args:
            model: A fitted model.
            best_f: The best (feasible) function value observed so far (assumed
                noiseless).
            mc_samples: The number of (quasi-) Monte-Carlo samples to use for
                approximating the probability of improvement.
            X_pending: A `m x d`-dim Tensor with `m` design points that are
                pending for evaluation.
            qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
            seed: If provided, perform deterministic optimization (i.e. the
                function to optimize is fixed and not stochastic).
        """
        super().__init__(
            model=model, mc_samples=mc_samples, X_pending=X_pending, qmc=qmc, seed=seed
        )
        self.best_f = best_f

    def _forward(self, X: Tensor) -> Tensor:
        """Evaluate q-PI at design X.

        Args:
            X: A `b x q x d` Tensor with `b` t-batches of `q` design points each.

        Returns:
            Tensor: The q-PI value of the design X for each of the `b` t-batches.
        """
        return batch_probability_of_improvement(
            X=X,
            model=self.model,
            best_f=self.best_f,
            mc_samples=self.mc_samples,
            base_samples=self.base_samples,
        )


class qSimpleRegret(BatchAcquisitionFunction):
    """q-Simple Regret."""

    def __init__(
        self,
        model: Model,
        mc_samples: int = 500,
        X_pending: Optional[Tensor] = None,
        qmc: Optional[bool] = True,
        seed: Optional[int] = None,
    ) -> None:
        """TODO: Revise

        Args:
            model: A fitted model.
            mc_samples: The number of (quasi-) Monte-Carlo samples to use for
                approximating the probability of improvement.
            X_pending: A `m x d`-dim Tensor with `m` design points that are
                pending for evaluation.
            qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
            seed: If provided, perform deterministic optimization (i.e. the
                function to optimize is fixed and not stochastic).
        """
        super().__init__(
            model=model, mc_samples=mc_samples, X_pending=X_pending, qmc=qmc, seed=seed
        )

    def _forward(self, X: Tensor) -> Tensor:
        """Evaluate q-SR at design X.

        Args:
            X: A `b x q x d` Tensor with `b` t-batches of `q` design points each.

        Returns:
            Tensor: The q-SR value of the design X for each of the `b` t-batches.
        """
        return batch_simple_regret(
            X=X,
            model=self.model,
            mc_samples=self.mc_samples,
            base_samples=self.base_samples,
        )


class qUpperConfidenceBound(BatchAcquisitionFunction):
    """q-Upper Confidence Bound."""

    def __init__(
        self,
        model: Model,
        beta: float,
        mc_samples: int = 500,
        X_pending: Optional[Tensor] = None,
        qmc: Optional[bool] = True,
        seed: Optional[int] = None,
    ) -> None:
        """TODO: Revise

        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            mc_samples: The number of (quasi-) Monte-Carlo samples to use for
                approximating the probability of improvement.
            X_pending: A `m x d`-dim Tensor with `m` design points that are
                pending for evaluation.
            qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
            seed: If provided, perform deterministic optimization (i.e. the
                function to optimize is fixed and not stochastic).
        """
        super().__init__(
            model=model, mc_samples=mc_samples, X_pending=X_pending, qmc=qmc, seed=seed
        )
        self.beta = beta

    def _forward(self, X: Tensor) -> Tensor:
        """Evaluate q-UCB at design X.

        Args:
            X: A `b x q x d` Tensor with `b` t-batches of `q` design points each.

        Returns:
            Tensor: The q-UCB value of the design X for each of the `b` t-batches.
        """
        return batch_upper_confidence_bound(
            X=X,
            model=self.model,
            beta=self.beta,
            mc_samples=self.mc_samples,
            seed=self.seed,
        )
