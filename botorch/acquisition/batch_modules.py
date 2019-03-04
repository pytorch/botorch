#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import torch
from torch import Tensor

from ..models.model import Model
from ..utils import squeeze_last_dim
from .batch_utils import (
    batch_mode_instance_method,
    construct_base_samples_from_posterior,
    match_batch_shape,
)
from .functional.batch_acquisition import (
    batch_expected_improvement,
    batch_knowledge_gradient,
    batch_knowledge_gradient_no_discretization,
    batch_noisy_expected_improvement,
    batch_probability_of_improvement,
    batch_simple_regret,
    batch_upper_confidence_bound,
)
from .modules import AcquisitionFunction


"""
Wraps the batch acquisition functions defined in botorch.acquisition.functional
into BatchAcquisitionFunction gpytorch modules.
"""


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
        posterior = self.model.posterior(X)
        self.base_samples = construct_base_samples_from_posterior(
            posterior=posterior,
            num_samples=self.mc_samples,
            qmc=self.qmc,
            seed=self.seed,
        )


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
        seed: Optional[int] = None,
        qmc: Optional[bool] = True,
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
            seed: If provided, perform deterministic optimization (i.e. the
                function to optimize is fixed and not stochastic).
            qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
        """
        super().__init__(
            model=model, mc_samples=mc_samples, X_pending=X_pending, seed=seed, qmc=qmc
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
        seed: Optional[int] = None,
        qmc: Optional[bool] = True,
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
            seed: If provided, perform deterministic optimization (i.e. the
                function to optimize is fixed and not stochastic).
            qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
        """
        super().__init__(
            model=model, mc_samples=mc_samples, X_pending=X_pending, seed=seed, qmc=qmc
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
        X_all = torch.cat([X, match_batch_shape(self.X_observed, X)], dim=-2)
        posterior = self.model.posterior(X_all)
        self.base_samples = construct_base_samples_from_posterior(
            posterior=posterior,
            num_samples=self.mc_samples,
            qmc=self.qmc,
            seed=self.seed,
        )

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


class qKnowledgeGradient(BatchAcquisitionFunction):
    """Constrained, multi-fidelity q-Knowledge Gradient.

    This function supports t-batches in an inefficient manner.

    Multifidelity optimization can be performed by using the
    optional project and cost callables.
    """

    def __init__(
        self,
        model: Model,
        X_observed: Tensor,
        objective: Callable[[Tensor], Tensor] = squeeze_last_dim,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        mc_samples: int = 40,
        inner_mc_samples: int = 1000,
        project: Callable[[Tensor], Tensor] = lambda X: X,
        cost: Optional[Callable[[Tensor], Tensor]] = None,
        use_posterior_mean: Optional[bool] = False,
        X_pending: Optional[Tensor] = None,
        seed: Optional[int] = None,
        qmc: Optional[bool] = True,
    ) -> None:
        """TODO: Revise!

        Args:
            model: A fitted GPyTorchModel (required to efficiently generate fantasies)
            X_observed: A q' x d Tensor of q' design points that have already been
                observed and would be considered as the best design point.  A
                judicious filtering of the points here can massively
                speed up the function evaluation without altering the
                function if points that are highly unlikely to be the
                best (regardless of what is observed at X) are removed.
                For example, points that clearly do not satisfy the constraints
                or have terrible objective observations can be safely
                excluded from X_observed.
            objective: A callable mapping a Tensor of size `b x q x t` to a
                Tensor of size `b x q`, where `t` is the number of outputs of
                the model. This callable must support broadcasting. If omitted,
                squeeze the output dimension (applicable to single-output models
                only).
            constraints: A list of callables, each mapping a Tensor of size
                `b x q x t` to a Tensor of size `b x q`, where negative values
                imply feasibility. This callable must support broadcasting.
                Only relevant for multi-output models (`t` > 1).
            mc_samples: The number of Monte-Carlo samples to draw from the model
                posterior for the outer sample.  GP memory usage is multiplied by
                this value.
            inner_mc_samples:  The number of Monte-Carlo samples to draw for the
                inner expectation
            project:  A callable mapping a Tensor of size `b x (q + q') x d` to a
                Tensor of the same size.  Use for multi-fidelity optimization where
                the returned Tensor should be projected to the highest fidelity.
            cost: A callable mapping a Tensor of size `b x q x d` to a Tensor of
                size `b x 1`.  The resulting Tensor's value is the cost of submitting
                each t-batch.
            use_posterior_mean: If True, instead of sampling, the mean of the
                posterior is sent into the objective and constraints. Should be
                used for linear objectives without constraints.
            X_pending: A q' x d Tensor with q' design points that are pending for
                evaluation.
            seed: If seed is provided, do deterministic optimization where the
                function to optimize is fixed and not stochastic.
            qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
        """
        super().__init__(
            model=model, mc_samples=mc_samples, X_pending=X_pending, seed=seed, qmc=qmc
        )
        self.X_observed = X_observed
        self.objective = objective
        self.constraints = constraints
        self.inner_mc_samples = inner_mc_samples
        self.project = project
        self.cost = cost
        self.use_posterior_mean = use_posterior_mean

        self.inner_old_base_samples = None
        self.inner_new_base_samples = None
        self.fantasy_base_samples = None

    @batch_mode_instance_method
    def _construct_base_samples(self, X: Tensor) -> None:
        """TODO: Revise!"""
        # TODO: remove [0] in base_samples when qKG works with t-batches
        old_posterior = self.model.posterior(self.X_observed)
        self.inner_old_base_samples = construct_base_samples_from_posterior(
            posterior=old_posterior,
            num_samples=self.inner_mc_samples,
            qmc=self.qmc,
            seed=self.seed,
        )
        posterior = self.model.posterior(X)
        # Increment seed by one to get different base samples
        self.fantasy_base_samples = construct_base_samples_from_posterior(
            posterior=posterior,
            num_samples=self.mc_samples,
            qmc=self.qmc,
            seed=None if self.seed is None else self.seed + 1,
        )
        X_all = torch.cat([X, match_batch_shape(self.X_observed, X)], dim=-2)
        X_posterior = self.model.posterior(X=X, observation_noise=True)
        fantasy_y = X_posterior.rsample(
            sample_shape=torch.Size([self.mc_samples]),
            base_samples=self.fantasy_base_samples,
        )
        fantasy_model = self.model.get_fantasy_model(inputs=X, targets=fantasy_y)
        new_posterior = fantasy_model.posterior(
            X_all.unsqueeze(0).repeat(self.mc_samples, 1, 1)
        )
        # Increment seed by two to get different base samples
        self.inner_new_base_samples = construct_base_samples_from_posterior(
            posterior=new_posterior,
            num_samples=self.inner_mc_samples,
            qmc=self.qmc,
            seed=None if self.seed is None else self.seed + 2,
        )

    def _forward(self, X: Tensor) -> Tensor:
        """TODO: Revise!

        Args:
            X: A `b x q x d` Tensor with `b` t-batches of `q` design points each.

        Returns:
            Tensor: The q-KG value of the design X for each of the `b` t-batches.
        """
        # TODO: update this when batch_knowledge_gradient supports t-batches
        Xs = list(X) if X.dim() > 2 else [X]
        return torch.cat(
            [
                batch_knowledge_gradient(
                    X=Xi,
                    model=self.model,
                    X_observed=self.X_observed,
                    objective=self.objective,
                    constraints=self.constraints,
                    mc_samples=self.mc_samples,
                    inner_mc_samples=self.inner_mc_samples,
                    inner_old_base_samples=self.inner_old_base_samples,
                    inner_new_base_samples=self.inner_new_base_samples,
                    fantasy_base_samples=self.fantasy_base_samples,
                    project=self.project,
                    cost=self.cost,
                    use_posterior_mean=self.use_posterior_mean,
                ).reshape(1)
                for Xi in Xs
            ]
        )


class qKnowledgeGradientNoDiscretization(BatchAcquisitionFunction):
    """Constrained, multi-fidelity q-Knowledege Gradient w/o discretization.

    Multifidelity optimization can be performed by using the
    optional project and cost callables.

    Unlike qKnowledgeGradient, this optimizes qKG without using discretization.
    """

    def __init__(
        self,
        model: Model,
        objective: Callable[[Tensor], Tensor] = squeeze_last_dim,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        mc_samples: int = 20,
        inner_mc_samples: int = 100,
        project: Callable[[Tensor], Tensor] = lambda X: X,
        cost: Optional[Callable[[Tensor], Tensor]] = None,
        use_posterior_mean: Optional[bool] = False,
        X_pending: Optional[Tensor] = None,
        seed: Optional[int] = None,
        qmc: Optional[bool] = True,
    ) -> None:
        """TODO: Revise!

        Args:
            model: A fitted GPyTorchModel (req. to efficiently generate fantasies)
            objective: A callable mapping a Tensor of size `b x q x t` to a
                Tensor of size `b x q`, where `t` is the number of outputs of
                the model. This callable must support broadcasting.
                If omitted, squeeze the output dimension (applicable to single-
                output models only).
            constraints: A list of callables, each mapping a Tensor of size
                `b x q x t` to a Tensor of size `b x q`, where negative values
                imply feasibility. This callable must support broadcasting.
                Only relevant for multi-output models (`t` > 1).
            mc_samples: The number of Monte-Carlo samples to draw from the model
                posterior for the outer sample.  GP memory usage is multiplied by
                this value.  This is the number of fantasies that will be used.
            inner_mc_samples:  The number of Monte-Carlo samples to draw for the
                inner expectation over each fantasy.
            project:  A callable mapping a Tensor of size `b x (q + q') x d` to a
                Tensor of the same size.  Use for multi-fidelity optimization where
                the returned Tensor should be projected to the highest fidelity.
            cost: A callable mapping a Tensor of size `b x q x d` to a Tensor of
                size `b x 1`.  The resulting Tensor's value is the cost of submitting
                each t-batch.
            use_posterior_mean: If True, instead of sampling, the mean of the
                posterior is sent into the objective and constraints. Should be
                used for linear objectives without constraints.
            X_pending: A q' x d Tensor with q' design points that are pending for
                evaluation.
            seed: If seed is provided, do deterministic optimization where the
                function to optimize is fixed and not stochastic. A seed should be
                provided to this batch module.
            qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
        """
        super().__init__(
            model=model, mc_samples=mc_samples, X_pending=X_pending, seed=seed, qmc=qmc
        )
        self.objective = objective
        self.constraints = constraints
        self.inner_mc_samples = inner_mc_samples
        self.project = project
        self.cost = cost
        self.use_posterior_mean = use_posterior_mean

        self.inner_old_base_samples = None
        self.inner_new_base_samples = None
        self.fantasy_base_samples = None

    @batch_mode_instance_method
    def _construct_base_samples(self, X: Tensor) -> None:
        """TODO: Revise!"""
        # See _forward below for explanation
        X_actual = X[: (-self.mc_samples - 1), :]
        X_fantasies = X[(-self.mc_samples - 1) : -1, :]
        X_old = X[-1:, :]

        # TODO: remove [0] in base_samples when this works with t-batches
        old_posterior = self.model.posterior(X_old)
        self.inner_old_base_samples = construct_base_samples_from_posterior(
            posterior=old_posterior,
            num_samples=self.inner_mc_samples,
            qmc=self.qmc,
            seed=self.seed,
        )
        posterior = self.model.posterior(X_actual)
        # Increments seed by one to get different base samples
        self.fantasy_base_samples = construct_base_samples_from_posterior(
            posterior=posterior,
            num_samples=X_fantasies.shape[0],
            qmc=self.qmc,
            seed=None if self.seed is None else self.seed + 1,
        )
        X_posterior = self.model.posterior(X=X_actual, observation_noise=True)
        fantasy_y = X_posterior.rsample(
            sample_shape=torch.Size([X_fantasies.shape[0]]),
            base_samples=self.fantasy_base_samples,
        )
        fantasy_model = self.model.get_fantasy_model(inputs=X, targets=fantasy_y)
        new_posterior = fantasy_model.posterior(X_fantasies.unsqueeze(1))
        # Increments seed by two to get different base samples
        self.inner_new_base_samples = construct_base_samples_from_posterior(
            posterior=new_posterior,
            num_samples=self.inner_mc_samples,
            qmc=self.qmc,
            seed=None if self.seed is None else self.seed + 2,
        )

    def _forward(self, X: Tensor) -> Tensor:
        """TODO: Revise!

        Args:
            X: A `b x q x d` Tensor with `b` t-batches of `q` design points each.
                We split this X tensor into three parts. The first corresponds
                to the design we are trying to evaluate, and the other
                corresponds to X_fantasies and X_old. We require that:

                X_old = X[:, -1:, :]
                X_old.shape = b x 1 x d

                X_fantasies = X[:, (-mc_samples - 1):-1, :]
                X_fantasies.shape = b x mc_samples x d

                X_actual = X[:, :(-mc_samples - 1), :]

        Returns:
            A `???`-dim Tensor. For t-batch b, the constrained q-KG value of the
                design X_actual[b] averaged across the fantasy models where
                X_fantasies[b,i] is chosen as the final selection for the i-th
                fantasy model and X_old[b, 0] is chosen as the final selection
                for the previous model. The maximum across X_fantasies[b,i]
                and X_old[b,0] evaluated at X_actual is the true q-KG of X_actual.

        """
        # TODO: update this when the batch acquisition supports t-batches
        Xs = list(X) if X.dim() > 2 else [X]
        return torch.cat(
            [
                batch_knowledge_gradient_no_discretization(
                    X=Xi[: (-self.mc_samples - 1), :],
                    X_fantasies=Xi[(-self.mc_samples - 1) : -1, :],
                    X_old=Xi[-1:, :],
                    model=self.model,
                    objective=self.objective,
                    constraints=self.constraints,
                    inner_mc_samples=self.inner_mc_samples,
                    inner_old_base_samples=self.inner_old_base_samples,
                    inner_new_base_samples=self.inner_new_base_samples,
                    fantasy_base_samples=self.fantasy_base_samples,
                    project=self.project,
                    cost=self.cost,
                    use_posterior_mean=self.use_posterior_mean,
                ).reshape(1)
                for Xi in Xs
            ]
        )

    def extract_candidates(self, X: Tensor) -> Tensor:
        """We only return X_actual as the set of candidates post-optimization.

        Args:
            X: optimized `b x q x d` Tensor

        Returns:
            X_actual
        """
        return X[..., : (-self.mc_samples - 1), :]


class qProbabilityOfImprovement(BatchAcquisitionFunction):
    """q-Probability of Improvement."""

    def __init__(
        self,
        model: Model,
        best_f: float,
        mc_samples: int = 500,
        X_pending: Optional[Tensor] = None,
        seed: Optional[int] = None,
        qmc: Optional[bool] = True,
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
            seed: If provided, perform deterministic optimization (i.e. the
                function to optimize is fixed and not stochastic).
            qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
        """
        super().__init__(
            model=model, mc_samples=mc_samples, X_pending=X_pending, seed=seed, qmc=qmc
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
        seed: Optional[int] = None,
        qmc: Optional[bool] = True,
    ) -> None:
        """TODO: Revise

        Args:
            model: A fitted model.
            mc_samples: The number of (quasi-) Monte-Carlo samples to use for
                approximating the probability of improvement.
            X_pending: A `m x d`-dim Tensor with `m` design points that are
                pending for evaluation.
            seed: If provided, perform deterministic optimization (i.e. the
                function to optimize is fixed and not stochastic).
            qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
        """
        super().__init__(
            model=model, mc_samples=mc_samples, X_pending=X_pending, seed=seed, qmc=qmc
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
        seed: Optional[int] = None,
        qmc: Optional[bool] = True,
    ) -> None:
        """TODO: Revise

        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            mc_samples: The number of (quasi-) Monte-Carlo samples to use for
                approximating the probability of improvement.
            X_pending: A `m x d`-dim Tensor with `m` design points that are
                pending for evaluation.
            seed: If provided, perform deterministic optimization (i.e. the
                function to optimize is fixed and not stochastic).
            qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
        """
        super().__init__(
            model=model, mc_samples=mc_samples, X_pending=X_pending, seed=seed, qmc=qmc
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
