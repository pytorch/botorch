#!/usr/bin/env python3

"""
Batch acquisition functions using the reparameterization trick in combination
with MC sampling.

.. [Wilson2017reparam] Wilson, J. T., Moriconi, R., Hutter, F., & Deisenroth,
    M. P. (2017). The reparameterization trick for acquisition functions.
    arXiv preprint arXiv:1712.00424.
"""

from math import pi, sqrt
from typing import Callable, List, Optional

import torch
from torch import Tensor

from ...models import Model
from ...optim.outcome_constraints import soft_eval_constraint
from ...utils import squeeze_last_dim
from ..batch_utils import batch_mode_transform


def apply_constraints_nonnegative_soft_(
    obj: Tensor,
    constraints: List[Callable[[Tensor], Tensor]],
    samples: Tensor,
    eta: float,
) -> None:
    """Revise!
    TODO: Get rid of this in favor of `apply_constraints_`.
    """
    if constraints is not None:
        obj.clamp_min_(0)  # Enforce non-negativity with constraints
        for constraint in constraints:
            obj.mul_(soft_eval_constraint(constraint(samples), eta=eta))


def apply_constraints_(
    obj: Tensor,
    constraints: List[Callable[[Tensor], Tensor]],
    samples: Tensor,
    M: float,
) -> None:
    """Apply constraints using an infeasiblity penalty.

    Assigsn a penalty of `-M` when not feasible, where obj is modified in-place.

    Args:
        obj: A `b x q` Tensor of objective values.
        constraints: A list of callables, each mapping a Tensor of size `b x q x t`
            to a Tensor of size `b x q`, where negative values imply feasibility.
            This callable must support broadcasting. Only relevant for multi-
            output models (`t` > 1).
        samples: A `b x q x t` Tensor of samples drawn from the posterior.
        M: The infeasible value.
    """
    if constraints is not None:
        # obj has dimensions n_samples x b x q
        # all_feasible has dimensions n_samples x b x q and tracks whether or not
        # each design point is feasible, i.e., for all constraints
        all_con_feasible = torch.ones(obj.shape, dtype=torch.uint8, device=obj.device)
        for constraint in constraints:
            # con has dimensions n_samples x b x q
            con = constraint(samples)
            this_con_feasible = con <= 0
            all_con_feasible.mul_(this_con_feasible)
        obj[~all_con_feasible] = -M


def get_infeasible_cost(
    X: Tensor, model: Model, objective: Callable[[Tensor], Tensor] = squeeze_last_dim
) -> float:
    """Get infeasible cost for a model and objective.

    Computes an infeasible cost M such that -M is almost always < min_x f(x),
        so that feasible points are preferred.

    Args:
        X: A `m x d` Tensor of `m` design points to use in evaluating the minimum.
        model: A fitted model.

    Returns:
        The infeasible cost M value.
    """
    posterior = model.posterior(X)
    lb = objective(posterior.mean - 6 * posterior.variance.sqrt()).min()
    M = -lb.clamp_max(0.0)
    return M.item()


@batch_mode_transform
def batch_expected_improvement(
    X: Tensor,
    model: Model,
    best_f: float,
    objective: Callable[[Tensor], Tensor] = squeeze_last_dim,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    M: float = 0.0,
    mc_samples: int = 500,
    base_samples: Optional[Tensor] = None,
) -> Tensor:
    """q-Expected Improvement acquisition function.

    Args:
        X: A `(b) x q x d` Tensor with `b` t-batches of `q` design points each.
            X is two-dimensional, assume `b = 1`.
        model: A fitted model.
        best_f: The best (feasible) function value observed so far (assumed
            noiseless).
        objective: A callable mapping a Tensor of size `b x q x t` to a
            Tensor of size `b x q`, where `t` is the number of outputs of
            the model. Note: the callable must support broadcasting.
            If omitted, squeeze the output dimension (applicable to single-
            output models only).
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t` to a Tensor of size `b x q`, where negative values
            imply feasibility. Note: the callable must support broadcasting.
            Only relevant for multi-output models (`t` > 1).
        M: The infeasibility cost. Should be set s.t. `-M < min_x obj(x)`.
        mc_samples: The number of (quasi-) Monte-Carlo samples to use for
            approximating the expectation.
        base_samples: A fixed Tensor of N(0,1) random variables used for
            deterministic optimization.

    Returns:
        Tensor: The q-EI value of the design X for each of the `b` t-batches.
    """
    posterior = model.posterior(X)
    samples = posterior.rsample(
        sample_shape=torch.Size([mc_samples]), base_samples=base_samples
    )
    obj = objective(samples)
    # apply the constraints to the objective first; then, compare with best_f
    # (which could be -M if no feasible point has been found)
    apply_constraints_(obj=obj, constraints=constraints, samples=samples, M=M)
    obj = (obj - best_f).clamp_min_(0)
    q_ei = obj.max(dim=2)[0].mean(dim=0)
    return q_ei


@batch_mode_transform
def batch_noisy_expected_improvement(
    X: Tensor,
    model: Model,
    X_observed: Tensor,
    objective: Callable[[Tensor], Tensor] = squeeze_last_dim,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    M: float = 0.0,
    mc_samples: int = 500,
    base_samples: Optional[Tensor] = None,
) -> Tensor:
    """q-Noisy Expected Improvement acquisition function.

    Args:
        X: A `b x q x d` Tensor with `b` t-batches of `q` design points each.
            If X is two-dimensional, assume `b = 1`.
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
            imply feasibility. This callable must support broadcasting. Only
            relevant for multi-output models (`t` > 1).
        M: The infeasibility cost. Should be set s.t. `-M < min_x obj(x)`.
        mc_samples: The number of (quasi-) Monte-Carlo samples to use for
            approximating the expectation.
        base_samples: A fixed Tensor of N(0,1) random variables used for
            deterministic optimization.

    Returns:
        Tensor: The q-NoisyEI value of the design X for each of the `b` t-batches.
    """
    q = X.shape[-2]
    # predict posterior (joint across points and tasks)
    posterior = model.posterior(torch.cat([X, X_observed], dim=-2))
    samples = posterior.rsample(
        sample_shape=torch.Size([mc_samples]), base_samples=base_samples
    )
    # compute objective value
    obj = objective(samples)
    apply_constraints_(obj=obj, constraints=constraints, samples=samples, M=M)
    diff = (
        (obj[:, :, :q].max(dim=2)[0] - obj[:, :, q:].max(dim=2)[0])
        .clamp_min_(0)
        .mean(dim=0)
    )
    return diff


def batch_knowledge_gradient(
    X: Tensor,
    model: Model,
    X_observed: Tensor,
    objective: Callable[[Tensor], Tensor] = squeeze_last_dim,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    mc_samples: int = 40,
    inner_mc_samples: int = 1000,
    eta: float = 1e-3,
    inner_old_base_samples: Optional[Tensor] = None,
    inner_new_base_samples: Optional[Tensor] = None,
    fantasy_base_samples: Optional[Tensor] = None,
    project: Callable[[Tensor], Tensor] = lambda X: X,
    cost: Optional[Callable[[Tensor], Tensor]] = None,
    use_X_for_old_posterior: Optional[bool] = False,
    use_posterior_mean: Optional[bool] = False,
) -> Tensor:
    """TODO: Revise!

    Constrained, multi-fidelity knowledge gradient supporting t-batch mode.

    Multifidelity optimization can be performed by using the
    optional project and cost callables.

    ** NOTE: THIS FUNCTION DOES NOT YET SUPPORT t-BATCHES.**
    ** This will require support for arbitrary batch shapes in gpytorch **

    ** TODO: Check whether soft-maxes help the gradients **

    Args:
        X: A `b x q x d` Tensor with `b` t-batches of `q` design points each.
            If X is two-dimensional, assume `b = 1`.
        model: A fitted GPyTorch model
        X_observed: A q' x d Tensor of q' design points that have already been
            observed and would be considered as the best design point. A judicious
            filtering of the points here can massively speed up the function
            evaluation without altering the function if points that are highly
            unlikely to be the best (regardless of what is observed at X) are
            removed. For example, points that clearly do not satisfy the
            constraints or have terrible objective observations can be safely
            excluded from X_observed.
        objective: A callable mapping a Tensor of size `b x q x (t)` to a Tensor
            of size `b x q`, where `t` is the number of outputs (tasks) of the model.
            Note: the callable must support broadcasting.
            If omitted, use the identity map (applicable to single-task models only).
            Assumed to be non-negative when the constraints are used!
        constraints: A list of callables, each mapping a Tensor of size `b x q x t`
            to a Tensor of size `b x q`, where negative values imply feasibility.
            Note: the callable must support broadcastingself.
            Only relevant for multi-task models (`t` > 1).
        mc_samples: The number of Monte-Carlo samples to draw from the model
            posterior. GP memory usage is multiplied by this value.
            Only used if fantasy_base_samples is not provided.
        inner_mc_samples: The number of Monte-Carlo samples to draw for the inner
            expectation. Only used if inner_base_samples is not provided.
        eta: The temperature parameter of the softmax function used in approximating
            the constraints. As `eta -> 0`, the exact (discontinuous) constraint
            is recovered.
        inner_old_base_samples: A Tensor of N(0,1) random variables used for
            deterministic optimization.
        inner_new_base_samples: A Tensor of N(0,1) random variables used for
            deterministic optimization.
        fantasy_base_samples: A Tensor of N(0,1) random variables used for
            deterministic optimization.
        project: A callable mapping a Tensor of size `b x (q + q') x d` to a
            Tensor of the same size. Use for multi-fidelity optimization where
            the returned Tensor should be projected to the highest fidelity.
        cost: A callable mapping a Tensor of size `b x q x d` to a Tensor of
            size `b x 1`. The resulting Tensor's value is the cost of submitting
            each t-batch.
        use_X_for_old_posterior: If True, concatenate `X` and `X_observed` for
            best point evaluation prior to the new observations. Defaults to
            False such that X is not included.
        use_posterior_mean: If True, instead of sampling, the mean of the posterior is
            sent into the objective and constraints.  Should be used for linear
            objectives without constraints.

    Returns:
        Tensor: The constrained q-KG value of the design X for each of the `b`
            t-batches.

    """
    # TODO: if projection is expensive, X_observed can be projected
    # just once outside this function
    X_all = project(torch.cat([X, X_observed]))

    old_posterior = model.posterior(
        X_all if use_X_for_old_posterior else project(X_observed)
    )
    if use_posterior_mean:
        old_samples = old_posterior.mean.unsqueeze(0)
    else:
        old_samples = old_posterior.rsample(
            sample_shape=torch.Size([inner_mc_samples]),
            base_samples=inner_old_base_samples,
        )
    # Shape of samples is inner_mc_samples x q' x t
    old_obj = objective(old_samples)

    # TODO: Change this to apply_constraints_ in the future (T40798532).
    apply_constraints_nonnegative_soft_(
        obj=old_obj, constraints=constraints, samples=old_samples
    )
    # Shape of obj is inner_mc_samples x q'. First compute mean across inner
    # samples, then maximize across X_observed. Uses soft-max for maximization
    # across X_observed instead of old_value = old_obj.mean(dim=0).max()
    old_per_point = old_obj.mean(dim=0)
    w = torch.softmax(old_per_point / eta, dim=-1)
    old_value = (old_per_point * w).sum()

    X_posterior = model.posterior(X=X, observation_noise=True)
    fantasy_y = X_posterior.rsample(
        sample_shape=torch.Size([mc_samples]), base_samples=fantasy_base_samples
    )
    fantasy_model = model.get_fantasy_model(inputs=X, targets=fantasy_y)
    # we need to make sure to tell gpytorch not to detach the test caches
    new_posterior = fantasy_model.posterior(X=X_all, detach_test_caches=False)
    # TODO: Tell the posterior to use the same set of Z's for each of the
    # "batches" in the fantasy model. This is doing
    # mc_samples x inner_mc_samples x (q + q') x t draws from the normal
    # distribution because a different Z is used for each of the fantasies.
    # We can probably safely reuse the same inner_samples x (q + q') x t Z tensor
    # for each of the fantasies. Possible since rsample accepts a base_sample argument.
    if use_posterior_mean:
        new_samples = new_posterior.mean.unsqueeze(0)
    else:
        new_samples = new_posterior.rsample(
            sample_shape=torch.Size([inner_mc_samples]),
            base_samples=inner_new_base_samples,
        )

    # Shape of new_samples is inner_mc_samples x mc_samples x (q + q') x t
    new_obj = objective(new_samples)

    # TODO: Change this to apply_constraints_ in the future (T40798532).
    apply_constraints_nonnegative_soft_(
        obj=new_obj, constraints=constraints, samples=new_samples
    )
    # Shape of obj is inner_mc_samples x mc_samples x (q + q'). First compute mean
    # across inner samples, then maximize across X_all, then compute mean across
    # outer samples. Uses soft-max for maximization across X_all instead of
    # new_value = new_obj.mean(dim=0).max(dim=-1)[0].mean()
    new_per_point = new_obj.mean(dim=0)
    new_value = (
        (new_per_point * torch.softmax(new_per_point / eta, dim=-1)).sum(dim=-1).mean()
    )

    result = new_value - old_value
    if cost is not None:
        result = result / cost(X)
    return result


def batch_knowledge_gradient_no_discretization(
    X: Tensor,
    X_fantasies: Tensor,
    X_old: Tensor,
    model: Model,
    objective: Callable[[Tensor], Tensor] = squeeze_last_dim,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    inner_old_base_samples: Optional[Tensor] = None,
    inner_new_base_samples: Optional[Tensor] = None,
    fantasy_base_samples: Optional[Tensor] = None,
    inner_mc_samples: int = 100,
    eta: float = 1e-3,
    project: Callable[[Tensor], Tensor] = lambda X: X,
    cost: Optional[Callable[[Tensor], Tensor]] = None,
    use_posterior_mean: Optional[bool] = False,
) -> Tensor:
    """TODO: Revise!

    Constrained, multi-fidelity knowledge gradient supporting t-batch mode.

    Multifidelity optimization can be performed by using the
    optional project and cost callables.

    Unlike batch_knowledge_gradient, this function optimizes
    the knowledge gradient without discretization.

    ** NOTE: THIS FUNCTION DOES NOT YET SUPPORT t-BATCHES. **

    ** TODO: Check whether soft-maxes help the gradients **

    Args:
        X: A `b x q x d` Tensor with `b` t-batches of `q` design points each.
            If X is two-dimensional, assume `b = 1`.
        X_fantasies:  A `q' x d` Tensor of q' design points where
            there is one design point for each fantasy.
        X_old:  A `1 x d` Tensor of a single design point
        model: A fitted GPyTorch model from which the fantasy_model was created.
        objective: A callable mapping a Tensor of size `b x q x (t)`
            to a Tensor of size `b x q`, where `t` is the number of
            outputs (tasks) of the model. Note: the callable must support broadcasting.
            If omitted, use the identity map (applicable to single-task models only).
            Assumed to be non-negative when the constraints are used!
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t` to a Tensor of size `b x q`, where negative values imply
            feasibility. Note: the callable must support broadcasting. Only
            relevant for multi-task models (`t` > 1).
        inner_mc_samples:  The number of Monte-Carlo samples to draw for the
            inner expectation.  Only used if inner_base_samples is not provided.
        inner_old_base_samples: A Tensor of N(0,1) random variables used for
            deterministic optimization.
        inner_new_base_samples: A Tensor of N(0,1) random variables used for
            deterministic optimization.
        fantasy_base_samples: A Tensor of N(0,1) random variables used for
            deterministic optimization.
        eta: The temperature parameter of the softmax function used in approximating
            the constraints. As `eta -> 0`, the exact (discontinuous) constraint
            is recovered.
        project:  A callable mapping a Tensor of size `q x d` to a Tensor of the
            same size. Use for multi-fidelity optimization where the returned Tensor
            should be projected to the highest fidelity.
        cost: A callable mapping a Tensor of size `b x q x d` to a Tensor of
            size `b x 1`.  The resulting Tensor's value is the cost of submitting
            each t-batch.
        use_posterior_mean: If True, instead of sampling, the mean of the posterior is
            sent into the Objective and Constraints.  Should be used for linear
            objectives without constraints.

    Returns:
        Tensor: The q-KG value of the design X averaged across the fantasy models
            where X_fantasies_i is chosen as the final selection for the i-th
            batch within fantasy_model and X_old is chosen as the final selection
            for the previous model. The maximum across X_fantasies and X_old
            evaluated at design X is the true q-KG of X.
    """
    old_posterior = model.posterior(project(X_old))
    if use_posterior_mean:
        old_samples = old_posterior.mean.unsqueeze(0)
    else:
        old_samples = old_posterior.rsample(
            sample_shape=torch.Size([inner_mc_samples]),
            base_samples=inner_old_base_samples,
        )
    # Shape of samples is inner_mc_samples x 1 x t
    old_obj = objective(old_samples)

    # TODO: Change this to apply_constraints_ in the future (T40798532).
    apply_constraints_nonnegative_soft_(
        obj=old_obj, constraints=constraints, samples=old_samples
    )
    # Shape of obj is inner_mc_samples x 1
    old_value = old_obj.mean()

    X_posterior = model.posterior(X=X, observation_noise=True)
    fantasy_y = X_posterior.rsample(
        sample_shape=torch.Size([X_fantasies.shape[0]]),
        base_samples=fantasy_base_samples,
    )
    fantasy_model = model.get_fantasy_model(inputs=X, targets=fantasy_y)
    # X_fantasies is q' x d, needs to be q' x 1 x d
    # for batch mode evaluation with q' fantasies
    # we need to make sure to tell gpytorch not to detach the test caches
    new_posterior = fantasy_model.posterior(
        X=project(X_fantasies).unsqueeze(1), detach_test_caches=False
    )
    # TODO: Tell the posterior to use the same set
    # of Z's for each of the "batches" in the fantasy model. This
    # is doing q' x inner_mc_samples x 1 x t
    # draws from the normal distribution because a different Z is used for
    # each of the fantasies.  We can probably safely reuse the
    # same inner_samples x 1 x t Z tensor for each of the
    # q' fantasies.  Possible since rsample accepts a base_sample argument.
    if use_posterior_mean:
        new_samples = new_posterior.mean.unsqueeze(0)
    else:
        new_samples = new_posterior.rsample(
            sample_shape=torch.Size([inner_mc_samples]),
            base_samples=inner_new_base_samples,
        )
    # Shape of new_samples is
    # inner_mc_samples x q' x 1 x t
    new_obj = objective(new_samples)

    # TODO: Change this to apply_constraints_ in the future (T40798532).
    apply_constraints_nonnegative_soft_(
        obj=new_obj, constraints=constraints, samples=new_samples
    )

    # Shape of new_obj is inner_mc_samples x q' x 1
    # First compute mean across inner samples, then
    # mean across fantasies
    new_value = new_obj.squeeze(-1).mean(dim=0).mean()

    result = new_value - old_value
    if cost is not None:
        result = result / cost(X)
    return result


@batch_mode_transform
def batch_probability_of_improvement(
    X: Tensor,
    model: Model,
    best_f: Tensor,
    mc_samples: int = 500,
    base_samples: Optional[Tensor] = None,
) -> Tensor:
    """q-Probability of Improvement acquisition function.

    Args:
        X: A `(b) x q x d` Tensor with `b` t-batches of `q` design points each.
            If `X` is two-dimensional, assume `b = 1`.
        model: A fitted model.
        best_f: The best (feasible) function value observed so far (assumed
            noiseless).
        mc_samples: The number of (quasi-) Monte-Carlo samples to use for
            approximating the probability of improvement.
        base_samples: A Tensor of N(0,1) random variables used for
            deterministic optimization.

    Returns:
        Tensor: The q-PI value of the design X for each of the `b` t-batches.
    """
    posterior = model.posterior(X)
    samples = posterior.rsample(
        sample_shape=torch.Size([mc_samples]), base_samples=base_samples
    ).max(dim=2)[0]
    val = torch.sigmoid(samples - best_f).mean(dim=0)
    return val


@batch_mode_transform
def batch_simple_regret(
    X: Tensor,
    model: Model,
    mc_samples: int = 500,
    base_samples: Optional[Tensor] = None,
) -> Tensor:
    """q-Simple Regret acquisition function.

    Args:
        X: A `(b) x q x d` Tensor with `b` t-batches of `q` design points each.
            If `X` is two-dimensional, assume `b = 1`.
        model: A fitted model.
        mc_samples: The number of (quasi-) Monte-Carlo samples to use for
            approximating the probability of improvement.
        base_samples: A Tensor of N(0,1) random variables used for
            deterministic optimization.

    Returns:
        Tensor: The q-simple regret value of the design X for each of the `b`
            t-batches.
    """
    posterior = model.posterior(X)
    val = (
        posterior.rsample(
            sample_shape=torch.Size([mc_samples]), base_samples=base_samples
        )
        .max(dim=2)[0]
        .mean(dim=0)
    )
    return val


@batch_mode_transform
def batch_upper_confidence_bound(
    X: Tensor,
    model: Model,
    beta: float,
    mc_samples: int = 500,
    base_samples: Optional[Tensor] = None,
) -> Tensor:
    """q-Upper Confidence Bound acquisition function.

    Args:
        X: A `(b) x q x d` Tensor with `b` t-batches of `q` design points each.
            If `X` is two-dimensional, assume `b = 1`.
        model: A fitted model.
        beta: Controls tradeoff between mean and standard deviation in UCB.
        mc_samples: The number of (quasi-) Monte-Carlo samples to use for
            approximating the probability of improvement.
        base_samples: A Tensor of `N(0,1)` random variables used for
            deterministic optimization.

    Returns:
        Tensor: The constrained q-UCB value of the design X for each of the
            `b` t-batches.
    """
    posterior = model.posterior(X)
    samples = posterior.rsample(
        sample_shape=torch.Size([mc_samples]), base_samples=base_samples
    )
    mean = posterior.mean
    ucb_mc_samples = (sqrt(beta * pi / 2) * (samples - mean).abs() + mean).max(dim=2)[0]
    return ucb_mc_samples.mean(dim=0)
