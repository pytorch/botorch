#!/usr/bin/env python3

from math import pi, sqrt
from typing import Callable, List, Optional

import torch
from botorch.models.utils import initialize_BFGP
from botorch.optim.constraints import soft_eval_constraint
from botorch.utils import manual_seed
from gpytorch import Module, fast_pred_var
from torch import Tensor
from torch.nn.functional import sigmoid


"""
Batch acquisition functions using the reparameterization trick in combination
with MC sampling as outlined in:

    Wilson, J. T., Moriconi, R., Hutter, F., & Deisenroth, M. P. (2017).
    The reparameterization trick for acquisition functions.
    arXiv preprint arXiv:1712.00424.

"""


def batch_expected_improvement(
    X: Tensor,
    model: Module,
    best_f: float,
    objective: Callable[[Tensor], Tensor] = lambda Y: Y,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    mc_samples: int = 5000,
    eta: float = 1e-3,
    seed: Optional[int] = None,
) -> Tensor:
    """q-EI with constraints, supporting t-batch mode.

    *** NOTE: THIS FUNCTION DOES NOT YET SUPPORT t-BATCHES. #TODO: T32892289 ***

    Args:
        X: A `b x q x n` Tensor with `b` t-batches of `q` design points
            each. If X is two-dimensional, assume `b = 1`.
        model: A fitted model implementing an `rsample` method to sample from the
            posterior function values
        best_f: The best (feasible) function value observed so far (assumed
            noiseless).
        objective: A callable mapping a Tensor of size `b x q x t x mc_samples`
            to a Tensor of size `b x q x mc_samples`, where `t` is the number of
            outputs (tasks) of the model. If omitted, use the identity map
            (applicable to single-task models only).
            Assumed to be non-negative when the constaints are used!
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t x mc_samples` to a Tensor of size `b x q x mc_samples`,
            where negative values imply feasibility. Only relevant for multi-task
            models (`t` > 1).
        mc_samples: The number of Monte-Carlo samples to draw from the model
            posterior.
        eta: The temperature parameter of the softmax function used in approximating
            the constraints. As `eta -> 0`, the exact (discontinuous) constraint
            is recovered.
        seed: The random seed to use for sampling.

    Returns:
        Tensor: The constrained q-EI value of the design X for each of the `b`
            t-batches.

    """
    model.eval()
    # predict posterior (joint across points and tasks)
    posterior = model(X)
    # Draw MC samples (shape is n_eval x n_tasks x n_samples)
    with manual_seed(seed=seed), fast_pred_var():
        samples = posterior.rsample(sample_shape=torch.Size([mc_samples]))
    # compute objective value
    obj = (objective(samples) - best_f).clamp_min_(0)
    if constraints is not None:
        for constraint in constraints:
            obj.mul_(soft_eval_constraint(constraint(samples), eta=eta))
    return obj.max(dim=1)[0].mean()


def batch_noisy_expected_improvement(
    X: Tensor,
    model: Module,
    X_observed: Tensor,
    objective: Callable[[Tensor], Tensor] = lambda Y: Y,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    mc_samples: int = 5000,
    eta: float = 1e-3,
    seed: Optional[int] = None,
) -> Tensor:
    """q-NoisyEI with constraints, supporting t-batch mode.

    *** NOTE: THIS FUNCTION DOES NOT YET SUPPORT t-BATCHES. TODO: T32892289 ***

    Args:
        X: A `b x q x n` Tensor with `b` t-batches of `q` design points each.
            If X is two-dimensional, assume `b = 1`.
        model: A fitted model implementing a `sample` method to sample from the
            posterior function values
        X_observed: A q' x n Tensor of q' design points that have already been
            observed and would be considered as the best design point.
        objective: A callable mapping a Tensor of size `b x q x t x mc_samples`
            to a Tensor of size `b x q x mc_samples`, where `t` is the number of
            outputs (tasks) of the model. If omitted, use the identity map
            (applicable to single-task models only).
            Assumed to be non-negative when the constaints are used!
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t x mc_samples` to a Tensor of size `b x q x mc_samples`,
            where negative values imply feasibility. Only relevant for multi-task
            models (`t` > 1).
        mc_samples: The number of Monte-Carlo samples to draw from the model
            posterior.
        eta: The temperature parameter of the softmax function used in approximating
            the constraints. As `eta -> 0`, the exact (discontinuous) constraint
            is recovered.
        seed: The random seed to use for sampling.

    Returns:
        Tensor: The constrained q-EI value of the design X for each of the `b`
            t-batches.

    """
    model.eval()
    # predict posterior (joint across points and tasks)
    posterior = model(torch.cat([X, X_observed]))
    # Draw MC samples (shape is n_eval x n_tasks x n_samples) for multi-task
    # models and n_eval x n_samples otherwise)
    with manual_seed(seed=seed), fast_pred_var():
        samples = posterior.rsample(sample_shape=torch.Size([mc_samples]))
    # compute objective value
    obj = objective(samples)
    if constraints is not None:
        obj.clamp_min_(0)  # Enforce non-negativity with constraints
        for constraint in constraints:
            obj.mul_(soft_eval_constraint(constraint(samples), eta=eta))
    diff = obj[:, : X.shape[0]].max(dim=1)[0] - obj[:, X.shape[0] :].max(dim=1)[0]
    return diff.clamp_min_(0).mean()


def batch_knowledge_gradient(
    X: Tensor,
    model: Module,
    X_observed: Tensor,
    objective: Callable[[Tensor], Tensor] = lambda Y: Y,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    mc_samples: int = 40,
    inner_mc_samples: int = 1000,
    eta: float = 1e-3,
    seed: Optional[int] = None,
    inner_seed: Optional[int] = None,
    project: Callable[[Tensor], Tensor] = lambda X: X,
    cost: Optional[Callable[[Tensor], Tensor]] = None,
    use_X_for_old_posterior: Optional[bool] = False,
) -> Tensor:
    """Constrained, multi-fidelity knowledge gradient supporting t-batch mode.

    Multifidelity optimization can be performed by using the
    optional project and cost callables.

    *** NOTE: THIS FUNCTION DOES NOT YET SUPPORT t-BATCHES. TODO: T32892289 ***

    *** TODO: Check whether soft-maxes help the gradients **

    *** TODO: If there are no constraints and the objective is linear then the
    inner sampling is unnecessary and just the posterior
    mean can be used. ***

    Args:
        X: A `b x q x n` Tensor with `b` t-batches of `q` design points each.
            If X is two-dimensional, assume `b = 1`.
        model: A fitted GPyTorch model
        X_observed: A q' x n Tensor of q' design points that have already been
            observed and would be considered as the best design point.  A
            judicious filtering of the points here can massively
            speed up the function evaluation without altering the
            function if points that are highly unlikely to be the
            best (regardless of what is observed at X) are removed.
            For example, points that clearly do not satisfy the constraints
            or have terrible objective observations can be safely
            excluded from X_observed.
        objective: A callable mapping a Tensor of size `b x q x t x mc_samples`
            to a Tensor of size `b x q x mc_samples`, where `t` is the number of
            outputs (tasks) of the model. If omitted, use the identity map
            (applicable to single-task models only).
            Assumed to be non-negative when the constraints are used!
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t x mc_samples` to a Tensor of size `b x q x mc_samples`,
            where negative values imply feasibility. Only relevant for multi-task
            models (`t` > 1).
        mc_samples: The number of Monte-Carlo samples to draw from the model
            posterior for the outer sample.  GP memory usage is multiplied by
            this value.
        inner_mc_samples:  The number of Monte-Carlo samples to draw for the
            inner expectation
        eta: The temperature parameter of the softmax function used in approximating
            the constraints. As `eta -> 0`, the exact (discontinuous) constraint
            is recovered.
        seed: The random seed to use for the outer sampling
        inner_seed:  The random seed to use for the inner sampling
        project:  A callable mapping a Tensor of size `b x (q + q') x n` to a
            Tensor of the same size.  Use for multi-fidelity optimization where
            the returned Tensor should be projected to the highest fidelity.
        cost: A callable mapping a Tensor of size `b x q x n` to a Tensor of
            size `b x 1`.  The resulting Tensor's value is the cost of submitting
            each t-batch.
        use_X_for_old_posterior:  If True, concatenate X and
            X_observed for best point evaluation prior to the new observations.
            Defaults to False such that X is not included.

    Returns:
        Tensor: The constrained q-KG value of the design X for each of the `b`
            t-batches.

    """
    # TODO: if projection is expensive, X_observed can be projected
    # just once outside this function
    X_all = project(torch.cat([X, X_observed]))

    model.eval()
    old_posterior = model(X_all if use_X_for_old_posterior else X_observed)
    with manual_seed(seed=inner_seed), fast_pred_var():
        old_samples = old_posterior.rsample(torch.Size([inner_mc_samples]))
    # Shape of samples is inner_mc_samples x q' x t
    old_obj = objective(old_samples)
    if constraints is not None:
        old_obj.clamp_min_(0)  # Enforce non-negativity with constraints
        for constraint in constraints:
            old_obj.mul_(soft_eval_constraint(constraint(old_samples), eta=eta))
    # Shape of obj is inner_mc_samples x q'
    # First compute mean across inner samples, then maximize across X_observed
    # Uses soft-max for maximization across X_observed instead of
    # old_value = old_obj.mean(dim=0).max()
    old_per_point = old_obj.mean(dim=0)
    w = torch.softmax(old_per_point / eta, dim=-1)
    old_value = (old_per_point * w).sum()

    fantasy_model = initialize_BFGP(model=model, X=X, num_samples=mc_samples, seed=seed)
    new_posterior = fantasy_model(X_all.unsqueeze(0).repeat(mc_samples, 1, 1))
    # TODO: Tell the posterior to use the same set
    # of Z's for each of the "batches" in the fantasy model. This
    # is doing mc_samples x inner_mc_samples x (q + q') x t
    # draws from the normal distribution because a different Z is used for
    # each of the fantasies.  We can probably safely reuse the
    # same inner_samples x (q + q') x t Z tensor for each of the
    # fantasies.  Possible since rsample accepts a base_sample argument.
    with manual_seed(seed=inner_seed), fast_pred_var():
        new_samples = new_posterior.rsample(torch.Size([inner_mc_samples]))
    # Shape of new_samples is
    # inner_mc_samples x mc_samples x (q + q') x t
    new_obj = objective(new_samples)
    if constraints is not None:
        new_obj.clamp_min_(0)  # Enforce non-negativity with constraints
        for constraint in constraints:
            new_obj.mul_(soft_eval_constraint(constraint(new_samples), eta=eta))

    # Shape of obj is inner_mc_samples x mc_samples x (q + q')
    # First compute mean across inner samples, then maximize across
    # X_all, then compute mean across outer samples
    # Uses soft-max for maximization across X_all instead of
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
    model: Module,
    objective: Callable[[Tensor], Tensor] = lambda Y: Y,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    seed: Optional[int] = None,
    inner_mc_samples: int = 1000,
    eta: float = 1e-3,
    inner_seed: Optional[int] = None,
    project: Callable[[Tensor], Tensor] = lambda X: X,
    cost: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """Constrained, multi-fidelity knowledge gradient supporting t-batch mode.

    Multifidelity optimization can be performed by using the
    optional project and cost callables.

    Unlike batch_knowledge_gradient, this function optimizes
    the knowledge gradient without discretization.

    *** NOTE: THIS FUNCTION DOES NOT YET SUPPORT t-BATCHES. IT MAY NOT BE
    POSSIBLE IN THIS CASE. TODO: T32892289 ***

    *** TODO: Check whether soft-maxes help the gradients **

    *** TODO: If there are no constraints and the objective is linear then the
    sampling is unnecessary and just the posterior
    mean can be used. ***

    Args:
        X: A `b x q x n` Tensor with `b` t-batches of `q` design points each.
            If X is two-dimensional, assume `b = 1`.
        X_fantasies:  A `q' x n` Tensor of q' design points where
            there is one design point for each fantasy.
        X_old:  A `1 x n` Tensor of a single design point
        model: A fitted GPyTorch model from which the fantasy_model
            was created.
        objective: A callable mapping a Tensor of size `b x q x t x mc_samples`
            to a Tensor of size `b x q x mc_samples`, where `t` is the number of
            outputs (tasks) of the model. If omitted, use the identity map
            (applicable to single-task models only).
            Assumed to be non-negative when the constraints are used!
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t x mc_samples` to a Tensor of size `b x q x mc_samples`,
            where negative values imply feasibility. Only relevant for multi-task
            models (`t` > 1).
        seed: The random seed to use for the fantasies
        inner_mc_samples:  The number of Monte-Carlo samples to draw for the
            inner expectation
        eta: The temperature parameter of the softmax function used in approximating
            the constraints. As `eta -> 0`, the exact (discontinuous) constraint
            is recovered.
        inner_seed:  The random seed to use for the inner sampling
        project:  A callable mapping a Tensor of size `q x n` to a
            Tensor of the same size.  Use for multi-fidelity optimization where
            the returned Tensor should be projected to the highest fidelity.
        cost: A callable mapping a Tensor of size `b x q x n` to a Tensor of
            size `b x 1`.  The resulting Tensor's value is the cost of submitting
            each t-batch.

    Returns:
        Tensor: The q-KG value of the design X averaged
            across the fantasy models where X_fantasies_i
            is chosen as the final selection for the i-th
            batch within fantasy_model and X_old is
            chosen as the final selection for the previous
            model.  The maximum across X_fantasies and X_old
            evaluated at design X is the true q-KG of X.

    """
    model.eval()
    old_posterior = model(project(X_old))
    with manual_seed(seed=inner_seed), fast_pred_var():
        old_samples = old_posterior.rsample(torch.Size([inner_mc_samples]))
    # Shape of samples is inner_mc_samples x 1 x t
    old_obj = objective(old_samples)
    if constraints is not None:
        old_obj.clamp_min_(0)  # Enforce non-negativity with constraints
        for constraint in constraints:
            old_obj.mul_(soft_eval_constraint(constraint(old_samples), eta=eta))
    # Shape of obj is inner_mc_samples x 1
    old_value = old_obj.mean()

    fantasy_model = initialize_BFGP(
        model=model, X=X, num_samples=X_fantasies.shape[0], seed=seed
    )
    # X_fantasies is q' x n, needs to be q' x 1 x n
    # for batch mode evaluation with q' fantasies
    new_posterior = fantasy_model(project(X_fantasies).unsqueeze(1))
    # TODO: Tell the posterior to use the same set
    # of Z's for each of the "batches" in the fantasy model. This
    # is doing q' x inner_mc_samples x 1 x t
    # draws from the normal distribution because a different Z is used for
    # each of the fantasies.  We can probably safely reuse the
    # same inner_samples x 1 x t Z tensor for each of the
    # q' fantasies.  Possible since rsample accepts a base_sample argument.
    with manual_seed(seed=inner_seed), fast_pred_var():
        new_samples = new_posterior.rsample(torch.Size([inner_mc_samples]))
    # Shape of new_samples is
    # inner_mc_samples x q' x 1 x t
    new_obj = objective(new_samples)
    if constraints is not None:
        new_obj.clamp_min_(0)  # Enforce non-negativity with constraints
        for constraint in constraints:
            new_obj.mul_(soft_eval_constraint(constraint(new_samples), eta=eta))

    # Shape of new_obj is inner_mc_samples x q' x 1
    # First compute mean across inner samples, then
    # mean across fantasies
    new_value = new_obj.squeeze(-1).mean(dim=0).mean()

    result = new_value - old_value
    if cost is not None:
        result = result / cost(X)
    return result


def batch_probability_of_improvement(
    X: Tensor,
    model: Module,
    best_f: Tensor,
    mc_samples: int = 5000,
    seed: Optional[int] = None,
) -> Tensor:
    """TODO"""
    model.eval()
    with manual_seed(seed=seed), fast_pred_var():
        samples = model(X).rsample(sample_shape=torch.Size([mc_samples])).max(1)[0]
        val = sigmoid(samples - best_f).mean()
    return val


def batch_simple_regret(
    X: Tensor, model: Module, mc_samples: int = 5000, seed: Optional[int] = None
) -> Tensor:
    """TODO"""
    model.eval()
    with manual_seed(seed=seed), fast_pred_var():
        val = model(X).rsample(sample_shape=torch.Size([mc_samples])).max(1)[0].mean()
    return val


def batch_upper_confidence_bound(
    X: Tensor,
    model: Module,
    beta: float,
    mc_samples: int = 5000,
    seed: Optional[int] = None,
) -> Tensor:
    """TODO"""
    model.eval()
    with manual_seed(seed=seed), fast_pred_var():
        mvn = model(X)
        val = (
            (
                sqrt(beta * pi / 2)
                * mvn.covar()
                .zero_mean_mvn_samples(sample_shape=torch.Size([mc_samples]))
                .abs()
                + mvn._mean.view(1, -1)
            )
            .max(1)[0]
            .mean()
        )
    return val
