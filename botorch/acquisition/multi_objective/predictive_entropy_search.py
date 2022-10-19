#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition function for predictive entropy search for multi-objective
Bayesian optimization (PES). The code supports multiple objectives and batching.

References:

.. [Garrido-Merchan2019]
    E. Garrido-Merchan and D. Hernandez-Lobato. Predictive Entropy Search for
    Multi-objective Bayesian Optimization with Constraints. Neurocomputing. 2019.
    The computation follows the procedure described in the supplementary material:
    https://www.sciencedirect.com/science/article/abs/pii/S0925231219308525

"""

from __future__ import annotations
from typing import Any, Optional, Tuple
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.models.utils import check_no_nans
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor, Size
from torch.distributions import Normal

CLAMP_LB = 1.0e-8
NEG_INF = -1.0e+10


class qPredictiveEntropySearch(AcquisitionFunction):
    r"""The acquisition function for Predictive Entropy Search. The code supports
    both single and multiple objectives as well as batching.

    This acquisition function approximates the mutual information between the
    observation at a candidate point X and the Pareto optimal input using expectation
    propagation.

    IMPORTANT NOTES:
    We observed that the expectation propagation (EP) could potentially fail for a
    number of reasons that we list below. When EP does not fail, we observed
    visually that the acquisition function is similar to the `truth` obstained using
    rejection sampling.

    We denote a training (or testing) input by x_n and a sampled Pareto optimal input
    by x_p (assuming maximization).

    (i) EP will fail if the sampled Pareto set is very poor compared to the
    training data. In particular, if alpha = (mean(x_n) - mean(x_p)) / std(x_n - x_p)
    is very large, then a log(0) error will occur from considering
    log(1 - cdf(alpha)). We have pre-emptively clamped some values depending on
    alpha in order to prevent this from happening accidentally i.e. when an EP update
    is especially poor.

    (ii) EP will fail during training or testing if data is very close to the Pareto
    optimal input. This problem arises because we compute matrix inverses for the
    two-dimensional marginals (x_n, x_p). To address this we manually
    add jitter to the diagonal of the covariance matrix i.e. `ep_jitter` when
    training and `test_jitter` when testing. The default choice is not always
    appropriate because the same jitter is used for the inversion of the covariance
    and precision matrix, which are on different scales.

    (iii) Setting the convergence threshold too small could potentially lead to the
    same invertibility issues discussed in (ii) after performing a poor EP update.

    (iv) The gradients computed using automatic differentiation is questionable and
    sometimes produces `nan` values. Hence, we use the two point finite difference
    approximation given in scipy.minimize when maximizing the acquisition function.

    We attribute the errors arising in the autograd gradients to the damping
    procedure used in the original paper. In particular, a damped EP update takes the
    form `a * param_new + (1 - a) *  param_old`, where `a` denotes the damping
    coefficient. This damping coefficient is set in a way to ensure the covariance
    matrix is still positive semi-definite. We found that the procedure is very
    sensitive to the damping schedule. Our implementation uses the strategy discussed
    in the original paper, where we initially check `a = 1` and if this is not
    suitable, then we check `a = 0.5` etc.

    """
    def __init__(
        self,
        model: Model,
        pareto_sets: Tensor,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        constraint_model: Optional[Model] = None,
        max_ep_iterations: Optional[int] = 250,
        ep_jitter: Optional[float] = 1e-4,
        test_jitter: Optional[float] = 1e-4,
        threshold: Optional[float] = 1e-2,
        phi_update: Optional[bool] = True,
        verbose: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        r"""Predictive entropy search acquisition function.

        Args:
            model: A fitted batched model with `M` number of outputs.
            pareto_sets: A `num_pareto_samples x P x d`-dim tensor containing the
                Pareto optimal set of inputs, where `P` is the number of pareto
                optimal points. The points in each sample have to be discrete
                otherwise expectation propagation will fail.
            maximize: If True, consider the problem a maximization problem.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            constraint_model: A fitted batched model with `K` number of outputs
                representing inequality constraints c_k(x) >= 0 for k=1,...,K as in
                the original paper.
            max_ep_iterations: The maximum number of expectation propagation
                iterations. (The minimum number of iterations is set at 3.)
            ep_jitter: The amount of jitter added for the matrix inversion that
                occurs during the expectation propagation update during the training
                phase.
            test_jitter: The amount of jitter added for the matrix inversion that
                occurs during the expectation propagation update in the testing
                phase.
            threshold: The convergence threshold for expectation propagation. This
                assesses the relative change in the mean and covariance. We default
                to one percent change i.e. `threshold = 1e-2`.
            phi_update: If true we update the phi factor if the number of constraints
                is non-zero.
            verbose: If true we display some text regarding the progress of
                expectation propagation.
        """
        super().__init__(model=model)

        self.model = model
        self.constraint_model = constraint_model
        self.maximize = maximize
        self.set_X_pending(X_pending)

        self.pareto_sets = pareto_sets

        # add the pareto set to the existing training data
        # [there is probably a better way to do this]
        self.num_pareto_samples = pareto_sets.shape[0]

        # TODO: There is probably a better way to extract the training data
        if model.num_outputs > 1:
            train_X = self.model.train_inputs[0][0]
        else:
            train_X = self.model.train_inputs[0]

        self.augmented_X = torch.cat([
            train_X.repeat(self.num_pareto_samples, 1, 1), self.pareto_sets],
            dim=-2
        )
        self.max_ep_iterations = max_ep_iterations
        self.ep_jitter = ep_jitter
        self.test_jitter = test_jitter
        self.verbose = verbose
        self.phi_update = phi_update
        self.threshold = threshold
        self._expectation_propagation()

    def _expectation_propagation(self) -> None:
        r"""Perform expectation propagation to obtain the covariance factors that
        depend on the Pareto sets.

        The updates are performed in the natural parameter space. For a multivariate
        normal distribution with mean mu and covariance Sigma, we call Sigma^{-1}
        the natural covariance and Sigma^{-1} mu the natural mean.
        """
        ###########################################################################
        # INITIALIZATION
        ###########################################################################
        M = self.model.num_outputs
        # TODO: Need a better way to get the number of training inputs.
        if M > 1:
            N = len(self.model.train_inputs[0][0])
        else:
            N = len(self.model.train_inputs[0])
        num_pareto_samples = self.num_pareto_samples
        P = self.pareto_sets.shape[-2]

        # initialize the predictive natural mean and variances
        (pred_f_nat_mean, pred_f_nat_cov,
         pred_f_mean, pred_f_cov) = _initialize_predictive_matrices(
            self.augmented_X, self.model,
            observation_noise=False, jitter=self.ep_jitter
        )
        # initialize the marginals
        # `num_pareto_samples x M x (N + P)`
        mean_f = pred_f_mean.clone()
        nat_mean_f = pred_f_nat_mean.clone()
        # `num_pareto_samples x M x (N + P)
        cov_f = pred_f_cov.clone()
        nat_cov_f = pred_f_nat_cov.clone()

        # initialize omega the function which encodes the fact that the pareto points
        # are optimal in the constrained space i.e. any point satisfying the
        # constraint should not dominate the pareto efficient points.

        # `num_pareto_samples x M x (N + P) x P x 2`
        omega_f_nat_mean = torch.zeros((num_pareto_samples, M, N + P, P, 2))
        # `num_pareto_samples x M x (N + P) x P x 2 x 2`
        omega_f_nat_cov = torch.zeros((num_pareto_samples, M, N + P, P, 2, 2))

        if self.constraint_model is not None:
            K = self.constraint_model.num_outputs

            # initialize the predictive natural mean and variances
            (pred_c_nat_mean, pred_c_nat_cov,
             pred_c_mean, pred_c_cov) = _initialize_predictive_matrices(
                self.augmented_X, self.constraint_model,
                observation_noise=False, jitter=self.ep_jitter
            )

            # initialize the marginals
            # `num_pareto_samples x K x (N + P)`
            mean_c = pred_c_mean.clone()
            nat_mean_c = pred_c_nat_mean.clone()
            # `num_pareto_samples x K x (N + P) x (N + P)`
            cov_c = pred_c_cov.clone()
            nat_cov_c = pred_c_nat_cov.clone()

            # initialize phi the function which encodes the fact that the pareto
            # points satisfies the constraints
            # `num_pareto_samples x K x P`
            phi_nat_mean = torch.zeros((num_pareto_samples, K, P))
            # `num_pareto_samples x K x P`
            phi_nat_var = torch.zeros((num_pareto_samples, K, P))

            # `num_pareto_samples x K x (N + P) x P`
            omega_c_nat_mean = torch.zeros((num_pareto_samples, K, N + P, P))
            # `num_pareto_samples x K x (N + P) x P`
            omega_c_nat_var = torch.zeros((num_pareto_samples, K, N + P, P))
        else:
            K = 0
        ###########################################################################
        # EXPECTATION PROPAGATION
        ###########################################################################
        damping = torch.ones(num_pareto_samples, M)

        if K > 0:
            damping_c = torch.ones(num_pareto_samples, K)
        else:
            damping_c = torch.zeros(num_pareto_samples, K)

        full_damp = torch.sum(torch.ones(num_pareto_samples))

        import time as time

        iteration = 0
        start_time = time.time()
        while ((torch.sum(damping) > 0) or (torch.sum(damping_c) > 0)) and \
                (iteration < self.max_ep_iterations):
            loop_time = time.time()
            # Compute the new natural mean and covariance
            if K > 0:
                ####################################################################
                # CONSTRAINT FUNCTION: PHI UPDATE
                ####################################################################
                if self.phi_update:
                    phi_nat_mean_new, phi_nat_var_new = _update_phi(
                        mean_c, cov_c, phi_nat_mean, phi_nat_var, N
                    )
                else:
                    phi_nat_mean_new = phi_nat_mean
                    phi_nat_var_new = phi_nat_var
                ####################################################################
                # CONSTRAINT FUNCTION: OMEGA UPDATE
                ####################################################################
                (omega_f_nat_mean_new, omega_f_nat_cov_new,
                 omega_c_nat_mean_new, omega_c_nat_var_new) = _update_omega(
                    mean_f, cov_f, omega_f_nat_mean, omega_f_nat_cov,
                    N, P, M, self.maximize, self.ep_jitter, K,
                    mean_c, cov_c, omega_c_nat_mean, omega_c_nat_var
                )
                ####################################################################
                # CONSTRAINT FUNCTION: MARGINAL UPDATE
                ####################################################################
                (nat_mean_f_new, nat_cov_f_new,
                 nat_mean_c_new, nat_cov_c_new) = _update_marginals(
                    pred_f_nat_mean, pred_f_nat_cov,
                    omega_f_nat_mean_new, omega_f_nat_cov_new,
                    N, P, K,
                    pred_c_nat_mean, pred_c_nat_cov,
                    phi_nat_mean_new, phi_nat_var_new,
                    omega_c_nat_mean_new, omega_c_nat_var_new
                )
                ####################################################################
                # CONSTRAINT FUNCTION: DAMPING UPDATE
                ####################################################################
                # update damping of constraints
                damping_c, cholesky_nat_cov_c = _update_damping(
                    nat_cov_c, nat_cov_c_new, damping_c, self.ep_jitter
                )
                ####################################################################
                # CONSTRAINT FUNCTION: DAMPED UPDATE
                ####################################################################
                # Damp update of phi
                phi_nat_mean = _damped_update(
                    phi_nat_mean, phi_nat_mean_new, damping_c
                )
                phi_nat_var = _damped_update(
                    phi_nat_var, phi_nat_var_new, damping_c
                )
                # Damp update of omega
                omega_c_nat_mean = _damped_update(
                    omega_c_nat_mean, omega_c_nat_mean_new, damping_c
                )
                omega_c_nat_var = _damped_update(
                    omega_c_nat_var, omega_c_nat_var_new, damping_c
                )
                # update the mean and covariance
                nat_mean_c = _damped_update(nat_mean_c, nat_mean_c_new, damping_c)
                nat_cov_c = _damped_update(nat_cov_c, nat_cov_c_new, damping_c)
                # compute cholesky inverse
                cov_c_new = torch.cholesky_inverse(cholesky_nat_cov_c)
                # cov_c_new = torch.inverse(nat_cov_c)
                mean_c_new = torch.einsum('...ij,...j->...i', cov_c_new, nat_mean_c)
                ####################################################################
                # CONSTRAINT FUNCTION: CONVERGENCE UPDATE
                ####################################################################
                # Compute the next damping factor by setting the damping to zero when
                # the change in the natural factors of the covariance is less than
                # the threshold
                (damping_c,
                 delta_mean_c, delta_cov_c) = _update_damping_when_converged(
                    mean_c, mean_c_new,
                    cov_c, cov_c_new, damping_c,
                    threshold=self.threshold, iteration=iteration
                )
                cov_c = cov_c_new
                mean_c = mean_c_new
            else:
                ####################################################################
                # OBJECTIVE FUNCTION: OMEGA UPDATE
                ####################################################################
                (omega_f_nat_mean_new, omega_f_nat_cov_new, _, _) = _update_omega(
                    mean_f, cov_f, omega_f_nat_mean, omega_f_nat_cov,
                    N, P, M, self.maximize, self.ep_jitter
                )

                ####################################################################
                # OBJECTIVE FUNCTION: MARGINAL UPDATE
                ####################################################################
                (nat_mean_f_new, nat_cov_f_new, _, _) = _update_marginals(
                    pred_f_nat_mean, pred_f_nat_cov,
                    omega_f_nat_mean_new, omega_f_nat_cov_new,
                    N, P
                )
            ########################################################################
            # OBJECTIVE FUNCTION: DAMPING UPDATE
            ########################################################################
            # update damping of objectives
            damping, cholesky_nat_cov_f = _update_damping(
                nat_cov_f, nat_cov_f_new, damping, self.ep_jitter
            )
            ########################################################################
            # OBJECTIVE FUNCTION: DAMPED UPDATE
            ########################################################################
            # Damp update of omega
            omega_f_nat_mean = _damped_update(
                omega_f_nat_mean, omega_f_nat_mean_new, damping
            )

            omega_f_nat_cov = _damped_update(
                omega_f_nat_cov, omega_f_nat_cov_new, damping
            )
            # update the mean and covariance
            nat_mean_f = _damped_update(nat_mean_f, nat_mean_f_new, damping)
            nat_cov_f = _damped_update(nat_cov_f, nat_cov_f_new, damping)

            # compute cholesky inverse
            cov_f_new = torch.cholesky_inverse(cholesky_nat_cov_f)
            # cov_f_new = torch.inverse(nat_cov_f)
            mean_f_new = torch.einsum('...ij,...j->...i', cov_f_new, nat_mean_f)
            ########################################################################
            # OBJECTIVE FUNCTION: CONVERGENCE UPDATE
            ########################################################################
            # Set the damping to zero when the change in the mean and
            # covariance is less than the threshold
            damping, delta_mean_f, delta_cov_f = _update_damping_when_converged(
                mean_f, mean_f_new,
                cov_f, cov_f_new, damping,
                threshold=self.threshold, iteration=iteration
            )
            cov_f = cov_f_new
            mean_f = mean_f_new

            if self.verbose:
                print("i={:4d}, time taken={:5.2f}, time elapsed={:5.2f}".format(
                    iteration,
                    time.time() - loop_time,
                    time.time() - start_time)
                )
                print("delta_cov_f={:.8f}, damping_completion={:.4f}/{:.4f}".format(
                    torch.max(delta_cov_f),
                    full_damp - torch.sum(damping), full_damp)
                )
                print("delta_mean_f={:.8f}, damping_completion={:.4f}/{:.4f}".format(
                    torch.max(delta_mean_f),
                    full_damp - torch.sum(damping), full_damp)
                )

                if K > 0:
                    print("delta_cov_c={:.8f}, damping_c_completion={:.4f}/{:.4f}".
                          format(torch.max(delta_cov_c),
                                 full_damp - torch.sum(damping), full_damp)
                          )
                    print("delta_mean_c={:.8f}, damping_c_completion={:.4f}/{:.4f}".
                          format(torch.max(delta_mean_c),
                                 full_damp - torch.sum(damping), full_damp)
                          )

            iteration = iteration + 1

        ############################################################################
        # SAVE OMEGA AND PHI FACTORS
        ############################################################################
        check_no_nans(omega_f_nat_mean)
        check_no_nans(omega_f_nat_cov)
        # save phi and omega for the forward
        self._omega_f_nat_mean = omega_f_nat_mean
        self._omega_f_nat_cov = omega_f_nat_cov

        if self.constraint_model is not None:
            check_no_nans(omega_c_nat_mean)
            check_no_nans(omega_c_nat_var)
            check_no_nans(phi_nat_mean)
            check_no_nans(phi_nat_var)
            self._omega_c_nat_mean = omega_c_nat_mean
            self._omega_c_nat_var = omega_c_nat_var
            self._phi_nat_mean = phi_nat_mean
            self._phi_nat_var = phi_nat_var
        else:
            self._omega_c_nat_mean = None
            self._omega_c_nat_var = None
            self._phi_nat_mean = None
            self._phi_nat_var = None

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qPredictiveEntropySearch on the candidate set `X`.
        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim
                design points each.
        Returns:
            A `batch_shape'`-dim Tensor of Predictive Entropy Search values at the
            given design points `X`.
        """
        batch_shape = X.shape[0:-2]
        q = X.shape[-2]
        M = self.model.num_outputs
        if M > 1:
            N = len(self.model.train_inputs[0][0])
        else:
            N = len(self.model.train_inputs[0])
        P = self.pareto_sets.shape[-2]
        num_pareto_samples = self.num_pareto_samples
        ###########################################################################
        # AUGMENT X WITH THE SAMPLED PARETO SET
        ###########################################################################
        new_shape = batch_shape + torch.Size([num_pareto_samples]) + X.shape[-2:]
        expanded_X = X.unsqueeze(-3).expand(new_shape)
        expanded_ps = self.pareto_sets.expand(X.shape[0:-2] + self.pareto_sets.shape)
        # `batch_shape x num_pareto_samples x (q + P) x d`
        aug_X = torch.cat([expanded_X, expanded_ps], dim=-2)

        ###########################################################################
        # COMPUTE THE POSTERIORS AND OBSERVATION NOISE
        ###########################################################################
        # compute predictive distribution without observation noise
        (pred_f_nat_mean, pred_f_nat_cov,
         pred_f_mean, pred_f_cov) = _initialize_predictive_matrices(
            aug_X, self.model, observation_noise=False, jitter=self.test_jitter
        )

        (_, _, _, pred_f_cov_noise) = _initialize_predictive_matrices(
            aug_X, self.model, observation_noise=True, jitter=self.test_jitter,
            natural=False
        )

        observation_noise = pred_f_cov_noise - pred_f_cov

        if self.constraint_model is not None:
            K = self.constraint_model.num_outputs

            (pred_c_nat_mean, pred_c_nat_cov,
             pred_c_mean, pred_c_cov) = _initialize_predictive_matrices(
                aug_X, self.constraint_model, observation_noise=False,
                jitter=self.test_jitter
            )

            (_, _, _, pred_c_cov_noise) = _initialize_predictive_matrices(
                aug_X, self.constraint_model, observation_noise=True,
                jitter=self.test_jitter, natural=False
            )
            observation_noise_c = pred_c_cov_noise - pred_c_cov
        else:
            K = 0
        ###########################################################################
        # INITIALIZE THE EP FACTORS
        ###########################################################################
        # `batch_shape x num_pareto_samples x M x (q + P) x P x 2`
        omega_f_nat_mean = torch.zeros(
            batch_shape + torch.Size([num_pareto_samples, M, q + P, P, 2])
        )
        # `batch_shape x num_pareto_samples x M x (q + P) x P x 2 x 2`
        omega_f_nat_cov = torch.zeros(
            batch_shape + torch.Size([num_pareto_samples, M, q + P, P, 2, 2])
        )

        if K > 0:
            # `batch_shape x num_pareto_samples x K x (q + P) x P`
            omega_c_nat_mean = torch.zeros(
                batch_shape + torch.Size([num_pareto_samples, K, q + P, P])
            )
            # `batch_shape x num_pareto_samples x K x (q + P) x P`
            omega_c_nat_var = torch.zeros(
                batch_shape + torch.Size([num_pareto_samples, K, q + P, P])
            )
        ###########################################################################
        # RUN EP ONCE
        ###########################################################################
        if K > 0:
            # run update omega once
            (omega_f_nat_mean, omega_f_nat_cov,
             omega_c_nat_mean, omega_c_nat_var) = _update_omega(
                pred_f_mean, pred_f_cov,
                omega_f_nat_mean, omega_f_nat_cov,
                q, P, M, self.maximize, self.test_jitter, K,
                pred_c_nat_mean, pred_c_nat_cov,
                omega_c_nat_mean, omega_c_nat_var
            )
        else:
            # run update omega once
            (omega_f_nat_mean, omega_f_nat_cov, _, _) = _update_omega(
                pred_f_mean, pred_f_cov,
                omega_f_nat_mean, omega_f_nat_cov,
                q, P, M, self.maximize, self.test_jitter
            )
        ###########################################################################
        # ADD THE CACHE FACTORS BACK
        ###########################################################################
        if K > 0:
            (omega_f_nat_mean, omega_f_nat_cov,
             omega_c_nat_mean, omega_c_nat_var,
             phi_nat_mean, phi_nat_var) = _augment_factors_with_cached_factors(
                batch_shape, q, N,
                omega_f_nat_mean, self._omega_f_nat_mean,
                omega_f_nat_cov, self._omega_f_nat_cov,
                omega_c_nat_mean, self._omega_c_nat_mean,
                omega_c_nat_var, self._omega_c_nat_var,
                self._phi_nat_mean, self._phi_nat_var
            )
        else:
            (omega_f_nat_mean, omega_f_nat_cov,
             _, _, _, _) = _augment_factors_with_cached_factors(
                batch_shape, q, N,
                omega_f_nat_mean, self._omega_f_nat_mean,
                omega_f_nat_cov, self._omega_f_nat_cov
            )
        ###########################################################################
        # COMPUTE THE MARGINAL
        ###########################################################################
        if K > 0:
            (nat_mean_f, nat_cov_f,
             nat_mean_c, nat_cov_c) = _update_marginals(
                pred_f_nat_mean, pred_f_nat_cov,
                omega_f_nat_mean, omega_f_nat_cov,
                q, P, K,
                pred_c_nat_mean, pred_c_nat_cov,
                phi_nat_mean, phi_nat_var,
                omega_c_nat_mean, omega_c_nat_var
            )

        else:
            (nat_mean_f, nat_cov_f, _, _) = _update_marginals(
                pred_f_nat_mean, pred_f_nat_cov,
                omega_f_nat_mean, omega_f_nat_cov,
                q, P
            )
        ###########################################################################
        # COMPUTE THE DAMPED UPDATE
        ###########################################################################
        # # update damping of objectives
        damping = torch.ones(batch_shape + torch.Size([num_pareto_samples, M]))
        damping, cholesky_nat_cov_f_new = _update_damping(
            pred_f_nat_cov, nat_cov_f, damping, self.test_jitter
        )

        # invert matrix
        cov_f_new = torch.cholesky_inverse(cholesky_nat_cov_f_new)
        # cov_f_new = torch.inverse(nat_cov_f_new + identity)
        check_no_nans(cov_f_new)

        # update damping of constraints
        if K > 0:
            damping_c = torch.ones(
                batch_shape + torch.Size([num_pareto_samples, K])
            )
            damping_c, cholesky_nat_cov_c_new = _update_damping(
                pred_c_nat_cov, nat_cov_c, damping_c, self.test_jitter
            )

            # invert matrix
            cov_c_new = torch.cholesky_inverse(cholesky_nat_cov_c_new)
            # cov_c_new = torch.inverse(nat_cov_c_new + self.jitter * identity_c)
            check_no_nans(cov_c_new)

        ###########################################################################
        # COMPUTE THE LOG DETERMINANTS
        ###########################################################################
        # compute the initial log determinant term
        log_det_pred_f_cov_noise = _compute_log_determinant(pred_f_cov_noise, q)
        # compute the post log determinant term
        log_det_cov_f = _compute_log_determinant(cov_f_new + observation_noise, q)
        if K > 0:
            # compute the initial log determinant term
            log_det_pred_c_cov_noise = _compute_log_determinant(pred_c_cov_noise, q)
            # compute the post log determinant term
            log_det_cov_c = _compute_log_determinant(
                cov_c_new + observation_noise_c, q
            )
        ###########################################################################
        # COMPUTE THE ACQUISITION FUNCTION
        ###########################################################################
        q_pes_f = log_det_pred_f_cov_noise - log_det_cov_f
        check_no_nans(q_pes_f)
        if K > 0:
            q_pes_c = log_det_pred_c_cov_noise - log_det_cov_c
            check_no_nans(q_pes_c)
        else:
            q_pes_c = 0

        return .5 * (q_pes_f + q_pes_c)


def log_cdf_robust(x: Tensor) -> Tensor:
    r""" Computes the logarithm of the normal cumulative density robustly. This uses
    the approximation log(1-z) ~ -z when z is small:
    if x > 5:
        log(cdf(x)) = log(1-cdf(-x)) approx -cdf(-x)
    else:
        log(cdf(x)).

    Args:
        x: a `x_shape`-dim Tensor.

    Returns
        A `x_shape`-dim Tensor.
    """
    normal = Normal(torch.zeros_like(x), torch.ones_like(x))
    cdf_x = normal.cdf(x)
    neg_cdf_neg_x = - normal.cdf(-x)
    log_cdf_x = torch.where(x < 5, torch.log(cdf_x), neg_cdf_neg_x)

    return log_cdf_x.clamp(NEG_INF, -CLAMP_LB)


def _initialize_predictive_matrices(
        X: Tensor,
        model: Model,
        observation_noise: Optional[bool] = True,
        jitter: Optional[float] = 1e-4,
        natural: Optional[bool] = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r""" Initializes the natural predictive mean and covariance matrix. For a
    multivariate normal distribution with mean mu and covariance Sigma, the natural
    mean is Sigma^{-1} mu and the natural covariance is Sigma^{-1}.

    Args:
        X: A `batch_shape x R x d`-dim Tensor.
        model: The fitted model.
        observation_noise: If true the posterior is computed with observation noise.
        jitter: The jitter added to the covariance matrix.
        natural: If true we compute the natural statistics as well.
    Return:
        pred_nat_mean: A `batch_shape x num_outputs x R `-dim Tensor containing the
            predictive natural mean vectors.
        pred_nat_cov: A `batch_shape x num_outputs x R x R`-dim Tensor containing
            the predictive natural covariance matrices.
        pred_mean: A `batch_shape x num_outputs x R`-dim Tensor containing the
            predictive mean vectors.
        pred_cov: A `batch_shape x num_outputs x R x R`-dim Tensor containing the
            predictive covariance matrices.
    """
    # compute the predictive mean and covariances at X
    posterior = model.posterior(X, observation_noise=observation_noise)

    # `batch_shape x (R * num_outputs) x (R * num_outputs)`
    init_pred_cov = posterior.mvn.covariance_matrix
    num_outputs = model.num_outputs
    R = int(init_pred_cov.shape[-1] / num_outputs)
    pred_cov = [
        init_pred_cov[..., (m * R):((m+1) * R), (m * R):((m+1) * R)].unsqueeze(-1)
        for m in range(num_outputs)
    ]
    # `batch_shape x R x R x num_outputs` (before swap axes)
    # `batch_shape x num_outputs x R * R`
    pred_cov = torch.cat(pred_cov, axis=-1).swapaxes(-2, -1).swapaxes(-3, -2)
    identity = torch.diag_embed(torch.ones(pred_cov.shape[:-1]))
    pred_cov = pred_cov + jitter * identity

    # `batch_shape x num_outputs x R`
    pred_mean = posterior.mean.swapaxes(-2, -1)

    #############################################################
    if natural:
        # natural parameters
        # `batch_shape x num_outputs x R x R`
        cholesky_pred_cov, _ = torch.linalg.cholesky_ex(pred_cov)
        pred_nat_cov = torch.cholesky_inverse(cholesky_pred_cov)
        # pred_nat_cov = torch.inverse(pred_cov + jitter)

        # `batch_shape x num_outputs x R`
        pred_nat_mean = torch.einsum('...ij,...j->...i', pred_nat_cov, pred_mean)

        return pred_nat_mean, pred_nat_cov, pred_mean, pred_cov
    else:
        return None, None, pred_mean, pred_cov


def _update_phi(
        mean: Tensor,
        cov: Tensor,
        phi_nat_mean: Tensor,
        phi_nat_var: Tensor,
        N: int
) -> Tuple[Tensor, Tensor]:
    r""" Computes the new phi factors by matching the moments.

    Args:
        mean: A `batch_shape x K x (N + P)`-dim Tensor containing the mean vector
            for the constraints.
        cov: A `batch_shape x K x (N + P) x (N + P)`-dim Tensor
            containing the covariance matrix for the constraints.
        phi_nat_mean: A `batch_shape x K x P`-dim Tensor containing the phi natural
            mean factors for the constraint matrix.
        phi_nat_var: A `batch_shape x K x P`-dim Tensor containing the phi natural
            variance factors for the constraint matrix.
        N: The number of design points.
        P: The number of Pareto optimal points.
        K: The number of constraints
    Return:
        phi_nat_mean_new: A `batch_shape x K x P`-dim Tensor containing the new phi
            natural mean factors for the constraint matrix.
        phi_nat_var_new: A `batch_shape x K x P`-dim Tensor containing the new phi
            natural variance factors for the constraint matrix.
    """

    ###############################################################################
    # EXTRACT THE NECESSARY COMPONENTS
    ###############################################################################
    # get the required contribution
    # i.e. the Pareto optimal points in the constrained matrix
    # here `S` is used to denote the Pareto set
    mean_cS = mean[..., N:]
    var_cS = cov[..., N:, N:]
    nat_var_cS = 1.0 / var_cS
    nat_mean_cS = nat_var_cS * mean_cS
    ###############################################################################
    # COMPUTE THE CAVITIES
    ###############################################################################
    # cavity distribution
    # natural parameters
    cav_nat_mean = nat_mean_cS - phi_nat_mean
    cav_nat_var = nat_var_cS - phi_nat_var
    check_no_nans(cav_nat_var)
    check_no_nans(cav_nat_mean)
    ###############################################################################
    # COMPUTE THE NORMALIZATION CONSTANT
    ###############################################################################
    # transform to standard parameters
    cav_var = 1.0 / cav_nat_var
    cav_mean = cav_var * cav_nat_mean
    cav_std = torch.sqrt(cav_var)

    # standardized normal value
    cav_alpha = cav_mean / cav_std
    check_no_nans(cav_alpha)

    # compute log normalization constant
    # Equation 20
    # this is logZ
    # compute the robust log cdf
    cav_alpha_log_cdf = log_cdf_robust(cav_alpha)
    # compute the log pdf
    normal = Normal(torch.zeros_like(cav_alpha), torch.ones_like(cav_alpha))
    cav_alpha_log_pdf = normal.log_prob(cav_alpha)
    ###############################################################################
    # COMPUTE THE PARTIAL DERIVATIVES
    ###############################################################################
    # first partial derivation of the log Z with respect to the mean
    # we compute robust version using exp-log trick

    # Equation 21
    cav_pdf_div_cdf = torch.exp(cav_alpha_log_pdf - cav_alpha_log_cdf)
    cav_dlogZ_dm = cav_pdf_div_cdf / cav_std

    # second partial derivation of the log Z with respect to the mean
    # we compute robust version using exp-log trick
    # Equation 22
    # [Typo in supp material should be exp(..)*(alpha + exp(..)) / v]
    cav_d2logZ_dm2 = - cav_pdf_div_cdf * (cav_alpha + cav_pdf_div_cdf) / cav_var
    ###############################################################################
    # COMPUTE THE NEW MEAN AND COVARIANCE
    ###############################################################################
    # compute the new mean and variance
    cav_updated_mean = cav_mean + cav_var * cav_dlogZ_dm
    cav_updated_var = cav_var + cav_var * cav_d2logZ_dm2 * cav_var

    # convert to natural parameters
    cav_updated_nat_var = 1.0 / cav_updated_var
    cav_updated_nat_mean = cav_updated_nat_var * cav_updated_mean

    # match the moments to compute the gain
    phi_nat_mean_new = cav_updated_nat_mean - cav_nat_mean
    phi_nat_var_new = cav_updated_nat_var - cav_nat_var

    return phi_nat_mean_new, phi_nat_var_new


def _get_omega_f_contribution(
        mean: Tensor,
        cov: Tensor,
        N: int,
        P: int,
        M: int
) -> Tuple[Tensor, Tensor]:
    r"""Extract the mean vector and covariance matrix corresponding to the `2 x 2`
    multivariate normal blocks in the objective model between the points in `X` and
    the Pareto optimal set.

    [There is likely a more efficient way to do this.]

    Args:
        mean: A `batch_shape x M x (N + P)`-dim Tensor containing the natural
            mean matrix for the objectives.
        cov: A `batch_shape x M x (N + P) x (N + P)`-dim Tensor containing
            the natural mean matrix for the objectives.
        N: The number of design points.
        P: The number of Pareto optimal points.
        M: The number of objectives.
    Return:
        mean_fX_fS: A `batch_shape x M x (N + P) x P x 2`-dim Tensor
        cov_fX_fS: A `batch_shape x M x (N + P) x P x 2 x 2`-dim Tensor
    """

    batch_shape = mean.shape[:-2]
    # `batch_shape x M x (N + P) x P x 2 x 2`
    cov_fX_fS = torch.zeros(batch_shape + torch.Size([M, N + P, P, 2, 2]))
    # `batch_shape x M x (N + P) x P x 2`
    mean_fX_fS = torch.zeros(batch_shape + torch.Size([M, N + P, P, 2]))

    # `batch_shape x M x (N + P) x P`
    mean_fX_fS[..., 0] = mean.unsqueeze(-1).expand(
        batch_shape + torch.Size([M, N + P, P])
    )
    # `batch_shape x M x (N + P) x P`
    mean_fX_fS[..., 1] = mean[..., N:].unsqueeze(-2).expand(
        batch_shape + torch.Size([M, N + P, P])
    )
    # `batch_shape x M x (N + P) x P`
    cov_fX_fS[..., 0, 0] = cov[..., range(N+P), range(N+P)].unsqueeze(-1).expand(
        batch_shape + torch.Size([M, N + P, P])
    )
    # `batch_shape x M x (N + P) x P`
    cov_fX_fS[..., 1, 1] = cov[..., range(N, N+P), range(N, N+P)].unsqueeze(-2).expand(
        batch_shape + torch.Size([M, N + P, P])
    )

    for p in range(P):
        # `batch_shape x M x (N + P)`
        cov_p = cov[..., range(N+P), N+p]
        cov_fX_fS[..., p, 0, 1] = cov_p
        cov_fX_fS[..., p, 1, 0] = cov_p

    return mean_fX_fS, cov_fX_fS


def _jitter_pareto_diagonal(A: Tensor, jitter=1e-3, replace=True) -> Tensor:
    """Jitter the pareto diagonal with identity matrices or replace with identity.

    The Pareto diagonal of the omega factor shouldn't be updated because does
    not contribute anything: `omega(x_p, x_p) = 1` for any pareto optimal input
    `x_p`.

    Args:
        A: a `batch_shape x M x (N + P) x P x 2 x 2`-dim Tensor.
        jitter: the amount of jitter used if replace is False.
        replace: if true we just replace the diagonal with the identity matrix.

    Returns:
        A `batch_shape x M x (N + P) x P x 2 x 2`-dim Tensor, where the
        Pareto diagonal is padded with identity matrices.
    """
    batch_shape = A.shape[:-5]
    P = A.shape[-3]
    N = A.shape[-4] - P
    M = A.shape[-5]
    identity = torch.diag_embed(
        torch.ones(batch_shape + torch.Size([M, 2]))
    )
    for p in range(P):
        if replace:
            A[..., N + p, p, :, :] = identity
        else:
            A[..., N + p, p, :, :] = A[..., N + p, p, :, :] + jitter * identity

    return A


def _get_omega_c_contribution(
        mean: Tensor,
        cov: Tensor,
        N: int,
        P: int,
        K: int
) -> Tuple[Tensor, Tensor]:
    r"""Extract the mean and variances between the points in `X` and the Pareto
    optimal set in the constraint model.

    Args:
        mean: A `batch_shape x K x (N + P)`-dim Tensor containing the mean vectors
            for the constraints.
        cov: A `batch_shape x K x (N + P) x (N + P)`-dim Tensor containing the
            covariance matrices for the constraints.
        N: The number of design points.
        P: The number of Pareto optimal points.
        K: The number of constraints.
    Return:
        mean_cX_cS: A `batch_shape x K x (N + P) x P`-dim Tensor
        var_cX_cS: A `batch_shape x K x (N + P) x P``-dim Tensor
    """
    batch_shape = mean.shape[:-1]
    new_shape = torch.zeros(batch_shape + torch.Size([N + P, P, K]))

    # `batch_shape x K x (N + P) x P`
    mean_cX_cS = mean[..., N:].unsqueeze(-2).expand(new_shape)
    var_cX_cS = cov[..., N:].unsqueeze(-2).expand(new_shape)

    return mean_cX_cS, var_cX_cS


def _update_omega(
        mean_f: Tensor,
        cov_f: Tensor,
        omega_f_nat_mean: Tensor,
        omega_f_nat_cov: Tensor,
        N: int,
        P: int,
        M: int,
        maximize: bool = True,
        jitter: Optional[float] = 1e-6,
        K: Optional[int] = 0,
        mean_c: Optional[Tensor] = None,
        cov_c: Optional[Tensor] = None,
        omega_c_nat_mean: Optional[Tensor] = None,
        omega_c_nat_var: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Computes the new Omega factors by matching the moments.

    Args:
        mean_f: A `batch_shape x M x (N + P)`-dim Tensor containing the mean vector
            for the objectives.
        cov_f: A `batch_shape x M x (N + P) x (N + P)`-dim Tensor
            containing the covariance matrix for the objectives.
        omega_f_nat_mean: A `batch_shape x M x (N + P) x P x 2`-dim Tensor containing
            the omega natural mean factors for the objective matrix.
        omega_f_nat_cov: A `batch_shape x M x (N + P) x P x 2 x 2`-dim Tensor
            containing the omega natural covariance factors for the objective
            matrix.
        N: The number of design points.
        P: The number of Pareto optimal points.
        M: The number of objectives.
        maximize: If true we consider the Pareto maximum domination relation.
        jitter: The jitter for the matrix inverse.
        K: The number of constraints.
        mean_c: A `batch_shape x K x (N + P)`-dim Tensor containing the natural
            mean matrix for the constraints.
        cov_c: A `batch_shape x K x (N + P) x (N + P)`-dim Tensor
            containing the natural covariance matrix for the constraints.
        omega_c_nat_mean: A `batch_shape x K x (N + P) x P`-dim Tensor containing
            the omega natural mean factors for the constraint matrix.
        omega_c_nat_var: A `batch_shape x K x (N + P) x P`-dim Tensor containing
            the omega natural covariance factors for the constraint matrix.
    Return:
        omega_f_nat_mean_new: A `batch_shape x M x (N + P) x P x 2` containing the
            new omega natural mean factors for the objective matrix.
        omega_f_nat_cov_new: A `batch_shape x M x (N + P) x P x 2 x 2` containing the
            new omega natural covariance factors for the objective matrix.
        omega_c_nat_mean_new: A `batch_shape x K x (N + P) x P`-dim Tensor containing
            the new omega natural mean factors for the constraint matrix.
        omega_c_nat_var_new: A `batch_shape x K x (N + P) x P`-dim Tensor containing
            the new  omega natural variance factors for the constraint matrix.
    """
    weight = 1.0 if maximize else -1.0
    ###############################################################################
    # EXTRACT THE NECESSARY COMPONENTS
    ###############################################################################
    # `batch_shape x M x (N + P) x P x 2`-dim mean
    # `batch_shape x M x (N + P) x P x 2 x 2`-dim covariance
    mean_fX_fS, cov_fX_fS = _get_omega_f_contribution(
        mean_f, cov_f, N, P, M
    )
    # remove the Pareto diagonal
    cov_fX_fS = _jitter_pareto_diagonal(cov_fX_fS, replace=True)
    nat_cov_fX_fS = torch.inverse(cov_fX_fS)
    nat_mean_fX_fS = torch.einsum('...ij,...j->...i', nat_cov_fX_fS, mean_fX_fS)

    if K > 0:
        # `batch_shape x K x (N + P) x P`-dim mean
        # `batch_shape x K x (N + P) x P`-dim covariance
        mean_cX_cS, var_cX_cS = _get_omega_c_contribution(mean_c, cov_c, N, P, K)
        nat_var_cX_cS = 1.0 / var_cX_cS
        nat_mean_cX_cS = nat_var_cX_cS * mean_cX_cS
    ###############################################################################
    # COMPUTE THE CAVITIES
    ###############################################################################
    # cavity distribution
    # natural parameters
    cav_nat_mean_f = nat_mean_fX_fS - omega_f_nat_mean
    cav_nat_cov_f = nat_cov_fX_fS - omega_f_nat_cov

    # transform to standard parameters
    # remove the Pareto diagonal
    cav_nat_cov_f = _jitter_pareto_diagonal(cav_nat_cov_f, replace=True)
    identity = torch.diag_embed(torch.ones(cav_nat_cov_f.shape[:-1]))
    cav_cov_f = torch.inverse(cav_nat_cov_f + jitter * identity)
    check_no_nans(cav_cov_f)

    # cav_cov_f = torch.inverse(cav_nat_cov_f)
    cav_mean_f = torch.einsum('...ij,...j->...i', cav_cov_f, cav_nat_mean_f)

    if K > 0:
        # cavity distribution
        # natural parameters
        cav_nat_mean_c = nat_mean_cX_cS - omega_c_nat_mean
        cav_nat_var_c = nat_var_cX_cS - omega_c_nat_var

        # transform to standard parameters
        cav_var_c = 1.0 / cav_nat_var_c
        cav_mean_c = cav_var_c * cav_nat_mean_c
        cav_std_c = torch.sqrt(cav_var_c)
    ###############################################################################
    # COMPUTE THE NORMALIZATION CONSTANT
    ###############################################################################
    # `batch_shape x M x (N + P) x P`
    # Equation 29
    cav_var_fX_minus_fS = (
            cav_cov_f[..., 0, 0] + cav_cov_f[..., 1, 1] - 2 * cav_cov_f[..., 0, 1]
    ).clamp_min(CLAMP_LB)
    cav_std_fX_minus_fS = torch.sqrt(cav_var_fX_minus_fS).clamp_min(CLAMP_LB)

    # `batch_shape x M x (N + P) x P`
    cav_mean_fX_minus_fS = weight * (cav_mean_f[..., 0] - cav_mean_f[..., 1])

    # Equation 30
    cav_alpha = cav_mean_fX_minus_fS / cav_std_fX_minus_fS
    check_no_nans(cav_alpha)
    # compute alpha pdf and cdf
    normal_alpha = Normal(torch.zeros_like(cav_alpha), torch.ones_like(cav_alpha))
    # `batch_shape x M x (N + P) x P`
    cav_alpha_log_cdf = log_cdf_robust(cav_alpha)
    # `batch_shape x M x (N + P) x P`
    cav_alpha_log_pdf = normal_alpha.log_prob(cav_alpha).clamp_min(NEG_INF)
    # `batch_shape x (N + P) x P`
    cav_sum_alpha_log_cdf = torch.sum(cav_alpha_log_cdf, dim=-3).clamp_min(NEG_INF)

    # compute normalization constant Z
    # Equation 35
    cav_log_zeta = torch.log1p(-torch.exp(cav_sum_alpha_log_cdf)).clamp_min(NEG_INF)

    # Need to clamp log values to prevent `exp(-inf) = nan`
    if K > 0:
        # Equation 31
        cav_beta = cav_mean_c / cav_std_c
        # compute beta pdf and cdf
        normal_beta = Normal(torch.zeros_like(cav_beta), torch.ones_like(cav_beta))
        # `batch_shape x K x (N + P) x P`
        cav_beta_log_cdf = log_cdf_robust(cav_beta)
        # `batch_shape x K x (N + P) x P`
        cav_beta_log_pdf = normal_beta.log_prob(cav_beta).clamp_min(NEG_INF)
        # `batch_shape x (N + P) x P`
        cav_sum_beta_log_cdf = torch.sum(cav_beta_log_cdf, dim=-3).clamp_min(NEG_INF)

        # Equation 36
        # this term represents `X` satisfying the constraint and being Pareto
        # dominated
        cav_log_eta = (cav_sum_beta_log_cdf + cav_log_zeta).clamp_min(NEG_INF)
        # Equation 37
        # this term represents `X` not satisfying the constraint
        cav_log_lambda = torch.log1p(-torch.exp(cav_sum_beta_log_cdf)).clamp_min(
            NEG_INF
        )
        # Equation 38
        cav_tau = torch.maximum(cav_log_eta, cav_log_lambda)
        # Equation 39
        cav_logZ = torch.log(
            torch.exp(cav_log_eta - cav_tau) +
            torch.exp(cav_log_lambda - cav_tau)
        ) + cav_tau

        # Equation 40 [first bit]
        # [contains typos]
        # `batch_shape x (N + P) x P`
        cav_log_rho = cav_sum_beta_log_cdf - cav_logZ + cav_sum_alpha_log_cdf

        # Equation 43
        # [some typos here as well]
        cav_log_omega = cav_sum_beta_log_cdf - cav_logZ + cav_sum_alpha_log_cdf
        cav_log_omega = cav_log_omega.unsqueeze(-3) - cav_beta_log_cdf + \
            cav_beta_log_pdf
        cav_omega = - torch.exp(cav_log_omega).clamp(NEG_INF, -NEG_INF)
        check_no_nans(cav_omega)
    else:
        cav_logZ = cav_log_zeta

        # Equation 40 [first bit]
        # `batch_shape x (N + P) x P`
        cav_log_rho = - cav_logZ + cav_sum_alpha_log_cdf

    # Equation 40 [second bit]
    # `batch_shape x M x (N + P) x P`
    cav_log_rho = cav_log_rho.unsqueeze(-3) - cav_alpha_log_cdf + cav_alpha_log_pdf
    cav_rho = - torch.exp(cav_log_rho).clamp(NEG_INF, -NEG_INF)
    check_no_nans(cav_rho)
    ###############################################################################
    # COMPUTE THE PARTIAL DERIVATIVES
    ###############################################################################
    # `batch_shape x M x (N + P) x P x 2`
    # Final vector: `[1, -1]`
    ones_mean = torch.ones(cav_mean_f.shape)
    ones_mean[..., 1] = - ones_mean[..., 1]

    # `batch_shape x M x (N + P) x P x 2 x 2`
    # Final matrix: `[[1, -1], [-1, 1]]`
    ones_cov = torch.ones(cav_cov_f.shape)
    ones_cov[..., 0, 1] = - ones_cov[..., 0, 1]
    ones_cov[..., 1, 0] = - ones_cov[..., 1, 0]

    # first partial derivation of the log Z with respect to the mean
    # assuming maximization (this is also where the sign will change)
    # Equation 41
    cav_dlogZ_dm = cav_rho / cav_std_fX_minus_fS
    cav_dlogZ_dm = weight * cav_dlogZ_dm.unsqueeze(-1) * ones_mean

    # second partial derivation of the log Z with respect to the mean
    # Equation 42
    cav_d2logZ_dm2 = - cav_rho * (cav_rho + cav_alpha) / cav_var_fX_minus_fS
    cav_d2logZ_dm2 = cav_d2logZ_dm2.unsqueeze(-1).unsqueeze(-1) * ones_cov
    check_no_nans(cav_d2logZ_dm2)

    if K > 0:
        # first partial derivation of the log Z with respect to the mean constraint
        # Equation 44
        # [typo here should be vij instead of sj]
        cav_dlogZ_dm_c = cav_omega / cav_std_c
        # second partial derivation of the log Z with respect to the mean constraint
        # Equation 44
        # [typo here should be vij instead of sj]
        cav_d2logZ_dm2_c = - cav_omega * (cav_omega + cav_beta) / cav_var_c
    ###############################################################################
    # COMPUTE THE NEW MEAN AND COVARIANCE
    ###############################################################################
    # compute the new mean and covariance
    cav_updated_mean_f = cav_mean_f + torch.einsum(
        '...ij,...j->...i', cav_cov_f, cav_dlogZ_dm
    )
    cav_updated_cov_f = cav_cov_f + torch.einsum(
        '...ij,...jk,...kl->...il', cav_cov_f, cav_d2logZ_dm2, cav_cov_f
    )
    # transform to natural parameters
    # remove the Pareto diagonal
    cav_updated_cov_f = _jitter_pareto_diagonal(cav_updated_cov_f, replace=True)
    check_no_nans(cav_updated_cov_f)

    identity = torch.diag_embed(torch.ones(cav_updated_cov_f.shape[:-1]))

    # if there is an inversion error here we don't update omega
    try:
        cav_updated_nat_cov_f = torch.inverse(cav_updated_cov_f + jitter * identity)
    except RuntimeError:
        if K > 0:
            return omega_f_nat_mean, omega_f_nat_cov, \
                   omega_c_nat_mean, omega_c_nat_var
        else:
            return omega_f_nat_mean, omega_f_nat_cov, None, None

    # cav_updated_nat_cov_f = torch.inverse(cav_updated_cov_f)
    check_no_nans(cav_updated_nat_cov_f)

    cav_updated_nat_mean_f = torch.einsum(
        '...ij,...j->...i', cav_updated_nat_cov_f, cav_updated_mean_f
    )

    # match the moments to compute the gain
    omega_f_nat_mean_new = cav_updated_nat_mean_f - cav_nat_mean_f
    omega_f_nat_cov_new = cav_updated_nat_cov_f - cav_nat_cov_f

    # it is also possible to calculate the update directly as in the original paper:
    # identity = torch.diag_embed(torch.ones(cav_d2logZ_dm2.shape[:-1]))
    # denominator = torch.inverse(cav_cov_f @ cav_d2logZ_dm2 + identity)
    # omega_f_nat_cov_new = - cav_d2logZ_dm2 @ denominator
    # omega_f_nat_mean_new = torch.einsum(
    #     '...ij,...j->...i', denominator,
    #     cav_dlogZ_dm - torch.einsum('...ij,...j->...i', cav_d2logZ_dm2, cav_mean_f)
    # )

    if K > 0:
        cav_updated_mean_c = cav_mean_c + cav_var_c * cav_dlogZ_dm_c
        cav_updated_var_c = cav_var_c + cav_var_c * cav_d2logZ_dm2_c * cav_var_c

        # transform to natural parameters
        cav_updated_nat_var_c = (1.0 / cav_updated_var_c).clamp_min(CLAMP_LB)
        cav_updated_nat_mean_c = cav_updated_nat_var_c * cav_updated_mean_c

        # match the moments to compute the gain
        omega_c_nat_mean_new = cav_updated_nat_mean_c - cav_nat_mean_c
        omega_c_nat_var_new = cav_updated_nat_var_c - cav_nat_var_c

        return omega_f_nat_mean_new, omega_f_nat_cov_new, \
            omega_c_nat_mean_new, omega_c_nat_var_new
    else:
        return omega_f_nat_mean_new, omega_f_nat_cov_new, None, None


def _update_marginals(
        pred_f_nat_mean: Tensor,
        pred_f_nat_cov: Tensor,
        omega_f_nat_mean: Tensor,
        omega_f_nat_cov: Tensor,
        N: int,
        P: int,
        K: Optional[int] = 0,
        pred_c_nat_mean: Optional[Tensor] = None,
        pred_c_nat_cov: Optional[Tensor] = None,
        phi_nat_mean: Optional[Tensor] = None,
        phi_nat_var: Optional[Tensor] = None,
        omega_c_nat_mean: Optional[Tensor] = None,
        omega_c_nat_var: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Computes the new marginal by summing up all the natural factors.

    Args:
        pred_f_nat_mean: A `batch_shape x M x (N + P)`-dim Tensor containing the
            natural predictive mean matrix for the objectives.
        pred_f_nat_cov: A `batch_shape x M x (N + P) x (N + P)`-dim Tensor containing
            the natural predictive covariance matrix for the objectives.
        omega_f_nat_mean: A `batch_shape x M x (N + P) x P x 2`-dim Tensor containing
            the omega natural mean factors for the objective matrix.
        omega_f_nat_cov: A `batch_shape x M x (N + P) x P x 2 x 2`-dim Tensor
            containing the omega natural covariance factors for the objective
            matrix.
        N: The number of design points.
        P: The number of Pareto optimal points.
        K: The number of constraints.
        pred_c_nat_mean: A `batch_shape x K x (N + P)`-dim Tensor containing the
            natural predictive mean matrix for the constraints.
        pred_c_nat_cov: A `batch_shape x K x (N + P) x (N + P)`-dim Tensor containing
            the natural predictive covariance matrix for the constraints.
        phi_nat_mean: A `batch_shape x K x P`-dim Tensor containing the phi natural
            mean factors for the constraint matrix.
        phi_nat_var: A `batch_shape x K x P`-dim Tensor containing the phi natural
            variance factors for the constraint matrix.
        omega_c_nat_mean: A `batch_shape x K x (N + P) x P`-dim Tensor containing
            the omega natural mean factors for the constraint matrix.
        omega_c_nat_var: A `batch_shape x K x (N + P) x P`-dim Tensor containing
            the omega natural covariance factors for the constraint matrix.

    Returns:
        nat_mean_f: A `batch_shape x M x (N + P)`-dim Tensor containing the updated
            natural mean matrix for the objectives.
        nat_cov_f: A `batch_shape x M x (N + P) x (N + P)`-dim Tensor containing the
            updated natural predictive covariance matrix for the objectives.
        nat_mean_c: A `batch_shape x K x (N + P)`-dim Tensor containing the updated
            natural mean matrix for the constraints.
        nat_cov_c: A `batch_shape x K x (N + P) x (N + P)`-dim Tensor containing the
            updated natural predictive covariance matrix for the constraints.
    """

    # `batch_shape x M x (N + P)`
    nat_mean_f = pred_f_nat_mean.clone()
    # `batch_shape x M x (N + P) x (N + P)
    nat_cov_f = pred_f_nat_cov.clone()

    ################################################################################
    # UPDATE THE CONSTRAINTS
    ################################################################################
    if K > 0:
        # `batch_shape x K x (N + P)`
        nat_mean_c = pred_c_nat_mean
        # `batch_shape x K x (N + P) x (N + P)`
        nat_cov_c = pred_c_nat_cov

        # `batch_shape x K x P`
        nat_mean_c[..., N:] = nat_mean_c[..., N:] + phi_nat_mean
        # `batch_shape x K x P`
        nat_cov_c[..., range(N, N+P), range(N, N+P)] = \
            nat_cov_c[..., range(N, N+P), range(N, N+P)] + phi_nat_var

        # zero out the diagonal
        omega_c_nat_mean[..., range(N, N + P), range(P)] = 0
        omega_c_nat_var[..., range(N, N + P), range(P)] = 0

        # `batch_shape x K x P`
        nat_mean_c[..., N:] = nat_mean_c[..., N:] + omega_c_nat_mean.sum(dim=-2)
        # `batch_shape x K x P`
        nat_cov_c[..., N:, N:] = \
            nat_cov_c[..., N:, N:] + omega_c_nat_var.sum(dim=-2)

    ################################################################################
    # UPDATE THE OBJECTIVES
    ################################################################################
    # remove Pareto diagonal
    # zero out the diagonal
    omega_f_nat_mean[..., range(N, N + P), range(P), :] = 0
    omega_f_nat_cov[..., range(N, N + P), range(P), :, :] = 0

    # `batch_shape x M x (N + P)`
    # sum over the pareto dim
    nat_mean_f = nat_mean_f + omega_f_nat_mean[..., 0].sum(dim=-1)
    # `batch_shape x M x P`
    # sum over the data dim
    nat_mean_f[..., N:] = nat_mean_f[..., N:] + omega_f_nat_mean[..., 1].sum(dim=-2)

    # `batch_shape x M x (N + P)`
    nat_cov_f[..., range(N+P), range(N+P)] = \
        nat_cov_f[..., range(N+P), range(N+P)] \
        + omega_f_nat_cov[..., 0, 0].sum(dim=-1)
    # `batch_shape x M x P`
    nat_cov_f[..., range(N, N+P), range(N, N+P)] = \
        nat_cov_f[..., range(N, N+P), range(N, N+P)] \
        + omega_f_nat_cov[..., 1, 1].sum(dim=-2)

    for p in range(P):
        # `batch_shape x M x (N + P)`
        nat_cov_f[..., range(N + P), N + p] = \
            nat_cov_f[..., range(N + P), N + p] + omega_f_nat_cov[..., p, 0, 1]

        # `batch_shape x M x (N + P)`
        nat_cov_f[..., N + p, range(N + P)] = \
            nat_cov_f[..., N + p, range(N + P)] + omega_f_nat_cov[..., p, 1, 0]

    if K > 0:
        return nat_mean_f, nat_cov_f, nat_mean_c, nat_cov_c
    else:
        return nat_mean_f, nat_cov_f, None, None


def _damped_update(
        old_factor: Tensor,
        new_factor: Tensor,
        damping_factor: Tensor,
) -> Tensor:
    r""" Computes the damped updated for natural factor.

    Args:
        old_factor: A `batch_shape x param_shape`-dim Tensor containing the old
            natural factor.
        new_factor: A `batch_shape x param_shape`-dim Tensor containing the new
            natural factor.
        damping_factor: A `batch_shape`-dim Tensor containing the damping factor.

    Returns:
        A `batch_shape x param_shape`-dim Tensor containing the updated natural
            factor.
    """
    bs = damping_factor.shape
    fs = old_factor.shape

    df = damping_factor
    for i in range(len(fs[len(bs):])):
        df = df.unsqueeze(-1)

    return df * new_factor + (1 - df) * old_factor


def _update_damping(
        nat_cov: Tensor,
        nat_cov_new: Tensor,
        damping_factor: Tensor,
        jitter: Tensor,
) -> Tuple[Tensor, Tensor]:
    r""" Updates the damping factor whilst ensuring the covariance matrix is
    positive definite by trying a Cholesky decomposition.

    Args:
        nat_cov: A `batch_shape x R x R`-dim Tensor containing the old natural
            covariance matrix.
        nat_cov_new: A `batch_shape x R x R`-dim Tensor containing the new natural
            covariance matrix.
        damping_factor: A`batch_shape`-dim Tensor containing the damping factor.
        jitter: The amount of jitter added before matrix inversion.

    Returns:
        A `batch_shape x param_shape`-dim Tensor containing the updated damping
            factor.
        A `batch_shape x R x R`-dim Tensor containing the Cholesky factor.
    """
    df = damping_factor
    jitter = jitter * torch.diag_embed(torch.ones(nat_cov.shape[:-1]))
    _, info = torch.linalg.cholesky_ex(nat_cov + jitter)

    if torch.sum(info) > 1:
        raise ValueError("The previous covariance is not positive semi-definite. "
                         "This usually happens if the predictive covariance is "
                         "ill-conditioned and the added jitter is enough to fix this.")

    damped_nat_cov = _damped_update(nat_cov, nat_cov_new, df)
    cholesky_factor, info = torch.linalg.cholesky_ex(damped_nat_cov)

    run = 0
    while (torch.sum(info) > 1) and (run < 100):
        # propose an alternate damping factor which is half the original
        df_alt = .5 * df
        # hard threshold at 1e-3
        df_alt = torch.where(
            df_alt > 1e-3, df_alt, torch.zeros(df_alt.shape)
        )
        # only change the damping factor where psd failure occurs
        df_new = torch.where(info == 0, df, df_alt)

        # new damped covariance
        damped_nat_cov = _damped_update(nat_cov, nat_cov_new, df_new)

        # try cholesky decomposition
        cholesky_factor, info = torch.linalg.cholesky_ex(damped_nat_cov + jitter)
        df = df_new
        run += 1

    # EP failed
    if run == 100:
        raise ValueError("Expectation propagation failed.")

    return df, cholesky_factor


def _update_damping_when_converged(
        mean_old: Tensor,
        mean_new: Tensor,
        cov_old: Tensor,
        cov_new: Tensor,
        damping_factor: Tensor,
        iteration: Tensor,
        threshold: float = 1e-3,
) -> Tensor:
    r"""Set the damping factor to 0 once converged. Convergence is determined by the
    relative change in the entries of the mean and covariance matrix.

    Args:
        mean_old: A `batch_shape x R`-dim Tensor containing the old natural mean
            matrix for the objective.
        mean_new: A `batch_shape x R`-dim Tensor containing the new natural mean
            matrix for the objective.
        cov_old: A `batch_shape x R x R`-dim Tensor containing the old natural
            covariance matrix for the objective.
        cov_new: A `batch_shape x R x R`-dim Tensor containing the new natural
            covariance matrix for the objective.
        iteration: The current iteration number
        damping_factor: A `batch_shape`-dim Tensor containing the damping factor.

    Returns:
        A `batch_shape x param_shape`-dim Tensor containing the updated damping
            factor.
    """
    df = damping_factor.clone()
    delta_mean = mean_new - mean_old
    delta_cov = cov_new - cov_old
    am = torch.amax(abs(mean_old), dim=-1)
    ac = torch.amax(abs(cov_old), dim=(-2, -1))

    if iteration > 2:
        mask_mean = torch.amax(abs(delta_mean), dim=-1) < threshold * am
        mask_cov = torch.amax(abs(delta_cov), dim=(-2, -1)) < threshold * ac
        mask = torch.logical_and(mask_mean, mask_cov)
        df[mask] = 0

    return df, delta_mean, delta_cov


def _augment_factors_with_cached_factors(
        batch_shape: Size,
        q: int,
        N: int,
        omega_f_nat_mean: Tensor,
        cached_omega_f_nat_mean: Tensor,
        omega_f_nat_cov: Tensor,
        cached_omega_f_nat_cov: Tensor,
        omega_c_nat_mean: Optional[Tensor] = None,
        cached_omega_c_nat_mean: Optional[Tensor] = None,
        omega_c_nat_var: Optional[Tensor] = None,
        cached_omega_c_nat_var: Optional[Tensor] = None,
        cached_phi_nat_mean: Optional[Tensor] = None,
        cached_phi_nat_var: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Incorporate the cached Pareto updated factors in the forward call and
    augment them with the previously computed factors.

    TODO: This can be done much cleaner.

    Args:
        batch_shape: The batch shape of `X`.
        q: The batch size.
        N: The number of training points.
        omega_f_nat_mean: A `batch_shape x num_pareto_samples x M x (q + P) x P x 2`
            -dim Tensor containing the omega natural mean for the objective at `X`.
        cached_omega_f_nat_mean: A `num_pareto_samples x M x (N + P) x P x 2`-dim
            Tensor containing the omega natural mean for the objective at `X`.
        omega_f_nat_cov: A `batch_shape x num_pareto_samples x M x (q + P) x P x 2
            x 2` -dim Tensor containing the omega natural covariance for the
            objective at `X`.
        cached_omega_f_nat_cov: A `num_pareto_samples x M x (N + P) x P x 2 x 2`-dim
            Tensor containing the omega covariance mean for the objective at `X`.

        omega_c_nat_mean: A `batch_shape x num_pareto_samples x K x (q + P) x P`-dim
            Tensor containing the omega natural mean for the constraint at `X`.
        cached_omega_c_nat_mean: A `num_pareto_samples x K x (N + P) x P`-dim Tensor
            containing the omega natural mean for the constraint at `X`.
        omega_c_nat_var: A `batch_shape x num_pareto_samples x K x (q + P) x P`-dim
            Tensor containing the omega natural variance for the constraint at `X`.
        cached_omega_c_nat_var: A `num_pareto_samples x K x (N + P) x P`-dim Tensor
            containing the omega variance mean for the constraint at `X`.

        cached_phi_nat_mean: A `num_pareto_samples x K x P`-dim Tensor containing the
            phi natural mean for the constraint at `X`.
        cached_phi_nat_var: A `num_pareto_samples x K x P`-dim Tensor containing the
            phi variance mean for the constraint at `X`.

    Returns:
        omega_f_nat_mean_new: A `batch_shape x num_pareto_samples x M x (q + P) x P x
            2`-dim Tensor containing the omega natural mean for the objective at `X`.
        omega_f_nat_cov_new: A `batch_shape x num_pareto_samples x M x (q + P) x P x
            2 x 2`-dim Tensor containing the omega natural covariance for the
            objective at `X`.
        omega_c_nat_mean_new: A `batch_shape x num_pareto_samples x K x (q + P) x P`
            -dim Tensor containing the omega natural mean for the constraint at `X`.
        omega_c_nat_var_new: A `batch_shape x num_pareto_samples x K x (q + P) x P`
            -dim Tensor containing the omega natural variance for the constraint at
            `X`.
        phi_nat_mean_new: A `batch_shape x num_pareto_samples x K x P` -dim Tensor
            containing the omega natural mean for the constraint at `X`.
        phi_nat_var_new: A `batch_shape x num_pareto_sample x Ks x P`-dim Tensor
            containing the phi natural variance for the constraint at `X`.
    """
    ##############################################################################
    # omega_f_nat_mean
    ##############################################################################
    # retrieve the natural mean contribution of the Pareto block omega(x_p, x_p) for
    # the objective
    exp_cached_omega_f_nat_mean = cached_omega_f_nat_mean[..., N:, :, :].expand(
        omega_f_nat_mean[..., q:, :, :].shape
    )
    omega_f_nat_mean[..., q:, :, :] = exp_cached_omega_f_nat_mean
    ##############################################################################
    # omega_f_nat_cov
    ##############################################################################
    # retrieve the natural covariance contribution of the Pareto block
    # omega(x_p, x_p) for the objective
    exp_omega_f_nat_cov = cached_omega_f_nat_cov[..., N:, :, :, :].expand(
        omega_f_nat_cov[..., q:, :, :, :].shape
    )
    omega_f_nat_cov[..., q:, :, :, :] = exp_omega_f_nat_cov
    ##############################################################################
    # omega_c_nat_mean
    ##############################################################################
    # retrieve the natural mean contribution of the Pareto block omega(x_p, x_p) for
    # the constraint
    if omega_c_nat_mean is not None:
        exp_cached_omega_c_nat_mean = cached_omega_c_nat_mean[..., N:, :].expand(
            expanded_shape = omega_c_nat_mean[..., q:, :].shape
        )
        omega_c_nat_mean[..., q:, :] = exp_cached_omega_c_nat_mean
    ##############################################################################
    # omega_c_nat_var
    ##############################################################################
    # retrieve the natural variance contribution of the Pareto block omega(x_p, x_p)
    # for the constraint
    if omega_c_nat_var is not None:
        exp_cached_omega_c_nat_var = cached_omega_c_nat_var[..., N:, :].expand(
            omega_c_nat_var[..., q:, :].shape
        )
        omega_c_nat_var[..., q:, :] = exp_cached_omega_c_nat_var
    ##############################################################################
    # phi_nat_mean
    ##############################################################################
    # retrieve the natural mean contribution from phi(x_p, x_p) for the constraint
    if cached_phi_nat_mean is not None:
        expanded_shape = batch_shape + cached_phi_nat_mean.shape
        phi_nat_mean = cached_phi_nat_mean.expand(expanded_shape)
    else:
        phi_nat_mean = None
    ##############################################################################
    # phi_nat_var
    ##############################################################################
    # retrieve the natural variance contribution from phi(x_p, x_p) for the
    # constraint
    if cached_phi_nat_var is not None:
        expanded_shape = batch_shape + cached_phi_nat_var.shape
        phi_nat_var = cached_phi_nat_var.expand(expanded_shape)
    else:
        phi_nat_var = None

    return (omega_f_nat_mean, omega_f_nat_cov,
            omega_c_nat_mean, omega_c_nat_var,
            phi_nat_mean, phi_nat_var)


def _compute_log_determinant(
        cov: Tensor,
        q: int
) -> Tensor:
    r""" Computes the sum of the log determinants of a block diagonal covariance
    matrices averaged over the Pareto samples.

    Args:
        cov: A `batch_shape x num_pareto_samples x num_outputs x (q + P) x (q + P)`
            -dim Tensor containing the covariance matrices.
        q: The batch size.
    Return:
        log_det_cov: A `batch_shape`-dim Tensor containing the sum of the
            determinants for each output.
    """
    log_det_cov = torch.logdet(cov[..., 0:q, 0:q])
    check_no_nans(log_det_cov)

    return log_det_cov.sum(dim=-1).mean(dim=-1)
