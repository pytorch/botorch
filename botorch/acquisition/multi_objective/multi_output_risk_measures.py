#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Multi-output extensions of the risk measures, implemented as Monte-Carlo
objectives. Except for MVaR, the risk measures are computed over each
output dimension independently. In contrast, MVaR is computed using the
joint distribution of the outputs, and provides more accurate risk estimates.

References

.. [Prekopa2012MVaR]
    A. Prekopa. Multivariate value at risk and related topics.
    Annals of Operations Research, 2012.

.. [Cousin2013MVaR]
    A. Cousin and E. Di Bernardino. On multivariate extensions of Value-at-Risk.
    Journal of Multivariate Analysis, 2013.
"""

import warnings
from abc import ABC, abstractmethod
from math import ceil
from typing import Optional

import torch
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.risk_measures import CVaR, RiskMeasureMCObjective
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor


class MultiOutputRiskMeasureMCObjective(
    RiskMeasureMCObjective, MCMultiOutputObjective, ABC
):
    r"""Objective transforming the multi-output posterior samples to samples
    of a multi-output risk measure.

    The risk measure is calculated over joint q-batch samples from the posterior.
    If the q-batch includes samples corresponding to multiple inputs, it is assumed
    that first `n_w` samples correspond to first input, second `n_w` samples
    correspond to second input, etc.
    """

    def __init__(
        self,
        n_w: int,
        weights: Optional[Tensor] = None,
    ) -> None:
        r"""Transform the posterior samples to samples of a risk measure.

        Args:
            n_w: The size of the `w_set` to calculate the risk measure over.
            weights: An optional `m`-dim tensor of weights for scaling
                multi-output samples before calculating the risk measure.
                This can also be used to make sure that all outputs are
                correctly aligned for maximization by negating those that are
                originally defined for minimization.
        """
        super().__init__(n_w=n_w, weights=weights)

    def _prepare_samples(self, samples: Tensor) -> Tensor:
        r"""Prepare samples for risk measure calculations by scaling and
        separating out the q-batch dimension.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.

        Returns:
            A `sample_shape x batch_shape x q x n_w x m`-dim tensor of prepared samples.
        """
        if self.weights is not None:
            samples = samples * self.weights
        return samples.view(*samples.shape[:-2], -1, self.n_w, samples.shape[-1])

    @abstractmethod
    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Calculate the risk measure corresponding to the given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q x m`-dim tensor of risk measure samples.
        """
        pass  # pragma: no cover


class IndependentCVaR(CVaR, MultiOutputRiskMeasureMCObjective):
    r"""The multi-output Conditional Value-at-Risk risk measure that operates on
    each output dimension independently. Since this does not consider the joint
    distribution of the outputs (i.e., that the outputs were evaluated on same
    perturbed input and are not independent), the risk estimates provided by
    `IndependentCVaR` in general are more optimistic than the definition of CVaR
    would suggest.

    The Conditional Value-at-Risk measures the expectation of the worst outcomes
    (small rewards or large losses) with a total probability of `1 - alpha`. It
    is commonly defined as the conditional expectation of the reward function,
    with the condition that the reward is smaller than the corresponding
    Value-at-Risk (also defined below).

    NOTE: Due to the use of a discrete `w_set` of samples, the VaR and CVaR
    calculated here are (possibly biased) Monte-Carlo approximations of the
    true risk measures.
    """

    def _get_sorted_prepared_samples(self, samples: Tensor) -> Tensor:
        r"""Get the prepared samples that are sorted over the `n_w` dimension.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.

        Returns:
            A `sample_shape x batch_shape x q x n_w x m`-dim tensor of sorted samples.
        """
        prepared_samples = self._prepare_samples(samples)
        return prepared_samples.sort(dim=-2, descending=True).values

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Calculate the CVaR corresponding to the given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q x m`-dim tensor of CVaR samples.
        """
        sorted_samples = self._get_sorted_prepared_samples(samples)
        return sorted_samples[..., self.alpha_idx :, :].mean(dim=-2)


class IndependentVaR(IndependentCVaR):
    r"""The multi-output Value-at-Risk risk measure that operates on each output
    dimension independently. For the same reasons as `IndependentCVaR`, the risk
    estimates provided by this are in general more optimistic than the definition
    of VaR would suggest.

    Value-at-Risk measures the smallest possible reward (or largest possible loss)
    after excluding the worst outcomes with a total probability of `1 - alpha`. It
    is commonly used in financial risk management, and it corresponds to the
    `1 - alpha` quantile of a given random variable.
    """

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Calculate the VaR corresponding to the given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q x m`-dim tensor of VaR samples.
        """
        sorted_samples = self._get_sorted_prepared_samples(samples)
        return sorted_samples[..., self.alpha_idx, :]


class MultiOutputWorstCase(MultiOutputRiskMeasureMCObjective):
    r"""The multi-output worst-case risk measure."""

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Calculate the worst-case measure corresponding to the given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q x m`-dim tensor of worst-case samples.
        """
        prepared_samples = self._prepare_samples(samples)
        return prepared_samples.min(dim=-2).values


class MVaR(MultiOutputRiskMeasureMCObjective):
    r"""The multivariate Value-at-Risk as introduced in [Prekopa2012MVaR]_.

    MVaR is defined as the non-dominated set of points in the extended domain
    of the random variable that have multivariate CDF greater than or equal to
    `alpha`. Note that MVaR is set valued and the size of the set depends on the
    particular realizations of the random variable. [Cousin2013MVaR]_ instead
    propose to use the expectation of the set-valued MVaR as the multivariate
    VaR. We support this alternative with an `expectation` flag.
    """

    _verify_output_shape = False

    def __init__(
        self,
        n_w: int,
        alpha: float,
        expectation: bool = False,
        weights: Optional[Tensor] = None,
        pad_to_n_w: bool = False,
        filter_dominated: bool = True,
    ) -> None:
        r"""The multivariate Value-at-Risk.

        Args:
            n_w: The size of the `w_set` to calculate the risk measure over.
            alpha: The risk level of MVaR, float in `(0.0, 1.0]`. Each MVaR value
                dominates `alpha` fraction of all observations.
            expectation: If True, returns the expectation of the MVaR set as is
                done in [Cousin2013MVaR]_. Otherwise, it returns the union of all
                values in the MVaR set. Default: False.
            weights: An optional `m`-dim tensor of weights for scaling
                multi-output samples before calculating the risk measure.
                This can also be used to make sure that all outputs are
                correctly aligned for maximization by negating those that are
                originally defined for minimization.
            pad_to_n_w: If True, instead of padding up to `k'`, which is the size of
                the largest MVaR set across all batches, we pad the MVaR set up to
                `n_w`. This produces a return tensor of known size, however, it may
                in general be much larger than the alternative. See `forward` for
                more details on the return shape.
                NOTE: this is only relevant if `expectation=False`.
            filter_dominated: If True, returns the non-dominated subset of
                alpha level points (this is MVaR as defined by [Prekopa2012MVaR]_).
                Disabling this will make it faster, and may be preferable if
                the dominated points will be filtered out later, e.g., while
                calculating the hypervolume. Disabling this is not recommended
                if `expectation=True`.
        """
        super().__init__(n_w=n_w, weights=weights)
        if not 0 < alpha <= 1:
            raise ValueError("`alpha` must be in (0.0, 1.0]")
        self.alpha = alpha
        self.expectation = expectation
        self.pad_to_n_w = pad_to_n_w
        self.filter_dominated = filter_dominated

    def get_mvar_set_cpu(self, Y: Tensor) -> Tensor:
        r"""Find MVaR set based on the definition in [Prekopa2012MVaR]_.

        NOTE: This is much faster on CPU for large `n_w` than the alternative but it
        is significantly slower on GPU. Based on empirical evidence, this is recommended
        when running on CPU with `n_w > 64`.

        This first calculates the CDF for each point on the extended domain of the
        random variable (the grid defined by the given samples), then takes the
        values with CDF equal to (rounded if necessary) `alpha`. The non-dominated
        subset of these form the MVaR set.

        Args:
            Y: A `batch x n_w x m`-dim tensor of outcomes. This is currently
                restricted to `m = 2` objectives.
                TODO: Support `m > 2` objectives.

        Returns:
            A `batch` length list of `k x m`-dim tensor of MVaR values, where `k`
            depends on the corresponding batch inputs. Note that MVaR values in general
            are not in-sample points.
        """
        if Y.dim() == 3:
            return [self.get_mvar_set_cpu(y_) for y_ in Y]
        m = Y.shape[-1]
        if m != 2:  # pragma: no cover
            raise ValueError("`get_mvar_set_cpu` only supports `m=2` outcomes!")
        # Generate sets of all unique values in each output dimension.
        # Note that points in MVaR are bounded from above by the
        # independent VaR of each objective. Hence, we only need to
        # consider the unique outcomes that are less than or equal to
        # the VaR of the independent objectives
        var_alpha_idx = ceil(self.alpha * self.n_w) - 1
        Y_sorted = Y.topk(Y.shape[0] - var_alpha_idx, dim=0, largest=False).values
        unique_outcomes_list = [
            Y_sorted[:, i].unique().tolist()[::-1] for i in range(m)
        ]
        # Convert this into a list of m dictionaries mapping values to indices.
        unique_outcomes = [
            dict(zip(outcomes, range(len(outcomes))))
            for outcomes in unique_outcomes_list
        ]
        # Initialize a tensor counting the number of points in Y that a given grid point
        # is dominated by. This will essentially be a non-normalized CDF.
        counter_tensor = torch.zeros(
            [len(outcomes) for outcomes in unique_outcomes],
            dtype=torch.long,
            device=Y.device,
        )
        # populate the tensor, counting the dominated points.
        # we only need to consider points in Y where at least one
        # objective is less than the max objective value in
        # unique_outcomes_list
        max_vals = torch.tensor(
            [o[0] for o in unique_outcomes_list], dtype=Y.dtype, device=Y.device
        )
        mask = (Y < max_vals).any(dim=-1)
        counter_tensor += self.n_w - mask.sum()
        Y_pruned = Y[mask]
        for y_ in Y_pruned:
            starting_idcs = [unique_outcomes[i].get(y_[i].item(), 0) for i in range(m)]
            counter_tensor[starting_idcs[0] :, starting_idcs[1] :] += 1

        # Get the count alpha-level points should have.
        alpha_count = ceil(self.alpha * self.n_w)
        # Get the alpha level indices.
        alpha_level_indices = (counter_tensor == alpha_count).nonzero(as_tuple=False)
        # If there are no exact alpha level points, get the smallest alpha' > alpha
        # and find the corresponding alpha level indices.
        if alpha_level_indices.numel() == 0:
            min_greater_than_alpha = counter_tensor[counter_tensor > alpha_count].min()
            alpha_level_indices = (counter_tensor == min_greater_than_alpha).nonzero(
                as_tuple=False
            )
        unique_outcomes = [
            torch.as_tensor(list(outcomes.keys()), device=Y.device, dtype=Y.dtype)
            for outcomes in unique_outcomes
        ]
        alpha_level_points = torch.stack(
            [
                unique_outcomes[i][alpha_level_indices[:, i]]
                for i in range(len(unique_outcomes))
            ],
            dim=-1,
        )
        # MVaR is simply the non-dominated subset of alpha level points.
        if self.filter_dominated:
            mask = is_non_dominated(alpha_level_points)
            mvar = alpha_level_points[mask]
        else:
            mvar = alpha_level_points
        return mvar

    def get_mvar_set_gpu(self, Y: Tensor) -> Tensor:
        r"""Find MVaR set based on the definition in [Prekopa2012MVaR]_.

        NOTE: This is much faster on GPU than the alternative but it scales very poorly
        on CPU as `n_w` increases. This should be preferred if a GPU is available or
        when `n_w <= 64`. In addition, this supports `m >= 2` outcomes (vs `m = 2` for
        the CPU version) and it should be used if `m > 2`.

        This first calculates the CDF for each point on the extended domain of the
        random variable (the grid defined by the given samples), then takes the
        values with CDF equal to (rounded if necessary) `alpha`. The non-dominated
        subset of these form the MVaR set.

        Args:
            Y: A `batch x n_w x m`-dim tensor of observations.

        Returns:
            A `batch` length list of `k x m`-dim tensor of MVaR values, where `k`
            depends on the corresponding batch inputs. Note that MVaR values in general
            are not in-sample points.
        """
        if Y.dim() == 2:
            Y = Y.unsqueeze(0)
        batch, m = Y.shape[0], Y.shape[-1]
        # Note that points in MVaR are bounded from above by the
        # independent VaR of each objective. Hence, we only need to
        # consider the unique outcomes that are less than or equal to
        # the VaR of the independent objectives
        var_alpha_idx = ceil(self.alpha * self.n_w) - 1
        n_points = Y.shape[-2] - var_alpha_idx
        Y_sorted = Y.topk(n_points, dim=-2, largest=False).values
        # `y_grid` is the grid formed by all inputs in each batch.
        if m == 2:
            # This is significantly faster but only works with m=2.
            y_grid = torch.stack(
                [
                    Y_sorted[..., 0].repeat_interleave(repeats=n_points, dim=-1),
                    Y_sorted[..., 1].repeat(1, n_points),
                ],
                dim=-1,
            )
        else:
            y_grid = torch.stack(
                [
                    torch.stack(
                        torch.meshgrid([Y_sorted[b, :, i] for i in range(m)]),
                        dim=-1,
                    ).view(-1, m)
                    for b in range(batch)
                ],
                dim=0,
            )
        # Get the non-normalized CDF.
        cdf = (Y.unsqueeze(-2) >= y_grid.unsqueeze(-3)).all(dim=-1).sum(dim=-2)
        # Get the alpha level points
        alpha_count = ceil(self.alpha * self.n_w)
        # NOTE: Need to loop here since mvar may have different shapes.
        mvar = []
        for b in range(batch):
            alpha_level_points = y_grid[b][cdf[b] == alpha_count]
            # If there are no exact alpha level points, get the smallest alpha' > alpha
            # and find the corresponding alpha level indices.
            if alpha_level_points.numel() == 0:
                min_greater_than_alpha = cdf[b][cdf[b] > alpha_count].min()
                alpha_level_points = y_grid[b][cdf[b] == min_greater_than_alpha]
            # MVaR is the non-dominated subset of alpha level points.
            if self.filter_dominated:
                mask = is_non_dominated(alpha_level_points)
                mvar.append(alpha_level_points[mask])
            else:
                mvar.append(alpha_level_points)
        return mvar

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Calculate the MVaR corresponding to the given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q x m`-dim tensor of MVaR values,
            if `self.expectation=True`.
            Otherwise, this returns a `sample_shape x batch_shape x (q * k') x m`-dim
            tensor, where `k'` is the maximum `k` across all batches that is returned
            by `get_mvar_set_...`. Each `(q * k') x m` corresponds to the `k` MVaR
            values for each `q` batch of `n_w` inputs, padded up to `k'` by repeating
            the last element. If `self.pad_to_n_w`, we set `k' = self.n_w`, producing
            a deterministic return shape.
        """
        batch_shape, m = samples.shape[:-2], samples.shape[-1]
        prepared_samples = self._prepare_samples(samples)
        # This is -1 x n_w x m.
        prepared_samples = prepared_samples.reshape(-1, *prepared_samples.shape[-2:])
        # Get the mvar set using the appropriate method based on device, m & n_w.
        # NOTE: The `n_w <= 64` part is based on testing on a 24 core CPU.
        # `get_mvar_set_gpu` heavily relies on parallelized batch computations and
        # may scale worse on CPUs with fewer cores.
        # Using `no_grad` here since `MVaR` is not differentiable.
        with torch.no_grad():
            if (
                samples.device == torch.device("cpu")
                and m == 2
                and prepared_samples.shape[-2] <= 64
            ):
                mvar_set = self.get_mvar_set_cpu(prepared_samples)
            else:
                mvar_set = self.get_mvar_set_gpu(prepared_samples)
        if samples.requires_grad:
            # TODO: Investigate differentiability of MVaR.
            warnings.warn(
                "Got `samples` that requires grad, but computing MVaR involves "
                "non-differentable operations and the results will not be "
                "differentiable. This may lead to errors down the line!",
                RuntimeWarning,
            )
        # Set the `pad_size` to either `self.n_w` or the size of the largest MVaR set.
        pad_size = self.n_w if self.pad_to_n_w else max([_.shape[0] for _ in mvar_set])
        padded_mvar_list = []
        for mvar_ in mvar_set:
            if self.expectation:
                padded_mvar_list.append(mvar_.mean(dim=0))
            else:
                # Repeat the last entry to make `mvar_set` `n_w x m`.
                repeats_needed = pad_size - mvar_.shape[0]
                padded_mvar_list.append(
                    torch.cat([mvar_, mvar_[-1].expand(repeats_needed, m)], dim=0)
                )
        mvars = torch.stack(padded_mvar_list, dim=0)
        return mvars.view(*batch_shape, -1, m)
