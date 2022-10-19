#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Linear Elliptical Slice Sampler.

References

.. [Gessner2020]
    A. Gessner, O. Kanjilal, and P. Hennig. Integrals over gaussians under
    linear domain constraints. AISTATS 2020.


This implementation is based (with multiple changes / optimiations) on
the following implementations based on the algorithm in [Gessner2020]_:
https://github.com/alpiges/LinConGauss
https://github.com/wjmaddox/pytorch_ess
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from botorch.utils.sampling import PolytopeSampler
from torch import Tensor

_twopi = 2.0 * math.pi
_delta_theta = 1.0e-6 * _twopi


class LinearEllipticalSliceSampler(PolytopeSampler):
    r"""Linear Elliptical Slice Sampler.

    TODOs:
    - clean up docstrings
    - optimize computations (if possible)

    Maybe TODOs:
    - Support degenerate domains (with zero volume)?
    - Add batch support ?
    """

    def __init__(
        self,
        inequality_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        bounds: Optional[Tensor] = None,
        interior_point: Optional[Tensor] = None,
        mean: Optional[Tensor] = None,
        covariance_matrix: Optional[Tensor] = None,
        covariance_root: Optional[Tensor] = None,
    ) -> None:
        r"""Initialize LinearEllipticalSliceSampler.

        Args:
            inequality_constraints: Tensors `(A, b)` describing inequality constraints
                 `A @ x <= b`, where `A` is an `n_ineq_con x d`-dim Tensor and `b` is
                 an `n_ineq_con x 1`-dim Tensor, with `n_ineq_con` the number of
                 inequalities and `d` the dimension of the sample space. If omitted,
                 must provide `bounds` instead.
            bounds: A `2 x d`-dim tensor of box bounds. If omitted, must provide
                `inequality_constraints` instead.
            interior_point: A `d x 1`-dim Tensor presenting a point in the (relative)
                interior of the polytope. If omitted, an interior point is determined
                automatically by solving a Linear Program. Note: It is crucial that
                the point lie in the interior of the feasible set (rather than on the
                boundary), otherwise the sampler will produce invalid samples.
            mean: The `d x 1`-dim mean of the MVN distribution (if omitted, use zero).
            covariance_matrix: The `d x d`-dim covariance matrix of the MVN
                distribution (if omitted, use the identity).
            covariance_root: A `d x k`-dim root of the covariance matrix such that
                covariance_root @ covariance_root.T = covariance_matrix.

        This sampler samples from a multivariante Normal `N(mean, covariance_matrix)`
        subject to linear domain constraints `A x <= b` (intersected with box bounds,
        if provided).
        """
        super().__init__(
            inequality_constraints=inequality_constraints,
            # TODO: Support equality constraints?
            interior_point=interior_point,
            bounds=bounds,
        )
        tkwargs = {"device": self.x0.device, "dtype": self.x0.dtype}
        self._mean = mean
        if covariance_matrix is not None:
            if covariance_root is not None:
                raise ValueError(
                    "Provide either covariance_matrix or covariance_root, not both."
                )
            try:
                covariance_root = torch.linalg.cholesky(covariance_matrix)
            except RuntimeError as e:
                raise_e = e
                if "positive-definite" in str(raise_e):
                    raise_e = ValueError(
                        "Covariance matrix is not positive definite. "
                        "Currently only non-degenerate distributions are supported."
                    )
                raise raise_e
        self._covariance_root = covariance_root
        self._x = self.x0.clone()  # state of the sampler ("current point")
        # We will need the following repeatedly, let's allocate them once
        self._zero = torch.zeros(1, **tkwargs)
        self._nan = torch.tensor(float("nan"), **tkwargs)
        self._full_angular_range = torch.tensor([0.0, _twopi], **tkwargs)

    def draw(self, n: int = 1) -> Tuple[Tensor, Tensor]:
        r"""Draw samples.

        Args:
            n: The number of samples.

        Returns:
            A `n x d`-dim tensor of `n` samples.
        """
        # TODO: Do we need to do any thinnning or warm-up here?
        samples = torch.cat([self.step() for _ in range(n)], dim=-1)
        return samples.transpose(-1, -2)

    def step(self) -> Tensor:
        r"""Take a step, return the new sample, update the internal state.

        Returns:
            A `d x 1`-dim sample from the domain.
        """
        nu = self._sample_base_rv()
        theta = self._draw_angle(nu=nu)
        self._x = self._get_cart_coords(nu=nu, theta=theta)
        return self._x

    def _sample_base_rv(self) -> Tensor:
        r"""Sample a base random variable from N(mean, covariance_matrix).

        Returns:
            A `d x 1`-dim sample from the domain
        """
        nu = torch.randn_like(self._x)
        if self._covariance_root is not None:
            nu = self._covariance_root @ nu
        if self._mean is not None:
            nu = self._mean + nu
        return nu

    def _draw_angle(self, nu: Tensor) -> Tensor:
        r"""Draw the rotation angle.

        Args:
            nu: A `d x 1`-dim tensor (the "new" direction, drawn from N(0, I)).

        Returns:
            A
        """
        rot_angle, rot_slices = self._find_rotated_intersections(nu)
        rot_lengths = rot_slices[:, 1] - rot_slices[:, 0]
        cum_lengths = torch.cumsum(rot_lengths, dim=0)
        cum_lengths = torch.cat((self._zero, cum_lengths), dim=0)
        rnd_angle = cum_lengths[-1] * torch.rand(
            1, device=cum_lengths.device, dtype=cum_lengths.dtype
        )
        idx = torch.searchsorted(cum_lengths, rnd_angle) - 1
        return rot_slices[idx, 0] + rnd_angle - cum_lengths[idx] + rot_angle

    def _get_cart_coords(self, nu: Tensor, theta: Tensor) -> Tensor:
        r"""Determine location on ellipsoid in cartesian coordinates.

        Args:
            nu: A `d x 1`-dim tensor (the "new" direction, drawn from N(0, I)).
            theta: A `k`-dim tensor of angles.

        Returns:
            A `d x k`-dim tensor of samples from the domain in cartesian coordinates.
        """
        return self._x * torch.cos(theta) + nu * torch.sin(theta)

    def _find_rotated_intersections(self, nu: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Finds rotated intersections.

        Rotates the intersections by the rotation angle and makes sure that all
        angles lie in [0, 2*pi].

        Args:
            nu: A `d x 1`-dim tensor (the "new" direction, drawn from N(0, I)).

        Returns:
            A two-tuple containing rotation angle (scalar) and a
            `num_active / 2 x 2`-dim tensor of shifted angles.
        """
        slices = self._find_active_intersections(nu)
        rot_angle = slices[0]
        slices = slices - rot_angle
        slices = torch.where(slices < 0, slices + _twopi, slices)
        return rot_angle, slices.reshape(-1, 2)

    def _find_active_intersections(self, nu: Tensor) -> Tensor:
        """
        Find angles of those intersections that are at the boundary of the integration
        domain by adding and subtracting a small angle and evaluating on the ellipse
        to see if we are on the boundary of the integration domain.

        Args:
            nu: A `d x 1`-dim tensor (the "new" direction, drawn from N(0, I)).

        Returns:
            A `num_active`-dim tensor containing the angles of active intersection in
            increasing order so that activation happens in positive direction. If a
            slice crosses `theta=0`, the first angle is appended at the end of the
            tensor. Every element of the returned tensor defines a slice for elliptical
            slice sampling.
        """
        theta = self._find_intersection_angles(nu)
        active_directions = self._index_active(
            nu=nu, theta=theta, delta_theta=_delta_theta
        )
        theta_active = theta[active_directions.nonzero()]
        delta_theta = _delta_theta
        while theta_active.numel() % 2 == 1:
            # Almost tangential ellipses, reduce delta_theta
            delta_theta /= 10
            active_directions = self._index_active(
                theta=theta, nu=nu, delta_theta=delta_theta
            )
            theta_active = theta[active_directions.nonzero()]

        if theta_active.numel() == 0:
            theta_active = self._full_angular_range
            # TODO: What about `self.ellipse_in_domain = False` in the original code ??
        elif active_directions[active_directions.nonzero()][0] == -1:
            theta_active = torch.cat((theta_active[1:], theta_active[:1]))

        return theta_active.view(-1)

    def _find_intersection_angles(self, nu: Tensor) -> Tensor:
        """Compute all of the up to 2*n_ineq_con intersections of the ellipse
        and the linear constraints.

        For background, see equation (2) in
        http://proceedings.mlr.press/v108/gessner20a/gessner20a.pdf

        Args:
            nu: A `d x 1`-dim tensor (the "new" direction, drawn from N(0, I)).

        Returns:
            An `M`-dim tensor, where `M <= 2 * n_ineq_con` (with `M = n_ineq_con`
            if all intermediate computations yield finite numbers).
        """
        # Compared to the implementation in https://github.com/alpiges/LinConGauss
        # we need to flip the sign of A b/c the original algorithm considers
        # A @ x + b >= 0 feasible, whereas we consider A @ x - b <= 0 feasible.
        g1 = -self.A @ self._x
        g2 = -self.A @ nu
        r = torch.sqrt(g1**2 + g2**2)
        phi = 2 * torch.atan(g2 / (r + g1)).squeeze()

        arg = -(self.b / r).squeeze()
        # Write NaNs if there is no intersection
        arg = torch.where(torch.absolute(arg) <= 1, arg, self._nan)

        # Two solutions per linear constraint, shape of theta: (n_ineq_con, 2)
        acos_arg = torch.arccos(arg)
        theta = torch.stack((phi + acos_arg, phi - acos_arg), dim=-1)
        theta = theta[torch.isfinite(theta)]  # shape: `n_ineq_con - num_not_finite`
        theta = torch.where(theta < 0, theta + _twopi, theta)  # [0, 2*pi]

        return torch.sort(theta).values

    def _index_active(
        self, nu: Tensor, theta: Tensor, delta_theta: float = _delta_theta
    ) -> Tensor:
        r"""Determine active indices.

        Args:
            nu: A `d x 1`-dim tensor (the "new" direction, drawn from N(0, I)).
            theta: An `M`-dim tensor of intersection angles.
            delta_theta: A small perturbation to be used for determining whether
                intersections are at the boundary of the integration domain.

        Returns:
            An `M`-dim tensor with elements taking on values in {-1, 0, 1}.
            A non-zero value indicates whether the associated intersection angle
            is an active constraint. For active constraints, the sign indicates
            the direction of the relevant domain (i.e. +1 (-1) means that
            increasing (decreasing) the angle renders the sample feasible).
        """
        samples_pos = self._get_cart_coords(nu=nu, theta=theta + delta_theta)
        samples_neg = self._get_cart_coords(nu=nu, theta=theta - delta_theta)
        pos_diffs = self._is_feasible(samples_pos)
        neg_diffs = self._is_feasible(samples_neg)
        # We don't use bit-wise XOR here since we need the signs of the directions
        return pos_diffs.to(nu) - neg_diffs.to(nu)

    def _is_feasible(self, points: Tensor) -> Tensor:
        r"""

        Args:
            points: A `M x d`-dim tensor of points.

        Returns:
            An `M`-dim binary tensor where `True` indicates that the associated
            point is feasible.
        """
        return (self.A @ points <= self.b).all(dim=0)
