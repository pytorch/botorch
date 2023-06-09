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


class LinearEllipticalSliceSampler(PolytopeSampler):
    r"""Linear Elliptical Slice Sampler.

    TODOs:
    - clean up docstrings
    - Add batch support, broadcasting over parallel chains.
    - optimize computations (if possible)

    Maybe TODOs:
    - Support degenerate domains (with zero volume)?
    """

    def __init__(
        self,
        inequality_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        bounds: Optional[Tensor] = None,
        interior_point: Optional[Tensor] = None,
        mean: Optional[Tensor] = None,
        covariance_matrix: Optional[Tensor] = None,
        covariance_root: Optional[Tensor] = None,
        check_feasibility: bool = False,
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
            check_feasibility: If True, raise an error if the sampling results in an
                infeasible sample. This creates some overhead and so is switched off
                by default.

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
        self.check_feasibility = check_feasibility

    def draw(self, n: int = 1) -> Tuple[Tensor, Tensor]:
        r"""Draw samples.

        Args:
            n: The number of samples.

        Returns:
            A `n x d`-dim tensor of `n` samples.
        """
        # TODO: Should apply thinning in higher dimensions, can check step size.
        samples = torch.cat([self.step() for _ in range(n)], dim=-1)
        return samples.transpose(-1, -2)

    def step(self) -> Tensor:
        r"""Take a step, return the new sample, update the internal state.

        Returns:
            A `d x 1`-dim sample from the domain.
        """
        nu = self._sample_base_rv()
        theta = self._draw_angle(nu=nu)
        x = self._get_cart_coords(nu=nu, theta=theta)
        self._x[:] = x
        if self.check_feasibility and (not self._is_feasible(self._x)):
            Axmb = self.A @ self._x - self.b
            violated_indices = Axmb > 0
            raise RuntimeError(
                "Sampling resulted in infeasible point. \n\t- Number "
                f"of violated constraints: {violated_indices.sum()}."
                f"\n\t- Magnitude of violations: {Axmb[violated_indices]}"
                "\n\t- If the error persists, please report this bug on GitHub."
            )
        return x

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
            A `1`-dim Tensor containing the rotation angle (radians).
        """
        rot_angle, rot_slices = self._find_rotated_intersections(nu)
        rot_lengths = rot_slices[:, 1] - rot_slices[:, 0]
        cum_lengths = torch.cumsum(rot_lengths, dim=0)
        cum_lengths = torch.cat((self._zero, cum_lengths), dim=0)
        rnd_angle = cum_lengths[-1] * torch.rand(
            1, device=cum_lengths.device, dtype=cum_lengths.dtype
        )
        idx = torch.searchsorted(cum_lengths, rnd_angle) - 1
        return (rot_slices[idx, 0] + rnd_angle + rot_angle) - cum_lengths[idx]

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
        slices = (slices - rot_angle).reshape(-1, 2)
        # Ensuring that we don't sample within numerical precision of the boundaries
        # due to resulting instabilities in the constraint satisfaction.
        eps = 1e-6 if slices.dtype == torch.float32 else 1e-12
        eps = torch.tensor(eps, dtype=slices.dtype, device=slices.device)
        eps = eps.minimum(slices.diff(dim=-1).abs() / 4)
        slices = slices + torch.cat((eps, -eps), dim=-1)
        # NOTE: The remainder call relies on the epsilon contraction, since the
        # remainder of_twopi divided by _twopi is zero, not _twopi.
        return rot_angle, slices.remainder(_twopi)

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
        theta_active, delta_active = self._active_theta_and_delta(
            nu=nu,
            theta=theta,
        )
        if theta_active.numel() == 0:
            theta_active = self._full_angular_range
            # TODO: What about `self.ellipse_in_domain = False` in the original code?
        elif delta_active[0] == -1:  # ensuring that the first interval is feasible

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
        theta = torch.where(theta < 0, theta + _twopi, theta)  # in [0, 2*pi]
        return torch.sort(theta).values

    def _active_theta_and_delta(self, nu: Tensor, theta: Tensor) -> Tensor:
        r"""Determine active indices.

        Args:
            nu: A `d x 1`-dim tensor (the "new" direction, drawn from N(0, I)).
            theta: A sorted `M`-dim tensor of intersection angles in [0, 2pi].

        Returns:
            A tuple of Tensors of active constraint intersection angles `theta_active`,
            and the change in the feasibility of the points on the ellipse on the left
            and right of the active intersection angles `delta_active`. `delta_active`
            is is negative if decreasing the angle renders the sample feasible, and
            positive if increasing the angle renders the sample feasible.
        """
        # In order to determine if an angle that gives rise to an intersection with a
        # constraint boundary leads to a change in the feasibility of the solution,
        # we evaluate the constraints on the midpoint of the intersection angles.
        # This gets rid of the `delta_theta` parameter in the original implementation,
        # which cannot be set universally since it can be both 1) too large, when
        # the distance in adjacent intersection angles is small, and 2) too small,
        # when it approaches the numerical precision limit.
        # The implementation below solves both problems and gets rid of the parameter.
        if len(theta) < 2:  # if we have no or only a tangential intersection
            theta_active = torch.tensor([], dtype=theta.dtype, device=theta.device)
            delta_active = torch.tensor([], dtype=int, device=theta.device)
            return theta_active, delta_active
        theta_mid = (theta[:-1] + theta[1:]) / 2  # midpoints of intersection angles
        last_mid = (theta[:1] + theta[-1:] + _twopi) / 2
        last_mid = last_mid.where(last_mid < _twopi, last_mid - _twopi)
        theta_mid = torch.cat((last_mid, theta_mid, last_mid), dim=0)
        samples_mid = self._get_cart_coords(nu=nu, theta=theta_mid)
        delta_feasibility = self._is_feasible(samples_mid).to(dtype=int).diff()
        active_indices = delta_feasibility.nonzero()
        return theta[active_indices], delta_feasibility[active_indices]

    def _is_feasible(self, points: Tensor) -> Tensor:
        r"""Returns a Boolean tensor indicating whether the `points` are feasible,
        i.e. they satisfy `A @ points <= b`, where `(A, b)` are the tensors passed
        as the `inequality_constraints` to the constructor of the sampler.

        Args:
            points: A `d x M`-dim tensor of points.

        Returns:
            An `M`-dim binary tensor where `True` indicates that the associated
            point is feasible.
        """
        return (self.A @ points <= self.b).all(dim=0)
