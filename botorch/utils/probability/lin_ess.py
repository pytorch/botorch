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
- https://github.com/alpiges/LinConGauss
- https://github.com/wjmaddox/pytorch_ess

The implementation here differentiates itself from the original implementations with:
1) Support for fixed feature equality constraints.
2) Support for non-standard Normal distributions.
3) Numerical stability improvements, especially relevant for high-dimensional cases.

Notably, this implementation does not rely on an adaptive `delta_theta` parameter in
order to determine if two neighboring constraint intersection angles `theta` lead to a
change in the feasibility of the sample. This both simplifies the implementation and
makes it more robust to numerical imprecisions when two constraint intersection angles
are close to each other.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
from botorch.utils.sampling import PolytopeSampler
from torch import Tensor

_twopi = 2.0 * math.pi


class LinearEllipticalSliceSampler(PolytopeSampler):
    r"""Linear Elliptical Slice Sampler.

    Ideas:
    - Add batch support, broadcasting over parallel chains.
    - Optimize computations if possible, potentially with torch.compile.
    - Extend fixed features constraint to general linear equality constraints.
    """

    def __init__(
        self,
        inequality_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        bounds: Optional[Tensor] = None,
        interior_point: Optional[Tensor] = None,
        fixed_indices: Optional[Union[List[int], Tensor]] = None,
        mean: Optional[Tensor] = None,
        covariance_matrix: Optional[Tensor] = None,
        covariance_root: Optional[Tensor] = None,
        check_feasibility: bool = False,
        burnin: int = 0,
        thinning: int = 0,
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
            fixed_indices: Integer list or `d`-dim Tensor representing the indices of
                dimensions that are constrained to be fixed to the values specified in
                the `interior_point`, which is required to be passed in conjunction with
                `fixed_indices`.
            mean: The `d x 1`-dim mean of the MVN distribution (if omitted, use zero).
            covariance_matrix: The `d x d`-dim covariance matrix of the MVN
                distribution (if omitted, use the identity).
            covariance_root: A `d x d`-dim root of the covariance matrix such that
                covariance_root @ covariance_root.T = covariance_matrix. NOTE: This
                matrix is assumed to be lower triangular.
            check_feasibility: If True, raise an error if the sampling results in an
                infeasible sample. This creates some overhead and so is switched off
                by default.
            burnin: Number of samples to generate upon initialization to warm up the
                sampler.
            thinning: Number of samples to skip before returning a sample in `draw`.

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
        if covariance_matrix is not None and covariance_root is not None:
            raise ValueError(
                "Provide either covariance_matrix or covariance_root, not both."
            )

        # can't unpack inequality constraints directly if bounds are passed
        A, b = self.A, self.b
        self._Az, self._bz = A, b
        self._is_fixed, self._not_fixed = None, None
        if fixed_indices is not None:
            mean, covariance_matrix = self._fixed_features_initialization(
                A=A,
                b=b,
                interior_point=interior_point,
                fixed_indices=fixed_indices,
                mean=mean,
                covariance_matrix=covariance_matrix,
                covariance_root=covariance_root,
            )

        self._mean = mean
        # Have to delay factorization until after fixed features initialization.
        if covariance_matrix is not None:  # implies root is None
            covariance_root, info = torch.linalg.cholesky_ex(covariance_matrix)
            not_psd = torch.any(info)
            if not_psd:
                raise ValueError(
                    "Covariance matrix is not positive definite. "
                    "Currently only non-degenerate distributions are supported."
                )
        self._covariance_root = covariance_root

        # Rewrite the constraints as a system that constrains a standard Normal.
        self._standardization_initialization()

        # state of the sampler ("current point")
        self._x = self.x0.clone()
        self._z = self._transform(self._x)

        # We will need the following repeatedly, let's allocate them once
        self._zero = torch.zeros(1, **tkwargs)
        self._nan = torch.tensor(float("nan"), **tkwargs)
        self._full_angular_range = torch.tensor([0.0, _twopi], **tkwargs)
        self.check_feasibility = check_feasibility
        self._lifetime_samples = 0
        if burnin > 0:
            self.thinning = 0
            self.draw(burnin)
        self.thinning = thinning

    def _fixed_features_initialization(
        self,
        A: Tensor,
        b: Tensor,
        interior_point: Optional[Tensor],
        fixed_indices: Union[List[int], Tensor],
        mean: Optional[Tensor],
        covariance_matrix: Optional[Tensor],
        covariance_root: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Modifies the constraint system (A, b) due to fixed indices and assigns
        the modified constraints system to `self._Az`, `self._bz`. NOTE: Needs to be
        called prior to `self._standardization_initialization` in the constructor.

        Returns:
            Tuple of `mean` and `covariance_matrix` tensors of the non-fixed dimensions.
        """
        if interior_point is None:
            raise ValueError(
                "If `fixed_indices` are provided, an interior point must also be "
                "provided in order to infer feasible values of the fixed features."
            )
        if covariance_root is not None:
            raise ValueError(
                "Provide either covariance_root or fixed_indices, not both."
            )
        d = interior_point.shape[0]
        is_fixed, not_fixed = get_index_tensors(fixed_indices=fixed_indices, d=d)
        self._is_fixed = is_fixed
        self._not_fixed = not_fixed
        # Transforming constraint system to incorporate fixed features:
        # A @ x - b = (A[:, fixed] @ x[fixed] + A[:, not fixed] @ x[not fixed]) - b
        #           = A[:, not fixed] @ x[not fixed] - (b - A[:, fixed] @ x[fixed])
        #           = Az @ z - bz
        self._Az = A[:, not_fixed]
        self._bz = b - A[:, is_fixed] @ interior_point[is_fixed]
        if mean is not None:
            mean = mean[not_fixed]
        if covariance_matrix is not None:  # subselect active dimensions
            covariance_matrix = covariance_matrix[
                not_fixed.unsqueeze(-1), not_fixed.unsqueeze(0)
            ]
        return mean, covariance_matrix

    def _standardization_initialization(self) -> None:
        """For non-standard mean and covariance, we're going to rewrite the problem as
        sampling from a standard normal distribution subject to modified constraints.
            A @ x - b = A @ (covar_root @ z + mean) - b
                      = (A @ covar_root) @ z - (b - A @ mean)
                      = _Az @ z - _bz
        NOTE: We need to standardize bz before Az in the following, because it relies
        on the untransformed Az. We can't simply use A instead because Az might have
        been subject to the fixed features transformation.
        """
        if self._mean is not None:
            self._bz = self._bz - self._Az @ self._mean
        if self._covariance_root is not None:
            self._Az = self._Az @ self._covariance_root

    @property
    def lifetime_samples(self) -> int:
        """The total number of samples generated by the sampler during its lifetime."""
        return self._lifetime_samples

    def draw(self, n: int = 1) -> Tuple[Tensor, Tensor]:
        r"""Draw samples.

        Args:
            n: The number of samples.

        Returns:
            A `n x d`-dim tensor of `n` samples.
        """
        samples = []
        for _ in range(n):
            for _ in range(self.thinning):
                self.step()
            samples.append(self.step())
        return torch.cat(samples, dim=-1).transpose(-1, -2)

    def step(self) -> Tensor:
        r"""Take a step, return the new sample, update the internal state.

        Returns:
            A `d x 1`-dim sample from the domain.
        """
        nu = torch.randn_like(self._z)
        theta = self._draw_angle(nu=nu)
        z = self._get_cart_coords(nu=nu, theta=theta)
        self._z[:] = z
        x = self._untransform(z)
        self._x[:] = x
        self._lifetime_samples += 1
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
        return self._z * torch.cos(theta) + nu * torch.sin(theta)

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
        g1 = -self._Az @ self._z
        g2 = -self._Az @ nu
        r = torch.sqrt(g1**2 + g2**2)
        phi = 2 * torch.atan(g2 / (r + g1)).squeeze()

        arg = -(self._bz / r).squeeze()
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
        delta_feasibility = (
            self._is_feasible(samples_mid, transformed=True).to(dtype=int).diff()
        )
        active_indices = delta_feasibility.nonzero()
        return theta[active_indices], delta_feasibility[active_indices]

    def _is_feasible(self, points: Tensor, transformed: bool = False) -> Tensor:
        r"""Returns a Boolean tensor indicating whether the `points` are feasible,
        i.e. they satisfy `A @ points <= b`, where `(A, b)` are the tensors passed
        as the `inequality_constraints` to the constructor of the sampler.

        Args:
            points: A `d x M`-dim tensor of points.
            transformed: Wether points are assumed to be transformed by a change of
                basis, which means feasibility should be computed based on the
                transformed constraint system (_Az, _bz), instead of (A, b).

        Returns:
            An `M`-dim binary tensor where `True` indicates that the associated
            point is feasible.
        """
        A, b = (self._Az, self._bz) if transformed else (self.A, self.b)
        return (A @ points <= b).all(dim=0)

    def _transform(self, x: Tensor) -> Tensor:
        """Transforms the input so that it is equivalent to a standard Normal variable
        constrained with the modified system constraints (self._Az, self._bz).

        Args:
            x: The input tensor to be transformed, usually `d x 1`-dimensional.

        Returns:
            A `d x 1`-dimensional tensor of transformed values subject to the modified
            system of constraints.
        """
        if self._not_fixed is not None:
            x = x[self._not_fixed]
        return self._standardize(x)

    def _untransform(self, z: Tensor) -> Tensor:
        """The inverse transform of the `_transform`, i.e. maps `z` back to the original
        space where it is subject to the original constraint system (self.A, self.b).

        Args:
            z: The transformed tensor to be un-transformed, usually `d x 1`-dimensional.

        Returns:
            A `d x 1`-dimensional tensor of un-transformed values subject to the
            original system of constraints.
        """
        if self._is_fixed is None:
            return self._unstandardize(z)
        else:
            x = self._x.clone()  # _x already contains the fixed values
            x[self._not_fixed] = self._unstandardize(z)
            return x

    def _standardize(self, x: Tensor) -> Tensor:
        """_transform helper standardizing the input `x`, which is assumed to be a
        `d x 1`-dim Tensor, or a `len(self._not_fixed) x 1`-dim if there are fixed
        indices.
        """
        z = x
        if self._mean is not None:
            z = z - self._mean
        if self._covariance_root is not None:
            z = torch.linalg.solve_triangular(self._covariance_root, z, upper=False)
        return z

    def _unstandardize(self, z: Tensor) -> Tensor:
        """_untransform helper un-standardizing the input `z`, which is assumed to be a
        `d x 1`-dim Tensor, or a `len(self._not_fixed) x 1`-dim if there are fixed
        indices.
        """
        x = z
        if self._covariance_root is not None:
            x = self._covariance_root @ x
        if self._mean is not None:
            x = x + self._mean
        return x


def get_index_tensors(
    fixed_indices: Union[List[int], Tensor], d: int
) -> Tuple[Tensor, Tensor]:
    """Converts `fixed_indices` to a `d`-dim integral Tensor that is True at indices
    that are contained in `fixed_indices` and False otherwise.

    Args:
        fixed_indices: A list or Tensoro of integer indices to fix.
        d: The dimensionality of the Tensors to be indexed.

    Returns:
        A Tuple of integral Tensors partitioning [1, d] into indices that are fixed
        (first tensor) and non-fixed (second tensor).
    """
    is_fixed = torch.as_tensor(fixed_indices)
    dtype, device = is_fixed.dtype, is_fixed.device
    dims = torch.arange(d, dtype=dtype, device=device)
    not_fixed = torch.tensor([i for i in dims if i not in is_fixed])
    return is_fixed, not_fixed
