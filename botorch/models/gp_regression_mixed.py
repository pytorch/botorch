#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable
from typing import Dict, List, Optional, Any

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.containers import TrainingData
from botorch.utils.transforms import normalize_indices
from gpytorch.constraints import GreaterThan
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.priors import GammaPrior
from torch import Tensor


class MixedSingleTaskGP(SingleTaskGP):
    r"""A single-task exact GP model for mixed search spaces.

    This model uses a kernel that combines a CategoricalKernel (based on
    Hamming distances) and a regular kernel into a kernel of the form

        K((x1, c1), (x2, c2)) =
            K_cont_1(x1, x2) + K_cat_1(c1, c2) +
            K_cont_2(x1, x2) * K_cat_2(c1, c2)

    where `xi` and `ci` are the continuous and categorical features of the
    input, respectively. The suffix `_i` indicates that we fit different
    lengthscales for the kernels in the sum and product terms.

    Since this model does not provide gradients for the categorical features,
    optimization of the acquisition function will need to be performed in
    a mixed fashion, i.e., treating the categorical features properly as
    discrete optimization variables.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: List[int],
        cont_kernel_factory: Optional[Callable[[int, List[int]], Kernel]] = None,
        likelihood: Optional[Likelihood] = None,
        outcome_transform: Optional[OutcomeTransform] = None,  # TODO
        input_transform: Optional[InputTransform] = None,  # TODO
    ) -> None:
        r"""A single-task exact GP model supporting categorical parameters.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            cat_dims: A list of indices corresponding to the columns of
                the input `X` that should be considered categorical features.
            cont_kernel_factory: A method that accepts `ard_num_dims` and
                `active_dims` arguments and returns an instatiated GPyTorch
                `Kernel` object to be used as the ase kernel for the continuous
                dimensions. If omitted, this model uses a Matern-2.5 kernel as
                the kernel for the ordinal parameters.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
            # outcome_transform: An outcome transform that is applied to the
            #     training data during instantiation and to the posterior during
            #     inference (that is, the `Posterior` obtained by calling
            #     `.posterior` on the model will be on the original scale).
            # input_transform: An input transform that is applied in the model's
            #     forward pass.

        Example:
            >>> train_X = torch.cat(
                    [torch.rand(20, 2), torch.randint(3, (20, 1))], dim=-1)
                )
            >>> train_Y = (
                    torch.sin(train_X[..., :-1]).sum(dim=1, keepdim=True)
                    + train_X[..., -1:]
                )
            >>> model = MixedSingleTaskGP(train_X, train_Y, cat_dims=[-1])
        """
        if outcome_transform is not None:
            raise UnsupportedError("outcome transforms not yet supported")
        if input_transform is not None:
            raise UnsupportedError("input transforms not yet supported")
        if len(cat_dims) == 0:
            raise ValueError(
                "Must specify categorical dimensions for MixedSingleTaskGP"
            )
        input_batch_shape, aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )

        if cont_kernel_factory is None:

            def cont_kernel_factory(
                batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
            ) -> MaternKernel:
                return MaternKernel(
                    nu=2.5,
                    batch_shape=batch_shape,
                    ard_num_dims=ard_num_dims,
                    active_dims=active_dims,
                    lengthscale_constraint=GreaterThan(1e-04),
                )

        if likelihood is None:
            # This Gamma prior is quite close to the Horseshoe prior
            min_noise = 1e-5 if train_X.dtype == torch.float else 1e-6
            likelihood = GaussianLikelihood(
                batch_shape=aug_batch_shape,
                noise_constraint=GreaterThan(
                    min_noise, transform=None, initial_value=1e-3
                ),
                noise_prior=GammaPrior(0.9, 10.0),
            )

        d = train_X.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))
        if len(ord_dims) == 0:
            covar_module = ScaleKernel(
                CategoricalKernel(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(cat_dims),
                    lengthscale_constraint=GreaterThan(1e-06),
                )
            )
        else:
            sum_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
                + ScaleKernel(
                    CategoricalKernel(
                        batch_shape=aug_batch_shape,
                        ard_num_dims=len(cat_dims),
                        active_dims=cat_dims,
                        lengthscale_constraint=GreaterThan(1e-06),
                    )
                )
            )
            prod_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
                * CategoricalKernel(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(cat_dims),
                    active_dims=cat_dims,
                    lengthscale_constraint=GreaterThan(1e-06),
                )
            )
            covar_module = sum_kernel + prod_kernel
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

    @classmethod
    def construct_inputs(
        cls, training_data: TrainingData, **kwargs: Any
    ) -> Dict[str, Any]:
        r"""Construct kwargs for the `Model` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "cat_dims": kwargs["categorical_features"],
            "likelihood": kwargs.get("likelihood"),
        }
