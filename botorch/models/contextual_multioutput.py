#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.constraints import Interval
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from linear_operator.operators import InterpolatedLinearOperator, LinearOperator
from torch import Tensor
from torch.nn import ModuleList


class LCEMGP(MultiTaskGP):
    r"""The Multi-Task GP with the latent context embedding multioutput
    (LCE-M) kernel.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        context_cat_feature: Optional[Tensor] = None,
        context_emb_feature: Optional[Tensor] = None,
        embs_dim_list: Optional[List[int]] = None,
        output_tasks: Optional[List[int]] = None,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
    ) -> None:
        r"""
        Args:
            train_X: (n x d) X training data.
            train_Y: (n x 1) Y training data.
            task_feature: Column index of train_X to get context indices.
            context_cat_feature: (n_contexts x k) one-hot encoded context
                features. Rows are ordered by context indices, where k is the
                number of categorical variables. If None, task indices will
                be used and k = 1.
            context_emb_feature: (n_contexts x m) pre-given continuous
                embedding features. Rows are ordered by context indices.
            embs_dim_list: Embedding dimension for each categorical variable.
                The length equals k. If None, the embedding dimension is set to 1
                for each categorical variable.
            output_tasks: A list of task indices for which to compute model
                outputs for. If omitted, return outputs for all task indices.
        """
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            output_tasks=output_tasks,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        self.device = train_X.device
        #  context indices
        all_tasks = train_X[:, task_feature].unique()
        self.all_tasks = all_tasks.to(dtype=torch.long).tolist()
        self.all_tasks.sort()  # unique in python does automatic sort; add for safety

        if context_cat_feature is None:
            context_cat_feature = all_tasks.unsqueeze(-1).to(device=self.device)
        self.context_cat_feature = context_cat_feature  # row indices = context indices
        self.context_emb_feature = context_emb_feature

        #  construct emb_dims based on categorical features
        if embs_dim_list is None:
            #  set embedding_dim = 1 for each categorical variable
            embs_dim_list = [1 for _i in range(context_cat_feature.size(1))]
        n_embs = sum(embs_dim_list)
        self.emb_dims = [
            (len(context_cat_feature[:, i].unique()), embs_dim_list[i])
            for i in range(context_cat_feature.size(1))
        ]
        # contruct embedding layer: need to handle multiple categorical features
        self.emb_layers = ModuleList(
            [
                torch.nn.Embedding(num_embeddings=x, embedding_dim=y, max_norm=1.0)
                for x, y in self.emb_dims
            ]
        )
        self.task_covar_module = RBFKernel(
            ard_num_dims=n_embs,
            lengthscale_constraint=Interval(
                0.0, 2.0, transform=None, initial_value=1.0
            ),
        )
        self.to(train_X)

    def _eval_context_covar(self) -> LinearOperator:
        """obtain context covariance matrix (num_contexts x num_contexts)"""
        all_embs = self._task_embeddings()
        return self.task_covar_module(all_embs)

    def _task_embeddings(self) -> Tensor:
        """generate embedding features for all contexts."""
        embeddings = [
            emb_layer(
                self.context_cat_feature[:, i].to(
                    dtype=torch.long, device=self.device
                )  # pyre-ignore
            )
            for i, emb_layer in enumerate(self.emb_layers)
        ]
        embeddings = torch.cat(embeddings, dim=1)

        # add given embeddings if any
        if self.context_emb_feature is not None:
            embeddings = torch.cat(
                [embeddings, self.context_emb_feature.to(self.device)],
                dim=1,  # pyre-ignore
            )
        return embeddings

    def task_covar_matrix(self, task_idcs: Tensor) -> Tensor:
        r"""compute covariance matrix of a list of given context

        Args:
            task_idcs: (n x 1) or (b x n x 1) task indices tensor
        """
        covar_matrix = self._eval_context_covar()
        return InterpolatedLinearOperator(
            base_linear_op=covar_matrix,
            left_interp_indices=task_idcs,
            right_interp_indices=task_idcs,
        ).to_dense()

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        x_basic, task_idcs = self._split_inputs(x)
        # Compute base mean and covariance
        mean_x = self.mean_module(x_basic)
        covar_x = self.covar_module(x_basic)
        # Compute task covariances
        covar_i = self.task_covar_matrix(task_idcs)
        covar = covar_x.mul(covar_i)
        return MultivariateNormal(mean_x, covar)


class FixedNoiseLCEMGP(LCEMGP):
    r"""The Multi-Task GP the latent context embedding multioutput
    (LCE-M) kernel, with known observation noise.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        task_feature: int,
        context_cat_feature: Optional[Tensor] = None,
        context_emb_feature: Optional[Tensor] = None,
        embs_dim_list: Optional[List[int]] = None,
        output_tasks: Optional[List[int]] = None,
    ) -> None:
        r"""
        Args:
            train_X: (n x d) X training data.
            train_Y: (n x 1) Y training data.
            train_Yvar: (n x 1) Noise variances of each training Y.
            task_feature: Column index of train_X to get context indices.
            context_cat_feature: (n_contexts x k) one-hot encoded context
                features. Rows are ordered by context indices, where k is the
                number of categorical variables. If None, task indices will
                be used and k = 1.
            context_emb_feature: (n_contexts x m) pre-given continuous
                embedding features. Rows are ordered by context indices.
            embs_dim_list: Embedding dimension for each categorical variable.
                The length equals to k. If None, the embedding dimension is set to
                1 for each categorical variable.
            output_tasks: A list of task indices for which to compute model
                outputs for. If omitted, return outputs for all task indices.
        """
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            context_cat_feature=context_cat_feature,
            context_emb_feature=context_emb_feature,
            embs_dim_list=embs_dim_list,
            output_tasks=output_tasks,
        )
        self.likelihood = FixedNoiseGaussianLikelihood(noise=train_Yvar)
        self.to(train_X)
