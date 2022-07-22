#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.kernels.contextual_lcea import LCEAKernel
from botorch.models.kernels.contextual_sac import SACKernel
from torch import Tensor


class SACGP(FixedNoiseGP):
    r"""A GP using a Structural Additive Contextual(SAC) kernel."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        decomposition: Dict[str, List[int]],
    ) -> None:
        r"""
        Args:
            train_X: (n x d) X training data.
            train_Y: (n x 1) Y training data.
            train_Yvar: (n x 1) Noise variances of each training Y.
            decomposition: Keys are context names. Values are the indexes of
                parameters belong to the context. The parameter indexes are in
                the same order across contexts.
        """
        super().__init__(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self.covar_module = SACKernel(
            decomposition=decomposition,
            batch_shape=self._aug_batch_shape,
            device=train_X.device,
        )
        self.decomposition = decomposition
        self.to(train_X)


class LCEAGP(FixedNoiseGP):
    r"""A GP using a Latent Context Embedding Additive (LCE-A) Kernel.

    Note that the model does not support batch training. Input training
    data sets should have dim = 2.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        decomposition: Dict[str, List[int]],
        train_embedding: bool = True,
        cat_feature_dict: Optional[Dict] = None,
        embs_feature_dict: Optional[Dict] = None,
        embs_dim_list: Optional[List[int]] = None,
        context_weight_dict: Optional[Dict] = None,
    ) -> None:
        r"""
        Args:
            train_X: (n x d) X training data.
            train_Y: (n x 1) Y training data.
            train_Yvar: (n x 1) Noise variance of Y.
            decomposition: Keys are context names. Values are the indexes of
                parameters belong to the context. The parameter indexes are in the
                same order across contexts.
            cat_feature_dict: Keys are context names and values are list of categorical
                features i.e. {"context_name" : [cat_0, ..., cat_k]}, where k is the
                number of categorical variables. If None, we use context names in the
                decomposition as the only categorical feature, i.e., k = 1.
            embs_feature_dict: Pre-trained continuous embedding features of each
                context.
            embs_dim_list: Embedding dimension for each categorical variable. The length
                equals the number of categorical features k. If None, the embedding
                dimension is set to 1 for each categorical variable.
            context_weight_dict: Known population weights of each context.
        """
        super().__init__(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self.covar_module = LCEAKernel(
            decomposition=decomposition,
            batch_shape=self._aug_batch_shape,
            train_embedding=train_embedding,
            cat_feature_dict=cat_feature_dict,
            embs_feature_dict=embs_feature_dict,
            embs_dim_list=embs_dim_list,
            context_weight_dict=context_weight_dict,
            device=train_X.device,
        )
        self.decomposition = decomposition
        self.to(train_X)
