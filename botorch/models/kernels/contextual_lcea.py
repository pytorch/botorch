#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import torch
from gpytorch.constraints import Positive
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.lazy.sum_lazy_tensor import SumLazyTensor
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor
from torch.nn import ModuleList


class LCEAKernel(Kernel):
    r"""The Latent Context Embedding Additive (LCE-A) Kernel.

    This kernel is similar to the SACKernel, and is used when context breakdowns are
    unbserverable. It assumes the same additive structure and a spatial kernel shared
    across contexts. Rather than assuming independence, LCEAKernel models the
    correlation in the latent functions for each context through learning context
    embeddings.

    Args:
        decomposition: Keys index context names. Values are the indexes of parameters
            belong to the context. The parameter indexes are in the same order across
            contexts.
        batch_shape: Batch shape as usual for gpytorch kernels. Model does not support
            batch training. When batch_shape is non-empty, it is used for loading
            hyper-parameter values generated from MCMC sampling.
        train_embedding: A boolean indictor of whether to learn context embeddings
        cat_feature_dict: Keys are context names and values are list of categorical
            features i.e. {"context_name" : [cat_0, ..., cat_k]}. k equals to number
            of categorical variables. If None, we use context names in the
            decomposition as the only categorical feature i.e. k = 1
        embs_feature_dict: Pre-trained continuous embedding features of each context.
        embs_dim_list: Embedding dimension for each categorical variable. The length
            equals to num of categorical features k. If None, emb dim is set to 1
            for each categorical variable.
        context_weight_dict: Known population Weights of each context.
    """

    def __init__(
        self,
        decomposition: Dict[str, List[int]],
        batch_shape: torch.Size,
        train_embedding: bool = True,
        cat_feature_dict: Optional[Dict] = None,
        embs_feature_dict: Optional[Dict] = None,
        embs_dim_list: Optional[List[int]] = None,
        context_weight_dict: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ) -> None:

        super().__init__(batch_shape=batch_shape)
        self.decomposition = decomposition
        self.batch_shape = batch_shape
        self.train_embedding = train_embedding
        self.device = device

        num_param = len(next(iter(decomposition.values())))
        self.context_list = list(decomposition.keys())
        self.num_contexts = len(self.context_list)

        # get parameter space decomposition
        for active_parameters in decomposition.values():
            # check number of parameters are same in each decomp
            if len(active_parameters) != num_param:
                raise ValueError(
                    "num of parameters needs to be same across all contexts"
                )
        self._indexers = {
            context: torch.tensor(active_params, device=self.device)
            for context, active_params in self.decomposition.items()
        }
        # get context features and set emb dim
        self.context_cat_feature = None
        self.context_emb_feature = None
        self.n_embs = 0
        self.emb_weight_matrix_list = None
        self.emb_dims = None
        self._set_context_features(
            cat_feature_dict=cat_feature_dict,
            embs_feature_dict=embs_feature_dict,
            embs_dim_list=embs_dim_list,
        )
        # contruct embedding layer
        if train_embedding:
            self._set_emb_layers()
        # task covariance matrix
        self.task_covar_module = MaternKernel(
            nu=2.5,
            ard_num_dims=self.n_embs,
            batch_shape=batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )
        # base kernel
        self.base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=num_param,
            batch_shape=batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )
        # outputscales for each context (note this is like sqrt of outputscale)
        self.context_weight = None
        if context_weight_dict is None:
            outputscale_list = torch.zeros(
                *batch_shape, self.num_contexts, device=self.device
            )
        else:
            outputscale_list = torch.zeros(*batch_shape, 1, device=self.device)
            self.context_weight = torch.tensor(
                [context_weight_dict[c] for c in self.context_list], device=self.device
            )
        self.register_parameter(
            name="raw_outputscale_list", parameter=torch.nn.Parameter(outputscale_list)
        )
        self.register_prior(
            "outputscale_list_prior",
            GammaPrior(2.0, 15.0),
            lambda m: m.outputscale_list,
            lambda m, v: m._set_outputscale_list(v),
        )
        self.register_constraint("raw_outputscale_list", Positive())

    @property
    def outputscale_list(self) -> Tensor:
        return self.raw_outputscale_list_constraint.transform(self.raw_outputscale_list)

    @outputscale_list.setter
    def outputscale_list(self, value: Tensor) -> None:
        self._set_outputscale_list(value)

    def _set_outputscale_list(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale_list)
        self.initialize(
            raw_outputscale_list=self.raw_outputscale_list_constraint.inverse_transform(
                value
            )
        )

    def _set_context_features(
        self,
        cat_feature_dict: Optional[Dict] = None,
        embs_feature_dict: Optional[Dict] = None,
        embs_dim_list: Optional[List[int]] = None,
    ) -> None:
        """Set context categorical features and continuous embedding features.
        If cat_feature_dict is None, context indices will be used; If embs_dim_list
        is None, we use 1-d embedding for each categorical features.
        """
        # get context categorical features
        if cat_feature_dict is None:
            self.context_cat_feature = torch.arange(
                self.num_contexts, device=self.device
            ).unsqueeze(-1)
        else:
            self.context_cat_feature = torch.tensor(
                [cat_feature_dict[c] for c in self.context_list]
            )
        #  construct emb_dims based on categorical features
        if embs_dim_list is None:
            #  set embedding_dim = 1 for each categorical variable
            embs_dim_list = [1 for _i in range(self.context_cat_feature.size(1))]
        self.emb_dims = [
            (len(self.context_cat_feature[:, i].unique()), embs_dim_list[i])
            for i in range(self.context_cat_feature.size(1))
        ]
        if self.train_embedding:
            self.n_embs = sum(embs_dim_list)  # total num of emb features
        # get context embedding features
        if embs_feature_dict is not None:
            self.context_emb_feature = torch.tensor(
                [embs_feature_dict[c] for c in self.context_list], device=self.device
            )
            self.n_embs += self.context_emb_feature.size(1)

    def _set_emb_layers(self) -> None:
        """Construct embedding layers.
        If model is non-batch, we use nn.Embedding to learn emb weights. If model is
        batched (sef.batch_shape is non-empty), we load emb weights posterior samples
        and construct a parameter list that each parameter is the emb weight of each
        layer. The shape of weight matrices are ns x num_contexts x emb_dim.
        """
        self.emb_layers = ModuleList(
            [
                torch.nn.Embedding(num_embeddings=x, embedding_dim=y, max_norm=1.0)
                for x, y in self.emb_dims
            ]
        )
        # use posterior of emb weights
        if len(self.batch_shape) > 0:
            self.emb_weight_matrix_list = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.zeros(
                            self.batch_shape + emb_layer.weight.shape,
                            device=self.device,
                        )
                    )
                    for emb_layer in self.emb_layers
                ]
            )

    def _eval_context_covar(self) -> Tensor:
        """Compute context covariance matrix.

        Returns:
            A (ns) x num_contexts x num_contexts tensor.
        """
        if len(self.batch_shape) > 0:
            # broadcast - (ns x num_contexts x k)
            all_embs = self._task_embeddings_batch()
        else:
            all_embs = self._task_embeddings()  # no broadcast - (num_contexts x k)

        context_covar = self.task_covar_module(all_embs).evaluate()
        if self.context_weight is None:
            context_outputscales = self.outputscale_list
        else:
            context_outputscales = self.outputscale_list * self.context_weight
        context_covar = (
            (context_outputscales.unsqueeze(-2))  # (ns) x 1 x num_contexts
            .mul(context_covar)
            .mul(context_outputscales.unsqueeze(-1))  # (ns) x num_contexts x 1
        )
        return context_covar

    def _task_embeddings(self) -> Tensor:
        """Generate embedding features of contexts when model is non-batch.

        Returns:
            a (num_contexts x n_embs) tensor. n_embs is the sum of embedding
            dimensions i.e. sum(embs_dim_list)
        """
        if self.train_embedding is False:
            return self.context_emb_feature  # use pre-trained embedding only
        context_features = torch.stack(
            [self.context_cat_feature[i, :] for i in range(self.num_contexts)], dim=0
        )
        embeddings = [
            emb_layer(context_features[:, i].to(device=self.device, dtype=torch.long))
            for i, emb_layer in enumerate(self.emb_layers)
        ]
        embeddings = torch.cat(embeddings, dim=1)
        # add given embeddings if any
        if self.context_emb_feature is not None:
            embeddings = torch.cat([embeddings, self.context_emb_feature], dim=1)
        return embeddings

    def _task_embeddings_batch(self) -> Tensor:
        """Generate embedding features of contexts when model has multiple batches.

        Returns:
            a (ns) x num_contexts x n_embs tensor. ns is the batch size i.e num of
            posterior samples; n_embs is the sum of embedding dimensions i.e.
            sum(embs_dim_list).
        """
        context_features = torch.cat(
            [
                self.context_cat_feature[i, :].unsqueeze(0)
                for i in range(self.num_contexts)
            ]
        )
        embeddings = []
        for b in range(self.batch_shape.numel()):  # pyre-ignore
            for i in range(len(self.emb_weight_matrix_list)):
                # loop over emb layer and concat embs from each layer
                embeddings.append(
                    torch.cat(
                        [
                            torch.nn.functional.embedding(
                                context_features[:, 0].to(
                                    dtype=torch.long, device=self.device
                                ),
                                self.emb_weight_matrix_list[i][b, :],
                            ).unsqueeze(0)
                        ],
                        dim=1,
                    )
                )
        embeddings = torch.cat(embeddings, dim=0)
        # add given embeddings if any
        if self.context_emb_feature is not None:
            embeddings = torch.cat(
                [
                    embeddings,
                    self.context_emb_feature.expand(
                        *self.batch_shape + self.context_emb_feature.shape
                    ),
                ],
                dim=-1,
            )
        return embeddings

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        """Iterate across each partition of parameter space and sum the
        covariance matrices together
        """
        # context covar matrix
        context_covar = self._eval_context_covar()
        # check input batch size if b x ns x n x d: expand context_covar to
        # b x ns x num_context x num_context
        if x1.dim() > context_covar.dim():
            context_covar = context_covar.expand(*x1.shape[:1] + context_covar.shape)
        covars = []
        # TODO: speed computation of covariance matrix
        for i in range(self.num_contexts):
            for j in range(self.num_contexts):
                context1 = self.context_list[i]
                context2 = self.context_list[j]
                active_params1 = self._indexers[context1]
                active_params2 = self._indexers[context2]
                covars.append(
                    (
                        context_covar.index_select(  # pyre-ignore
                            -1, torch.tensor([j], device=self.device)
                        ).index_select(
                            -2, torch.tensor([i], device=self.device)
                        )  # b x ns x 1 x 1
                    )
                    * self.base_kernel(
                        x1=x1.index_select(-1, active_params1),
                        x2=x2.index_select(-1, active_params2),
                        diag=diag,
                    )
                )
        if diag:
            res = sum(covars)
        else:
            res = SumLazyTensor(*covars)
        return res
