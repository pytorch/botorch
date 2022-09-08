#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Outcome transformations for automatically transforming and un-transforming
model outputs. Outcome transformations are typically part of a Model and
applied (i) within the model constructor to transform the train observations
to the model space, and (ii) in the `Model.posterior` call to untransform
the model posterior back to the original space.

References

.. [eriksson2021scalable]
    D. Eriksson, M. Poloczek. Scalable Constrained Bayesian Optimization.
    International Conference on Artificial Intelligence and Statistics. PMLR, 2021,
    http://proceedings.mlr.press/v130/eriksson21a.html

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
from botorch.models.transforms.utils import (
    norm_to_lognorm_mean,
    norm_to_lognorm_variance,
)
from botorch.posteriors import GPyTorchPosterior, Posterior, TransformedPosterior
from botorch.utils.transforms import normalize_indices
from linear_operator.operators import (
    BlockDiagLinearOperator,
    CholLinearOperator,
    DiagLinearOperator,
)
from torch import Tensor
from torch.nn import Module, ModuleDict


class OutcomeTransform(Module, ABC):
    r"""
    Abstract base class for outcome transforms.

    :meta private:
    """

    @abstractmethod
    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Transform the outcomes in a model's training targets

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        pass  # pragma: no cover

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        This functionality is used to properly treat outcome transformations
        in the `subset_model` functionality.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the "
            "`subset_output` method"
        )

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Un-transform previously transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of transfomred training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of transformed observation
                noises associated with the training targets (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the `untransform` method"
        )

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        r"""Un-transform a posterior

        Args:
            posterior: A posterior in the transformed space.

        Returns:
            The un-transformed posterior.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the "
            "`untransform_posterior` method"
        )


class ChainedOutcomeTransform(OutcomeTransform, ModuleDict):
    r"""An outcome transform representing the chaining of individual transforms"""

    def __init__(self, **transforms: OutcomeTransform) -> None:
        r"""Chaining of outcome transforms.

        Args:
            transforms: The transforms to chain. Internally, the names of the
                kwargs are used as the keys for accessing the individual
                transforms on the module.
        """
        super().__init__(OrderedDict(transforms))

    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Transform the outcomes in a model's training targets

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        for tf in self.values():
            Y, Yvar = tf.forward(Y, Yvar)
        return Y, Yvar

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        return self.__class__(
            **{name: tf.subset_output(idcs=idcs) for name, tf in self.items()}
        )

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Un-transform previously transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of transfomred training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of transformed observation
                noises associated with the training targets (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        for tf in reversed(self.values()):
            Y, Yvar = tf.untransform(Y, Yvar)
        return Y, Yvar

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        r"""Un-transform a posterior

        Args:
            posterior: A posterior in the transformed space.

        Returns:
            The un-transformed posterior.
        """
        for tf in reversed(self.values()):
            posterior = tf.untransform_posterior(posterior)
        return posterior


class Standardize(OutcomeTransform):
    r"""Standardize outcomes (zero mean, unit variance).

    This module is stateful: If in train mode, calling forward updates the
    module state (i.e. the mean/std normalizing constants). If in eval mode,
    calling forward simply applies the standardization using the current module
    state.
    """

    def __init__(
        self,
        m: int,
        outputs: Optional[List[int]] = None,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        min_stdv: float = 1e-8,
    ) -> None:
        r"""Standardize outcomes (zero mean, unit variance).

        Args:
            m: The output dimension.
            outputs: Which of the outputs to standardize. If omitted, all
                outputs will be standardized.
            batch_shape: The batch_shape of the training targets.
            min_stddv: The minimum standard deviation for which to perform
                standardization (if lower, only de-mean the data).
        """
        super().__init__()
        self.register_buffer("means", torch.zeros(*batch_shape, 1, m))
        self.register_buffer("stdvs", torch.zeros(*batch_shape, 1, m))
        self.register_buffer("_stdvs_sq", torch.zeros(*batch_shape, 1, m))
        self._outputs = normalize_indices(outputs, d=m)
        self._m = m
        self._batch_shape = batch_shape
        self._min_stdv = min_stdv

    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Standardize outcomes.

        If the module is in train mode, this updates the module state (i.e. the
        mean/std normalizing constants). If the module is in eval mode, simply
        applies the normalization using the module state.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        if self.training:
            if Y.shape[:-2] != self._batch_shape:
                raise RuntimeError("wrong batch shape")
            if Y.size(-1) != self._m:
                raise RuntimeError("wrong output dimension")
            stdvs = Y.std(dim=-2, keepdim=True)
            stdvs = stdvs.where(stdvs >= self._min_stdv, torch.full_like(stdvs, 1.0))
            means = Y.mean(dim=-2, keepdim=True)
            if self._outputs is not None:
                unused = [i for i in range(self._m) if i not in self._outputs]
                means[..., unused] = 0.0
                stdvs[..., unused] = 1.0
            self.means = means
            self.stdvs = stdvs
            self._stdvs_sq = stdvs.pow(2)

        Y_tf = (Y - self.means) / self.stdvs
        Yvar_tf = Yvar / self._stdvs_sq if Yvar is not None else None
        return Y_tf, Yvar_tf

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        new_m = len(idcs)
        if new_m > self._m:
            raise RuntimeError(
                "Trying to subset a transform have more outputs than "
                " the original transform."
            )
        nlzd_idcs = normalize_indices(idcs, d=self._m)
        new_outputs = None
        if self._outputs is not None:
            new_outputs = [i for i in self._outputs if i in nlzd_idcs]
        new_tf = self.__class__(
            m=new_m,
            outputs=new_outputs,
            batch_shape=self._batch_shape,
            min_stdv=self._min_stdv,
        )
        new_tf.means = self.means[..., nlzd_idcs]
        new_tf.stdvs = self.stdvs[..., nlzd_idcs]
        new_tf._stdvs_sq = self._stdvs_sq[..., nlzd_idcs]
        if not self.training:
            new_tf.eval()
        return new_tf

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Un-standardize outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of standardized targets.
            Yvar: A `batch_shape x n x m`-dim tensor of standardized observation
                noises associated with the targets (if applicable).

        Returns:
            A two-tuple with the un-standardized outcomes:

            - The un-standardized outcome observations.
            - The un-standardized observation noise (if applicable).
        """
        Y_utf = self.means + self.stdvs * Y
        Yvar_utf = self._stdvs_sq * Yvar if Yvar is not None else None
        return Y_utf, Yvar_utf

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        r"""Un-standardize the posterior.

        Args:
            posterior: A posterior in the standardized space.

        Returns:
            The un-standardized posterior. If the input posterior is a MVN,
            the transformed posterior is again an MVN.
        """
        if self._outputs is not None:
            raise NotImplementedError(
                "Standardize does not yet support output selection for "
                "untransform_posterior"
            )
        is_mtgp_posterior = False
        if type(posterior) is GPyTorchPosterior:
            is_mtgp_posterior = posterior._is_mt
        if not self._m == posterior.event_shape[-1] and not is_mtgp_posterior:
            raise RuntimeError(
                "Incompatible output dimensions encountered for transform "
                f"{self._m} and posterior {posterior.event_shape[-1]}."
            )

        if type(posterior) is not GPyTorchPosterior:
            # fall back to TransformedPosterior
            # this applies to subclasses of GPyTorchPosterior like MultitaskGPPosterior
            return TransformedPosterior(
                posterior=posterior,
                sample_transform=lambda s: self.means + self.stdvs * s,
                mean_transform=lambda m, v: self.means + self.stdvs * m,
                variance_transform=lambda m, v: self._stdvs_sq * v,
            )
        # GPyTorchPosterior (TODO: Should we Lazy-evaluate the mean here as well?)
        mvn = posterior.mvn
        offset = self.means
        scale_fac = self.stdvs
        if not posterior._is_mt:
            mean_tf = offset.squeeze(-1) + scale_fac.squeeze(-1) * mvn.mean
            scale_fac = scale_fac.squeeze(-1).expand_as(mean_tf)
        else:
            mean_tf = offset + scale_fac * mvn.mean
            reps = mean_tf.shape[-2:].numel() // scale_fac.size(-1)
            scale_fac = scale_fac.squeeze(-2)
            if mvn._interleaved:
                scale_fac = scale_fac.repeat(*[1 for _ in scale_fac.shape[:-1]], reps)
            else:
                scale_fac = torch.repeat_interleave(scale_fac, reps, dim=-1)

        if (
            not mvn.islazy
            # TODO: Figure out attribute namming weirdness here
            or mvn._MultivariateNormal__unbroadcasted_scale_tril is not None
        ):
            # if already computed, we can save a lot of time using scale_tril
            covar_tf = CholLinearOperator(mvn.scale_tril * scale_fac.unsqueeze(-1))
        else:
            lcv = mvn.lazy_covariance_matrix
            scale_fac = scale_fac.expand(lcv.shape[:-1])
            # TODO: Remove the custom logic with next GPyTorch release (T126095032).
            if isinstance(lcv, BlockDiagLinearOperator):
                # Keep the block diag structure of lcv.
                base_lcv = lcv.base_linear_op
                scale_mat = DiagLinearOperator(
                    scale_fac.view(*scale_fac.shape[:-1], lcv.num_blocks, -1)
                )
                base_lcv_tf = scale_mat @ base_lcv @ scale_mat
                covar_tf = BlockDiagLinearOperator(base_linear_op=base_lcv_tf)
            else:
                # allow batch-evaluation of the model
                scale_mat = DiagLinearOperator(scale_fac)
                covar_tf = scale_mat @ lcv @ scale_mat

        kwargs = {"interleaved": mvn._interleaved} if posterior._is_mt else {}
        mvn_tf = mvn.__class__(mean=mean_tf, covariance_matrix=covar_tf, **kwargs)
        return GPyTorchPosterior(mvn_tf)


class Log(OutcomeTransform):
    r"""Log-transform outcomes.

    Useful if the targets are modeled using a (multivariate) log-Normal
    distribution. This means that we can use a standard GP model on the
    log-transformed outcomes and un-transform the model posterior of that GP.
    """

    def __init__(self, outputs: Optional[List[int]] = None) -> None:
        r"""Log-transform outcomes.

        Args:
            outputs: Which of the outputs to log-transform. If omitted, all
                outputs will be standardized.
        """
        super().__init__()
        self._outputs = outputs

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        new_outputs = None
        if self._outputs is not None:
            if min(self._outputs + idcs) < 0:
                raise NotImplementedError(
                    f"Negative indexing not supported for {self.__class__.__name__} "
                    "when subsetting outputs and only transforming some outputs."
                )
            new_outputs = [i for i in self._outputs if i in idcs]
        new_tf = self.__class__(outputs=new_outputs)
        if not self.training:
            new_tf.eval()
        return new_tf

    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Log-transform outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        Y_tf = torch.log(Y)
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_tf = torch.stack(
                [
                    Y_tf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            # TODO: Delta method, possibly issue warning
            raise NotImplementedError(
                "Log does not yet support transforming observation noise"
            )
        return Y_tf, Yvar

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Un-transform log-transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of log-transfomred targets.
            Yvar: A `batch_shape x n x m`-dim tensor of log- transformed
                observation noises associated with the training targets
                (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The exponentiated outcome observations.
            - The exponentiated observation noise (if applicable).
        """
        Y_utf = torch.exp(Y)
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_utf = torch.stack(
                [
                    Y_utf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            # TODO: Delta method, possibly issue warning
            raise NotImplementedError(
                "Log does not yet support transforming observation noise"
            )
        return Y_utf, Yvar

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        r"""Un-transform the log-transformed posterior.

        Args:
            posterior: A posterior in the log-transformed space.

        Returns:
            The un-transformed posterior.
        """
        if self._outputs is not None:
            raise NotImplementedError(
                "Log does not yet support output selection for untransform_posterior"
            )
        return TransformedPosterior(
            posterior=posterior,
            sample_transform=torch.exp,
            mean_transform=norm_to_lognorm_mean,
            variance_transform=norm_to_lognorm_variance,
        )


class Power(OutcomeTransform):
    r"""Power-transform outcomes.

    Useful if the targets are modeled using a (multivariate) power transform of
    a Normal distribution. This means that we can use a standard GP model on the
    power-transformed outcomes and un-transform the model posterior of that GP.
    """

    def __init__(self, power: float, outputs: Optional[List[int]] = None) -> None:
        r"""Power-transform outcomes.

        Args:
            outputs: Which of the outputs to power-transform. If omitted, all
                outputs will be standardized.
        """
        super().__init__()
        self._outputs = outputs
        self.power = power

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        new_outputs = None
        if self._outputs is not None:
            if min(self._outputs + idcs) < 0:
                raise NotImplementedError(
                    f"Negative indexing not supported for {self.__class__.__name__} "
                    "when subsetting outputs and only transforming some outputs."
                )
            new_outputs = [i for i in self._outputs if i in idcs]
        new_tf = self.__class__(power=self.power, outputs=new_outputs)
        if not self.training:
            new_tf.eval()
        return new_tf

    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Power-transform outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        Y_tf = Y.pow(self.power)
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_tf = torch.stack(
                [
                    Y_tf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            # TODO: Delta method, possibly issue warning
            raise NotImplementedError(
                "Power does not yet support transforming observation noise"
            )
        return Y_tf, Yvar

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Un-transform power-transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of power-transfomred targets.
            Yvar: A `batch_shape x n x m`-dim tensor of power-transformed
                observation noises associated with the training targets
                (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The un-power transformed outcome observations.
            - The un-power transformed observation noise (if applicable).
        """
        Y_utf = Y.pow(1.0 / self.power)
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_utf = torch.stack(
                [
                    Y_utf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            # TODO: Delta method, possibly issue warning
            raise NotImplementedError(
                "Power does not yet support transforming observation noise"
            )
        return Y_utf, Yvar

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        r"""Un-transform the power-transformed posterior.

        Args:
            posterior: A posterior in the power-transformed space.

        Returns:
            The un-transformed posterior.
        """
        if self._outputs is not None:
            raise NotImplementedError(
                "Power does not yet support output selection for untransform_posterior"
            )
        return TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda x: x.pow(1.0 / self.power),
        )


class Bilog(OutcomeTransform):
    r"""Bilog-transform outcomes.

    The Bilog transform [eriksson2021scalable]_ is useful for modeling outcome
    constraints as it magnifies values near zero and flattens extreme values.
    """

    def __init__(self, outputs: Optional[List[int]] = None) -> None:
        r"""Bilog-transform outcomes.

        Args:
            outputs: Which of the outputs to Bilog-transform. If omitted, all
                outputs will be transformed.
        """
        super().__init__()
        self._outputs = outputs

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        new_outputs = None
        if self._outputs is not None:
            if min(self._outputs + idcs) < 0:
                raise NotImplementedError(
                    f"Negative indexing not supported for {self.__class__.__name__} "
                    "when subsetting outputs and only transforming some outputs."
                )
            new_outputs = [i for i in self._outputs if i in idcs]
        new_tf = self.__class__(outputs=new_outputs)
        if not self.training:
            new_tf.eval()
        return new_tf

    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Bilog-transform outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        Y_tf = Y.sign() * (Y.abs() + 1.0).log()
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_tf = torch.stack(
                [
                    Y_tf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            raise NotImplementedError(
                "Bilog does not yet support transforming observation noise"
            )
        return Y_tf, Yvar

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Un-transform bilog-transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of bilog-transfomred targets.
            Yvar: A `batch_shape x n x m`-dim tensor of bilog-transformed
                observation noises associated with the training targets
                (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        Y_utf = Y.sign() * (Y.abs().exp() - 1.0)
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_utf = torch.stack(
                [
                    Y_utf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            # TODO: Delta method, possibly issue warning
            raise NotImplementedError(
                "Bilog does not yet support transforming observation noise"
            )
        return Y_utf, Yvar

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        r"""Un-transform the bilog-transformed posterior.

        Args:
            posterior: A posterior in the bilog-transformed space.

        Returns:
            The un-transformed posterior.
        """
        if self._outputs is not None:
            raise NotImplementedError(
                "Bilog does not yet support output selection for untransform_posterior"
            )
        return TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda x: x.sign() * (x.abs().exp() - 1.0),
        )
