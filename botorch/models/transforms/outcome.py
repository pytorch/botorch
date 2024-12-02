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
from itertools import product

import torch
from botorch.models.transforms.utils import (
    norm_to_lognorm_mean,
    norm_to_lognorm_variance,
)
from botorch.posteriors import GPyTorchPosterior, Posterior, TransformedPosterior
from botorch.utils.transforms import normalize_indices
from linear_operator.operators import CholLinearOperator, DiagLinearOperator
from torch import Tensor
from torch.nn import Module, ModuleDict


class OutcomeTransform(Module, ABC):
    """Abstract base class for outcome transforms."""

    @abstractmethod
    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
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

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
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
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
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

    @property
    def _is_linear(self) -> bool:
        """
        True for transformations such as `Standardize`; these should be able to apply
        `untransform_posterior` to a GPyTorchPosterior and return a GPyTorchPosterior,
        because a multivariate normal distribution should remain multivariate normal
        after applying the transform.
        """
        return False

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        r"""Un-transform a posterior.

        Posteriors with `_is_linear=True` should return a `GPyTorchPosterior` when
        `posterior` is a `GPyTorchPosterior`. Posteriors with `_is_linear=False`
        likely return a `TransformedPosterior` instead.

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
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
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

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
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
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
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

    @property
    def _is_linear(self) -> bool:
        """
        A `ChainedOutcomeTransform` is linear only if all of the component transforms
        are linear.
        """
        return all(octf._is_linear for octf in self.values())

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
        outputs: list[int] | None = None,
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
        self.register_buffer("stdvs", torch.ones(*batch_shape, 1, m))
        self.register_buffer("_stdvs_sq", torch.ones(*batch_shape, 1, m))
        self.register_buffer("_is_trained", torch.tensor(False))
        self._outputs = normalize_indices(outputs, d=m)
        self._m = m
        self._batch_shape = batch_shape
        self._min_stdv = min_stdv

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
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
                raise RuntimeError(
                    f"Expected Y.shape[:-2] to be {self._batch_shape}, matching "
                    "the `batch_shape` argument to `Standardize`, but got "
                    f"Y.shape[:-2]={Y.shape[:-2]}."
                )

            if Y.size(-1) != self._m:
                raise RuntimeError(
                    f"Wrong output dimension. Y.size(-1) is {Y.size(-1)}; expected "
                    f"{self._m}."
                )

            if Y.shape[-2] < 1:
                raise ValueError(f"Can't standardize with no observations. {Y.shape=}.")
            elif Y.shape[-2] == 1:
                stdvs = torch.ones(
                    (*Y.shape[:-2], 1, Y.shape[-1]), dtype=Y.dtype, device=Y.device
                )
            else:
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
            self._is_trained = torch.tensor(True)

        Y_tf = (Y - self.means) / self.stdvs
        Yvar_tf = Yvar / self._stdvs_sq if Yvar is not None else None
        return Y_tf, Yvar_tf

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
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
        new_tf._is_trained = self._is_trained
        if not self.training:
            new_tf.eval()
        return new_tf

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
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
        if not self._is_trained:
            raise RuntimeError(
                "`Standardize` transforms must be called on outcome data "
                "(e.g. `transform(Y)`) before calling `untransform`, since "
                "means and standard deviations need to be computed."
            )

        Y_utf = self.means + self.stdvs * Y
        Yvar_utf = self._stdvs_sq * Yvar if Yvar is not None else None
        return Y_utf, Yvar_utf

    @property
    def _is_linear(self) -> bool:
        return True

    def untransform_posterior(
        self, posterior: Posterior
    ) -> GPyTorchPosterior | TransformedPosterior:
        r"""Un-standardize the posterior.

        Args:
            posterior: A posterior in the standardized space.

        Returns:
            The un-standardized posterior. If the input posterior is a
            `GPyTorchPosterior`, return a `GPyTorchPosterior`. Otherwise, return a
            `TransformedPosterior`.
        """
        if self._outputs is not None:
            raise NotImplementedError(
                "Standardize does not yet support output selection for "
                "untransform_posterior"
            )
        if not self._is_trained:
            raise RuntimeError(
                "`Standardize` transforms must be called on outcome data "
                "(e.g. `transform(Y)`) before calling `untransform_posterior`, since "
                "means and standard deviations need to be computed."
            )
        is_mtgp_posterior = False
        if type(posterior) is GPyTorchPosterior:
            is_mtgp_posterior = posterior._is_mt
        if not self._m == posterior._extended_shape()[-1] and not is_mtgp_posterior:
            raise RuntimeError(
                "Incompatible output dimensions encountered. Transform has output "
                f"dimension {self._m} and posterior has "
                f"{posterior._extended_shape()[-1]}."
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
        mvn = posterior.distribution
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

    def __init__(self, outputs: list[int] | None = None) -> None:
        r"""Log-transform outcomes.

        Args:
            outputs: Which of the outputs to log-transform. If omitted, all
                outputs will be standardized.
        """
        super().__init__()
        self._outputs = outputs

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
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
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
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
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
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

    def untransform_posterior(self, posterior: Posterior) -> TransformedPosterior:
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

    def __init__(self, power: float, outputs: list[int] | None = None) -> None:
        r"""Power-transform outcomes.

        Args:
            outputs: Which of the outputs to power-transform. If omitted, all
                outputs will be standardized.
        """
        super().__init__()
        self._outputs = outputs
        self.power = power

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
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
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
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
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
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

    def untransform_posterior(self, posterior: Posterior) -> TransformedPosterior:
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

    def __init__(self, outputs: list[int] | None = None) -> None:
        r"""Bilog-transform outcomes.

        Args:
            outputs: Which of the outputs to Bilog-transform. If omitted, all
                outputs will be transformed.
        """
        super().__init__()
        self._outputs = outputs

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
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
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
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
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
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
        Y_utf = Y.sign() * Y.abs().expm1()
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

    def untransform_posterior(self, posterior: Posterior) -> TransformedPosterior:
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
            sample_transform=lambda x: x.sign() * x.abs().expm1(),
        )


def _nanmax(
    tensor: Tensor, dim: int | None = None, keepdim: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    min_value = torch.finfo(tensor.dtype).min
    if dim is None:
        return tensor.nan_to_num(min_value).max()
    return tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)


def _nanmin(
    tensor: Tensor, dim: int | None = None, keepdim: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    max_value = torch.finfo(tensor.dtype).max
    if dim is None:
        return tensor.nan_to_num(max_value).min()
    return tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)


def _check_batched_output(Y: Tensor, batch_shape: Tensor) -> None:
    """Utility for common output transform checks."""
    if Y.shape[:-2] != batch_shape:
        raise RuntimeError(
            f"Expected Y.shape[:-2] to be {batch_shape}, matching "
            "the `batch_shape` argument to `Standardize`, but got "
            f"Y.shape[:-2]={Y.shape[:-2]}."
        )

    if Y.shape[-2] < 1:
        raise ValueError(f"Can't transform with no observations. {Y.shape=}.")


class InfeasibleTransform(OutcomeTransform):
    """Transforms infeasible (NaN) values to feasible values."""

    def __init__(self, batch_shape: torch.Size | None = None) -> None:
        """Transforms infeasible (NaN) values to feasible values.

        Args:
            batch_shape: The batch shape of the outcomes.
        """
        super().__init__()
        self._is_trained = False
        self.register_buffer("_shift", None)
        self.register_buffer("warped_bad_value", torch.tensor(float("nan")))

        self._batch_shape = batch_shape

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Transform the outcomes by handling NaN values.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:
            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        _check_batched_output(Y, self._batch_shape)

        if Yvar is not None:
            raise NotImplementedError(
                "InfeasibleTransform does not support transforming observation noise"
            )

        if self.training:
            if torch.isnan(Y).all(dim=-2).any():
                raise RuntimeError("For at least one batch, all outcomes are NaN")

            labels_range = _nanmax(Y, dim=-2).values - _nanmin(Y, dim=-2).values
            warped_bad_value = _nanmin(Y, dim=-2).values - (0.5 * labels_range + 1)
            num_feasible = Y.shape[-2] - torch.isnan(Y).sum(dim=-2)

            # Estimate the relative frequency of feasible points
            p_feasible = (0.5 + num_feasible) / (1 + Y.numel())

            self.warped_bad_value = warped_bad_value
            self._shift = -torch.nanmean(Y, dim=-2) * p_feasible - warped_bad_value * (
                1 - p_feasible
            )

            self._is_trained = torch.tensor(True)

        # Expand warped_bad_value to match Y's shape
        expanded_bad_value = self.warped_bad_value.unsqueeze(-2).expand(
            *Y.shape[:-2], Y.shape[-2], -1
        )
        expanded_shift = self._shift.unsqueeze(-2).expand(
            *Y.shape[:-2], Y.shape[-2], -1
        )
        Y = torch.where(torch.isnan(Y), expanded_bad_value, Y)
        Y = torch.where(~torch.isnan(Y), Y + expanded_shift, Y)

        return Y, Yvar

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Un-transform the outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of transformed targets.
            Yvar: A `batch_shape x n x m`-dim tensor of transformed observation
                noises associated with the targets (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:
            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        if not self._is_trained:
            raise RuntimeError(
                "forward() needs to be called before untransform() is called."
            )

        if Yvar is not None:
            raise NotImplementedError(
                "InfeasibleTransform does not support untransforming observation noise"
            )

        # Expand shift to match Y's shape
        expanded_shift = self._shift.unsqueeze(-2).expand(
            *Y.shape[:-2], Y.shape[-2], -1
        )
        Y -= expanded_shift
        # TODO: Handle Yvar
        return Y, Yvar


class LogWarperTransform(OutcomeTransform):
    """Warps an array of labels to highlight the difference between good values.

    Note that this warping is performed on finite values of the array and NaNs are
    untouched.
    """

    def __init__(
        self, batch_shape: torch.Size | None = None, offset: float = 1.5
    ) -> None:
        """Initialize transform.

        Args:
            offset: Offset parameter for the log transformation. Must be > 0.
        """
        super().__init__()
        if offset <= 0:
            raise ValueError("offset must be positive")
        self._is_trained = False
        self._batch_shape = batch_shape
        self.register_buffer("offset", torch.tensor(offset))
        self.register_buffer("_labels_min", torch.tensor(float("nan")))
        self.register_buffer("_labels_max", torch.tensor(float("nan")))

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Transform the outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:
            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        _check_batched_output(Y, self._batch_shape)

        if Yvar is not None:
            raise NotImplementedError(
                "LogWarperTransform does not support transforming observation noise"
            )

        if self.training:
            if torch.isnan(Y).all(dim=-2).any():
                raise RuntimeError("For at least one batch, all outcomes are NaN")

            self._labels_min = _nanmin(Y, dim=-2).values
            self._labels_max = _nanmax(Y, dim=-2).values
            self._is_trained = torch.tensor(True)

        expanded_labels_min = self._labels_min.unsqueeze(-2).expand(
            *Y.shape[:-2], Y.shape[-2], -1
        )
        expanded_labels_max = self._labels_max.unsqueeze(-2).expand(
            *Y.shape[:-2], Y.shape[-2], -1
        )

        # Calculate normalized difference
        norm_diff = (expanded_labels_max - Y) / (
            expanded_labels_max - expanded_labels_min
        )
        Y_transformed = 0.5 - (
            torch.log1p(norm_diff * (self.offset - 1)) / torch.log(self.offset)
        )

        return Y_transformed, Yvar

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Un-transform the outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of transformed targets.
            Yvar: A `batch_shape x n x m`-dim tensor of transformed observation
                noises associated with the targets (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:
            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        if not self._is_trained:
            raise RuntimeError("forward() needs to be called before untransform()")

        if Yvar is not None:
            raise NotImplementedError(
                "LogWarperTransform does not support untransforming observation noise"
            )

        expanded_labels_min = self._labels_min.unsqueeze(-2).expand(
            *Y.shape[:-2], Y.shape[-2], -1
        )
        expanded_labels_max = self._labels_max.unsqueeze(-2).expand(
            *Y.shape[:-2], Y.shape[-2], -1
        )

        Y_untransformed = expanded_labels_max - (
            (torch.exp(torch.log(self.offset) * (0.5 - Y)) - 1)
            * (expanded_labels_max - expanded_labels_min)
            / (self.offset - 1)
        )

        return Y_untransformed, Yvar


class HalfRankTransform(OutcomeTransform):
    """Warps half of the outcomes to fit into a Gaussian distribution.

    This transform warps values below the median to follow a Gaussian distribution while
    leaving values above the median unchanged. NaN values are preserved.
    """

    def __init__(self, batch_shape: torch.Size | None = None) -> None:
        """Initialize transform.

        Args:
            outputs: Which of the outputs to transform. If omitted, all outputs
                will be transformed.
        """
        super().__init__()
        self._batch_shape = batch_shape
        self._is_trained = False
        self._unique_labels = {}
        self._warped_labels = {}
        self.register_buffer("_original_label_medians", torch.tensor([]))

    def _get_std_above_median(self, unique_y: Tensor, y_median: Tensor) -> Tensor:
        # Estimate std of good half
        good_half = unique_y[unique_y >= y_median]
        std = torch.sqrt(((good_half - y_median) ** 2).mean())

        if std == 0:
            std = torch.sqrt(((unique_y - y_median) ** 2).mean())

        if torch.isnan(std):
            std = torch.abs(unique_y - y_median).mean()

        return std

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Transform the outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:
            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        if Yvar is not None:
            raise NotImplementedError(
                "HalfRankTransform does not support transforming observation noise"
            )

        _check_batched_output(Y, self._batch_shape)

        if self.training:
            if torch.isnan(Y).all(dim=-2).any():
                raise RuntimeError("For at least one batch, all outcomes are NaN")

            Y_transformed = Y.clone()

            # Compute median for each batch
            Y_medians = torch.nanmedian(Y, dim=-2).values

            self._original_label_medians.resize_(
                torch.Size((*self._batch_shape, Y.shape[-1]))
            )

            for dim in range(Y.shape[-1]):
                batch_indices = (
                    product(*([m for m in range(n)] for n in self._batch_shape))
                    if len(self._batch_shape) > 0
                    else [  # this allows it to work with no batch dim
                        ...,
                    ]
                )
                for batch_idx in batch_indices:
                    y_median = Y_medians[*batch_idx, dim]
                    y = Y[*batch_idx, :, dim]

                    # Get finite values and their ranks for each batch
                    is_finite_mask = ~torch.isnan(y)
                    ranks = torch.zeros_like(y)

                    unique_y = torch.unique(y[is_finite_mask])

                    for i, val in enumerate(unique_y):
                        ranks[y == val] = i + 1

                    ranks = torch.where(is_finite_mask, ranks, len(unique_y) + 1)

                    # Calculate rank quantiles
                    dedup_median_index = torch.searchsorted(unique_y, y_median)
                    denominator = dedup_median_index + 0.5 * (
                        unique_y[dedup_median_index] == y_median
                    )
                    rank_quantile = 0.5 * (ranks - 0.5) / denominator

                    y_above_median_std = self._get_std_above_median(unique_y, y_median)

                    # Apply transformation
                    rank_ppf = (
                        torch.erfinv(2 * rank_quantile - 1)
                        * y_above_median_std
                        * torch.sqrt(torch.tensor(2.0))
                    )
                    Y_transformed[*batch_idx, :, dim] = torch.where(
                        y < y_median,
                        rank_ppf + y_median,
                        Y_transformed[*batch_idx, :, dim],
                    )

                    # save intermediate values for untransform
                    self._original_label_medians[*batch_idx, dim] = y_median
                    self._unique_labels[(*batch_idx, dim)] = unique_y
                    self._warped_labels[(*batch_idx, dim)] = unique_y

            self._is_trained = torch.tensor(True)

        return Y_transformed, Yvar

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Un-transform the outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of transformed targets.
            Yvar: A `batch_shape x n x m`-dim tensor of transformed observation
                noises associated with the targets (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:
            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        if not self._is_trained:
            raise RuntimeError("forward() needs to be called before untransform()")

        if Yvar is not None:
            raise NotImplementedError(
                "HalfRankTransform does not support untransforming observation noise"
            )

        Y_utf = Y.clone()

        for dim in range(Y.shape[-1]):
            batch_indices = (
                product(*(range(n) for n in self._batch_shape))
                if len(self._batch_shape) > 0
                else [  # this allows it to work with no batch dim
                    ...,
                ]
            )
            for batch_idx in batch_indices:
                y = Y[*batch_idx, :, dim]
                unique_labels = self._unique_labels[(*batch_idx, dim)]
                warped_labels = self._warped_labels[(*batch_idx, dim)]

                # Process values below median
                below_median = y < self._original_label_medians[*batch_idx, dim]
                if below_median.any():
                    # Find nearest warped values and interpolate
                    warped_idx = torch.searchsorted(warped_labels, y[below_median])

                    # Handle edge cases and interpolation
                    for i, (val, idx) in enumerate(zip(y[below_median], warped_idx)):
                        if idx == 0:
                            # Extrapolate below minimum
                            scale = (val - warped_labels[0]) / (
                                warped_labels[-1] - warped_labels[0]
                            )
                            Y_utf[below_median][i] = unique_labels[0] - scale * (
                                unique_labels[-1] - unique_labels[0]
                            )
                        else:
                            # Interpolate between points
                            lower_idx = idx - 1
                            upper_idx = min(idx, len(warped_labels) - 1)

                            original_gap = (
                                unique_labels[upper_idx] - unique_labels[lower_idx]
                            )
                            warped_gap = (
                                warped_labels[upper_idx] - warped_labels[lower_idx]
                            )

                            if warped_gap > 0:
                                scale = (val - warped_labels[lower_idx]) / warped_gap
                                Y_utf[below_median][i] = (
                                    unique_labels[lower_idx] + scale * original_gap
                                )
                            else:
                                Y_utf[below_median][i] = unique_labels[lower_idx]

        return Y_utf, Yvar
