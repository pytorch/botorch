#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any, Dict, List, Optional, Type

import torch
from torch import Tensor
from torch.nn import ModuleList

from ..exceptions.errors import UnsupportedError
from ..exceptions.warnings import BotorchWarning
from ..models.model import Model
from ..sampling.samplers import MCSampler, SobolQMCNormalSampler
from ..utils.transforms import match_batch_shape, t_batch_mode_transform
from .acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from .analytic import AnalyticAcquisitionFunction, PosteriorMean
from .monte_carlo import MCAcquisitionFunction
from .objective import AcquisitionObjective, MCAcquisitionObjective, ScalarizedObjective


class qMultiStepLookahead(MCAcquisitionFunction, OneShotAcquisitionFunction):
    r"""MC-based batch Multi-Step Look-Ahead (one-shot optimization)."""

    def __init__(
        self,
        model: Model,
        batch_sizes: List[int],
        num_fantasies: List[int],
        value_function_cls: Type[AcquisitionFunction] = PosteriorMean,
        value_function_kwargs: Optional[Dict[str, Any]] = None,
        objective: Optional[AcquisitionObjective] = None,
        inner_sampler: Optional[MCSampler] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""q-Multi-Step Look-Ahead (one-shot optimization).

        Performs a `k`-step lookahead by means of repeated fantasizing.

        Allows to specify a (terminal) value function by passing in the respective class
        object and (optionally) the kwargs for the constuctor. If `value_function_cls`
        is `PosteriorMean`, this is the (parallel) multi-step Knowledge Gradient. If, in
        addition, `k=1` and `q_1 = 1`, this reduces to the classic Knowledge Gradient.

        WARNING: The complexity of evaluating this function is exponential in the number
        of lookahead steps!

        Args:
            model: A fitted model.
            batch_sizes: A list `[q_0, ..., q_k]` containing the batch sizes for the
                initial step (`q_0`) as well as the batch sizes to use for the `k`
                look-ahead steps.
            num_fantasies: A list `[f_1, ..., f_k]` containing the number of fantasy
                points to use for the `k` look-ahead steps.
            value_function_cls: The acquisition function class to be used as the
                terminal value function.
            value_function_kwargs: A dictionary of keyword arguments for the value
                function constructor.
            objective: The objective under which the output is evaluated. If `None`, use
                the model output (requires a single-output model). If a
                `ScalarizedObjective` and `value_function_cls` is a subclass of
                `AnalyticAcquisitonFunction`, then the analytic posterior mean is used.
                Otherwise the objective is MC-evaluated (using `inner_sampler`).
            inner_sampler: The sampler used for inner sampling. Ignored if the objective
                is `None` or a `ScalarizedObjective`.
            X_pending: A `m x d`-dim Tensor of `m` design points that have points that
                have been submitted for function evaluation but have not yet been
                evaluated. Concatenated into `X` upon forward call. Copied and set to
                have no gradient.
        """
        super(MCAcquisitionFunction, self).__init__(model=model)
        self.batch_sizes = batch_sizes
        self.num_fantasies = num_fantasies
        # construct samplers for the look-ahead steps (excluding inner_sampler)
        samplers: List[MCSampler] = [
            SobolQMCNormalSampler(
                num_samples=nf, resample=False, collapse_batch_dims=True
            )
            for nf in num_fantasies
        ]
        # ensure combination of value function and objective makes sense before
        # computing a bunch of stuff
        if issubclass(value_function_cls, AnalyticAcquisitionFunction):
            if objective is not None and not isinstance(objective, ScalarizedObjective):
                raise UnsupportedError(
                    "Only objectives of type ScalarizedObjective are supported "
                    "for analytic value functions."
                )
            if inner_sampler is not None:
                warnings.warn(
                    "inner_sampler is ignored for analytic acquistion functions",
                    BotorchWarning,
                )
                inner_sampler = None
        elif issubclass(value_function_cls, MCAcquisitionFunction):
            if value_function_cls == qMultiStepLookahead:
                raise UnsupportedError(
                    "qMultiStepLookahead not supported as a value function "
                    "(I see what you did there, nice try)."
                )
            if objective is not None and not isinstance(
                objective, MCAcquisitionObjective
            ):
                raise UnsupportedError(
                    "Only objectives of type MCAcquisitionObjective are supported "
                    "for MC value functions."
                )
            if inner_sampler is None:
                inner_sampler = SobolQMCNormalSampler(
                    num_samples=256, resample=False, collapse_batch_dims=True
                )
        self.samplers = ModuleList(samplers + [inner_sampler])
        self.objective = objective
        self.set_X_pending(X_pending)
        self._value_function_cls = value_function_cls
        self._value_function_kwargs = value_function_kwargs or {}

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qMultiStepLookahead on the candidate set X.

        Args:
            X: A `batch_shape x q' x d`-dim Tensor with `q'` design points for each
                batch, where `q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`. Here `q_i`
                is the number of candidates jointly considered in look-ahead step
                `i`, and `f_i` is respective number of fantasies.

        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape`.
        """
        batch_shape, d = X.shape[:-2], X.shape[-1]
        k = len(self.num_fantasies)
        # X_i needs to have shape f_i x .... x f_1 x batch_shape x q_i x d
        shapes = [
            torch.Size(
                self.num_fantasies[:i][::-1] + [*batch_shape, self.batch_sizes[i], d]
            )
            for i in range(k + 1)
        ]
        # Each X_i in Xsplit has shape batch_shape x qtilde x d with
        # qtilde = f_i * ... * f_1 * q_i
        split_sizes = [s[:-3].numel() * s[-2] for s in shapes]
        Xsplit = torch.split(X, split_sizes, dim=-2)
        # now reshape (need to permute batch_shape and qtilde dimensions for i > 0)
        perm = [-2] + list(range(len(batch_shape))) + [-1]
        X0 = Xsplit[0].reshape(shapes[0])
        Xother = [
            X.permute(*perm).reshape(shape) for X, shape in zip(Xsplit[1:], shapes[1:])
        ]
        # concatenate in pending points
        if self.X_pending is not None:
            X0 = torch.cat([X0, match_batch_shape(self.X_pending, X0)], dim=-2)
        return _step(
            model=self.model,
            Xs=[X0] + Xother,
            samplers=self.samplers,
            value_function_cls=self._value_function_cls,
            value_function_kwargs=self._value_function_kwargs,
            objective=self.objective,
            first_step=True,
        )

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""We only return X as the set of candidates post-optimization.

        Args:
            X_full: A `batch_shape x q' x d`-dim Tensor with `q'` design points for
                each batch, where `q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`.

        Returns:
            A `batch_shape x q x d`-dim Tensor with `q` design points for each batch.
        """
        return X_full[..., : self.batch_sizes[0], :]


def _step(
    model: Model,
    Xs: List[Tensor],
    samplers: List[Optional[MCSampler]],
    value_function_cls: Type[AcquisitionFunction],
    value_function_kwargs: Dict[str, Any],
    objective: AcquisitionObjective,
    first_step: bool = False,
) -> Tensor:
    r"""Recursive multi-step look-ahead computation.

    Helper function computing the "value-to-go" of a multi-step lookahead scheme.

    Args:
        model: A Model of appropriate batch size. Specifically, it must be possible to
            evaluate the model's posterior at `Xs[0]`.
        Xs: A list `[X_j, ..., X_k]` of tensors, where `X_i` has shape
            `f_i x .... x f_1 x batch_shape x q_i x d`.
        samplers: A list of `k - j` samplers, such that the number of samples of sampler
            `i` is `f_i`. The last element of this list is considered the
            "inner sampler", which is used for evaluating the objective in case it is an
            MCAcquisitionObjective.
        value_function_cls: The acquisition function class to be used as the value
            function (providing the terminal value).
        value_function_kwargs: A dictionary of arguments used in the value function
            constructor.
        objective: The AcquisitionObjective under which the model output is evaluated.
        first_step: If True, this is considered to be the first step (resulting
            in not propagating gradients through the training inputs of the model).

    Returns:
        A `b`-dim tensor containing the multi-step value of the design `X`.
    """
    if len(Xs) != len(samplers):
        raise ValueError("Xs and samplers must have the same number elements")

    # base case: no more fantasizing, compute value
    if len(Xs) == 1:
        kwargs: Dict[str, Any] = {"model": model, "objective": objective}
        if issubclass(value_function_cls, MCAcquisitionFunction):
            kwargs["sampler"] = samplers[0]
        terminal_value_func = value_function_cls(**kwargs, **value_function_kwargs)
        obj = terminal_value_func(X=Xs[0])
        # shape of obj is (inner_mc_samples) x f_k x ... x f_1 x batch_shape
        # we average across all dimensions except for the batch dimension
        return obj.view(-1, obj.size(-1)).mean(dim=0)

    # construct fantasy model (with batch shape f_{j+1} x ... x f_1 x batch_shape)
    fantasy_model = model.fantasize(
        X=Xs[0],
        sampler=samplers[0],
        observation_noise=True,
        propagate_grads=not first_step,
    )

    return _step(
        model=fantasy_model,
        Xs=Xs[1:],
        samplers=samplers[1:],
        value_function_cls=value_function_cls,
        value_function_kwargs=value_function_kwargs,
        objective=objective,
    )
