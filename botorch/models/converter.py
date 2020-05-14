#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for converting between different models.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from botorch.exceptions import UnsupportedError
from botorch.models.gp_regression import FixedNoiseGP, HeteroskedasticSingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model_list_gp_regression import ModelListGP
from torch.nn import Module


def _get_module(module: Module, name: str) -> Module:
    """Recursively get a sub-module from a module.

    Args:
        module: A `torch.nn.Module`.
        name: The name of the submodule to return, in the form of a period-delinated
            string: `sub_module.subsub_module.[...].leaf_module`.

    Returns:
        The requested sub-module.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> noise_prior = _get_module(gp, "likelihood.noise_covar.noise_prior")
    """
    current = module
    if name != "":
        for a in name.split("."):
            current = getattr(current, a)
    return current


def _check_compatibility(models: ModelListGP) -> None:
    """Check if a ModelListGP can be converted."""
    # check that all submodules are of the same type
    for modn, mod in models[0].named_modules():
        mcls = mod.__class__
        if not all(isinstance(_get_module(m, modn), mcls) for m in models[1:]):
            raise UnsupportedError(
                "Sub-modules must be of the same type across models."
            )

    # check that each model is a BatchedMultiOutputGPyTorchModel
    if not all(isinstance(m, BatchedMultiOutputGPyTorchModel) for m in models):
        raise UnsupportedError(
            "All models must be of type BatchedMultiOutputGPyTorchModel."
        )

    # TODO: Add support for HeteroskedasticSingleTaskGP
    if any(isinstance(m, HeteroskedasticSingleTaskGP) for m in models):
        raise NotImplementedError(
            "Conversion of HeteroskedasticSingleTaskGP is currently unsupported."
        )

    # TODO: Add support for custom likelihoods
    if any(getattr(m, "_is_custom_likelihood", False) for m in models):
        raise NotImplementedError(
            "Conversion of models with custom likelihoods is currently unsupported."
        )

    # check that each model is single-output
    if not all(m._num_outputs == 1 for m in models):
        raise UnsupportedError("All models must be single-output.")

    # check that training inputs are the same
    if not all(
        torch.equal(ti, tj)
        for m in models[1:]
        for ti, tj in zip(models[0].train_inputs, m.train_inputs)
    ):
        raise UnsupportedError("training inputs must agree for all sub-models.")


def model_list_to_batched(model_list: ModelListGP) -> BatchedMultiOutputGPyTorchModel:
    """Convert a ModelListGP to a BatchedMultiOutputGPyTorchModel.

    Args:
        model_list: The `ModelListGP` to be converted to the appropriate
            `BatchedMultiOutputGPyTorchModel`. All sub-models must be of the same
            type and have the shape (batch shape and number of training inputs).

    Returns:
        The model converted into a `BatchedMultiOutputGPyTorchModel`.

    Example:
        >>> list_gp = ModelListGP(gp1, gp2)
        >>> batch_gp = model_list_to_batched(list_gp)
    """
    models = model_list.models
    _check_compatibility(models)

    # if the list has only one model, we can just return a copy of that
    if len(models) == 1:
        return deepcopy(models[0])

    # construct inputs
    train_X = deepcopy(models[0].train_inputs[0])
    train_Y = torch.stack([m.train_targets.clone() for m in models], dim=-1)
    kwargs = {"train_X": train_X, "train_Y": train_Y}
    if isinstance(models[0], FixedNoiseGP):
        kwargs["train_Yvar"] = torch.stack(
            [m.likelihood.noise_covar.noise.clone() for m in models], dim=-1
        )
    if isinstance(models[0], SingleTaskMultiFidelityGP):
        init_args = models[0]._init_args
        if not all(
            v == m._init_args[k] for m in models[1:] for k, v in init_args.items()
        ):
            raise UnsupportedError("All models must have the same fidelity parameters.")
        kwargs.update(init_args)

    # construct the batched GP model
    batch_gp = models[0].__class__(**kwargs)

    tensors = {n for n, p in batch_gp.state_dict().items() if len(p.shape) > 0}
    scalars = set(batch_gp.state_dict()) - tensors
    input_batch_dims = len(models[0]._input_batch_shape)

    # ensure scalars agree (TODO: Allow different priors for different outputs)
    for n in scalars:
        v0 = _get_module(models[0], n)
        if not all(torch.equal(_get_module(m, n), v0) for m in models[1:]):
            raise UnsupportedError("All scalars must have the same value.")

    # ensure dimensions of all tensors agree
    for n in tensors:
        shape0 = _get_module(models[0], n).shape
        if not all(_get_module(m, n).shape == shape0 for m in models[1:]):
            raise UnsupportedError("All tensors must have the same shape.")

    # now construct the batched state dict
    scalar_state_dict = {
        s: p.clone() for s, p in models[0].state_dict().items() if s in scalars
    }
    tensor_state_dict = {
        t: (
            torch.stack(
                [m.state_dict()[t].clone() for m in models], dim=input_batch_dims
            )
            if "active_dims" not in t
            else models[0].state_dict()[t].clone()
        )
        for t in tensors
    }
    batch_state_dict = {**scalar_state_dict, **tensor_state_dict}

    # load the state dict into the new model
    batch_gp.load_state_dict(batch_state_dict)

    return batch_gp


def batched_to_model_list(batch_model: BatchedMultiOutputGPyTorchModel) -> ModelListGP:
    """Convert a BatchedMultiOutputGPyTorchModel to a ModelListGP.

    Args:
        model_list: The `BatchedMultiOutputGPyTorchModel` to be converted to a
            `ModelListGP`.

    Returns:
        The model converted into a `ModelListGP`.

    Example:
        >>> train_X = torch.rand(5, 2)
        >>> train_Y = torch.rand(5, 2)
        >>> batch_gp = SingleTaskGP(train_X, train_Y)
        >>> list_gp = batched_to_model_list(batch_gp)
    """
    # TODO: Add support for HeteroskedasticSingleTaskGP
    if isinstance(batch_model, HeteroskedasticSingleTaskGP):
        raise NotImplementedError(
            "Conversion of HeteroskedasticSingleTaskGP currently not supported."
        )
    batch_sd = batch_model.state_dict()

    tensors = {n for n, p in batch_sd.items() if len(p.shape) > 0}
    scalars = set(batch_sd) - tensors
    input_bdims = len(batch_model._input_batch_shape)

    models = []

    for i in range(batch_model._num_outputs):
        scalar_sd = {s: batch_sd[s].clone() for s in scalars}
        tensor_sd = {
            t: (
                batch_sd[t].select(input_bdims, i).clone()
                if "active_dims" not in t
                else batch_sd[t].clone()
            )
            for t in tensors
        }
        sd = {**scalar_sd, **tensor_sd}
        kwargs = {
            "train_X": batch_model.train_inputs[0].select(input_bdims, i).clone(),
            "train_Y": batch_model.train_targets.select(input_bdims, i)
            .clone()
            .unsqueeze(-1),
        }
        if isinstance(batch_model, FixedNoiseGP):
            noise_covar = batch_model.likelihood.noise_covar
            kwargs["train_Yvar"] = (
                noise_covar.noise.select(input_bdims, i).clone().unsqueeze(-1)
            )
        if isinstance(batch_model, SingleTaskMultiFidelityGP):
            kwargs.update(batch_model._init_args)
        model = batch_model.__class__(**kwargs)
        model.load_state_dict(sd)
        models.append(model)

    return ModelListGP(*models)
