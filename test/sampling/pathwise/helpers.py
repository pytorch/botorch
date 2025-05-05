#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Type, TypeVar

import torch
from botorch import models
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.pathwise.utils import get_train_inputs
from gpytorch import kernels
from torch import Size
from torch.nn.functional import pad

T = TypeVar("T")
TFactory = Callable[[], Iterator[T]]


# TestCaseConfig: Configuration dataclass for test setup
# - Provides consistent test parameters across different test cases
# - Includes device, dtype, dimensions, and other key parameters
@dataclass(frozen=True)
class TestCaseConfig:
    device: torch.device
    dtype: torch.dtype = torch.float64
    seed: int = 0
    num_inputs: int = 2
    num_tasks: int = 2
    num_train: int = 5
    batch_shape: Size = field(default_factory=Size)
    num_random_features: int = 4096


# gen_random_inputs: Generates random input tensors for testing
# - Handles both single-task and multi-task models
# - Supports transformed/untransformed inputs
# - Manages task indices for multi-task models
def gen_random_inputs(
    model: Model,
    batch_shape: Iterable[int],
    transformed: bool = False,
    task_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate random inputs for testing.

    Args:
        model: Model to generate inputs for
        batch_shape: Shape of batch dimension
        transformed: Whether to return transformed inputs
        task_id: Optional task ID for multi-task models
        seed: Optional random seed

    Returns:
        Tensor: Random input tensor
    """
    with nullcontext() if seed is None else torch.random.fork_rng():
        if seed:
            torch.random.manual_seed(seed)

        (train_X,) = get_train_inputs(model, transformed=True)
        tkwargs = {"device": train_X.device, "dtype": train_X.dtype}
        X = torch.rand((*batch_shape, train_X.shape[-1]), **tkwargs)
        if isinstance(model, models.MultiTaskGP):
            num_tasks = model.task_covar_module.raw_var.shape[-1]
            X[..., model._task_feature] = (
                torch.randint(num_tasks, size=X.shape[:-1], **tkwargs)
                if task_id is None
                else task_id
            )

        if not transformed and hasattr(model, "input_transform"):
            return model.input_transform.untransform(X)

        return X


class FactoryFunctionRegistry:
    def __init__(self, factories: Optional[Dict[T, TFactory]] = None):
        """Initialize the registry with optional factories dictionary.

        Args:
            factories: Optional dictionary mapping types to factory functions
        """
        self.factories = {} if factories is None else factories

    def register(self, typ: T, **kwargs: Any) -> None:
        def _(factory: TFactory) -> TFactory:
            self.set_factory(typ, factory, **kwargs)
            return factory

        return _

    def set_factory(self, typ: T, factory: TFactory, exist_ok: bool = False) -> None:
        if not exist_ok and typ in self.factories:
            raise ValueError(f"A factory for {typ} already exists but {exist_ok=}.")
        self.factories[typ] = factory

    def get_factory(self, typ: T) -> Optional[TFactory]:
        return self.factories.get(typ)

    def __call__(self, typ: T, *args: Any, **kwargs: Any) -> T:
        factory = self.get_factory(typ)
        if factory is None:
            raise RuntimeError(f"Factory lookup failed for {typ=}.")
        return factory(*args, **kwargs)


gen_module = FactoryFunctionRegistry()


def _randomize_lengthscales(
    kernel: kernels.Kernel, seed: Optional[int] = None
) -> kernels.Kernel:
    if kernel.ard_num_dims is None:
        raise NotImplementedError

    with nullcontext() if seed is None else torch.random.fork_rng():
        if seed:
            torch.random.manual_seed(seed)

        kernel.lengthscale = (0.25 * kernel.ard_num_dims**0.5) * (
            0.25 + 0.75 * torch.rand_like(kernel.lengthscale)
        )

    return kernel


@gen_module.register(kernels.RBFKernel)
def _gen_kernel_rbf(config: TestCaseConfig, **kwargs: Any) -> kernels.RBFKernel:
    kwargs.setdefault("batch_shape", config.batch_shape)
    kwargs.setdefault("ard_num_dims", config.num_inputs)

    kernel = kernels.RBFKernel(**kwargs)
    return _randomize_lengthscales(
        kernel.to(device=config.device, dtype=config.dtype), seed=config.seed
    )


@gen_module.register(kernels.MaternKernel)
def _gen_kernel_matern(config: TestCaseConfig, **kwargs: Any) -> kernels.MaternKernel:
    kwargs.setdefault("batch_shape", config.batch_shape)
    kwargs.setdefault("ard_num_dims", config.num_inputs)
    kwargs.setdefault("nu", 2.5)
    kernel = kernels.MaternKernel(**kwargs)
    return _randomize_lengthscales(
        kernel.to(device=config.device, dtype=config.dtype), seed=config.seed
    )


@gen_module.register(kernels.LinearKernel)
def _gen_kernel_linear(config: TestCaseConfig, **kwargs: Any) -> kernels.LinearKernel:
    kwargs.setdefault("batch_shape", config.batch_shape)
    kwargs.setdefault("active_dims", [0])

    kernel = kernels.LinearKernel(**kwargs)
    return kernel.to(device=config.device, dtype=config.dtype)


@gen_module.register(kernels.IndexKernel)
def _gen_kernel_index(config: TestCaseConfig, **kwargs: Any) -> kernels.IndexKernel:
    kwargs.setdefault("batch_shape", config.batch_shape)
    kwargs.setdefault("num_tasks", config.num_tasks)
    kwargs.setdefault("rank", kwargs["num_tasks"])
    kwargs.setdefault("active_dims", [0])

    kernel = kernels.IndexKernel(**kwargs)
    return kernel.to(device=config.device, dtype=config.dtype)


@gen_module.register(kernels.ScaleKernel)
def _gen_kernel_scale(config: TestCaseConfig, **kwargs: Any) -> kernels.ScaleKernel:
    kernel = kernels.ScaleKernel(gen_module(kernels.LinearKernel, config), **kwargs)
    return kernel.to(device=config.device, dtype=config.dtype)


@gen_module.register(kernels.ProductKernel)
def _gen_kernel_product(config: TestCaseConfig, **kwargs: Any) -> kernels.ProductKernel:
    kernel = kernels.ProductKernel(
        gen_module(kernels.RBFKernel, config),
        gen_module(kernels.LinearKernel, config),
        **kwargs,
    )
    return kernel.to(device=config.device, dtype=config.dtype)


@gen_module.register(kernels.AdditiveKernel)
def _gen_kernel_additive(
    config: TestCaseConfig, **kwargs: Any
) -> kernels.AdditiveKernel:
    kernel = kernels.AdditiveKernel(
        gen_module(kernels.RBFKernel, config),
        gen_module(kernels.LinearKernel, config),
        **kwargs,
    )
    return kernel.to(device=config.device, dtype=config.dtype)


@gen_module.register(kernels.MultitaskKernel)
def _gen_kernel_multitask(
    config: TestCaseConfig, **kwargs: Any
) -> kernels.MultitaskKernel:
    kwargs.setdefault("batch_shape", config.batch_shape)
    kwargs.setdefault("num_tasks", config.num_tasks)
    kwargs.setdefault("rank", kwargs["num_tasks"])

    kernel = kernels.MultitaskKernel(gen_module(kernels.LinearKernel, config), **kwargs)
    return kernel.to(device=config.device, dtype=config.dtype)


@gen_module.register(kernels.LCMKernel)
def _gen_kernel_lcm(config: TestCaseConfig, **kwargs) -> kernels.LCMKernel:
    kwargs.setdefault("num_tasks", config.num_tasks)
    kwargs.setdefault("rank", kwargs["num_tasks"])

    base_kernels = (
        gen_module(kernels.RBFKernel, config),
        gen_module(kernels.LinearKernel, config),
    )
    kernel = kernels.LCMKernel(base_kernels, **kwargs)
    return kernel.to(device=config.device, dtype=config.dtype)


def _gen_single_task_model(
    model_type: Type[Model],
    config: TestCaseConfig,
    covar_module: Optional[kernels.Kernel] = None,
) -> Model:
    if len(config.batch_shape) > 1:
        raise NotImplementedError

    d = config.num_inputs
    n = config.num_train
    tkwargs = {"device": config.device, "dtype": config.dtype}
    with torch.random.fork_rng():
        torch.random.manual_seed(config.seed)
        covar_module = covar_module or gen_module(kernels.MaternKernel, config)
        uppers = 1 + 9 * torch.rand(d, **tkwargs)
        bounds = pad(uppers.unsqueeze(0), (0, 0, 1, 0))
        X = uppers * torch.rand(n, d, **tkwargs)
        Y = X @ torch.randn(*config.batch_shape, d, 1, **tkwargs)
        if config.batch_shape:
            Y = Y.squeeze(-1).transpose(-2, -1)

        model_args = {
            "train_X": X,
            "train_Y": Y,
            "covar_module": covar_module,
            "input_transform": Normalize(d=X.shape[-1], bounds=bounds),
            "outcome_transform": Standardize(m=Y.shape[-1]),
        }
        if model_type is models.SingleTaskGP:
            model = models.SingleTaskGP(**model_args)
        elif model_type is models.SingleTaskVariationalGP:
            model = models.SingleTaskVariationalGP(
                num_outputs=Y.shape[-1], **model_args
            )
        else:
            raise UnsupportedError(f"Encountered unexpected model type: {model_type}.")

    return model.to(**tkwargs)


for typ in (models.SingleTaskGP, models.SingleTaskVariationalGP):
    gen_module.set_factory(typ, partial(_gen_single_task_model, typ))


@gen_module.register(models.ModelListGP)
def _gen_model_list(config: TestCaseConfig, **kwargs: Any) -> models.ModelListGP:
    return models.ModelListGP(
        gen_module(models.SingleTaskGP, config),
        gen_module(models.SingleTaskGP, replace(config, seed=config.seed + 1)),
        **kwargs,
    )


@gen_module.register(models.MultiTaskGP)
def _gen_model_multitask(
    config: TestCaseConfig,
    covar_module: Optional[kernels.Kernel] = None,
) -> models.MultiTaskGP:
    d = config.num_inputs
    if d == 1:
        raise NotImplementedError("MultiTaskGP inputs must have two or more features.")

    m = config.num_tasks
    n = config.num_train
    tkwargs = {"device": config.device, "dtype": config.dtype}
    batch_shape = Size()  # MTGP currently does not support batch mode
    with torch.random.fork_rng():
        torch.random.manual_seed(config.seed)
        covar_module = covar_module or gen_module(
            kernels.MaternKernel, replace(config, num_inputs=d - 1)
        )
        X = torch.concat(
            [
                torch.rand(*batch_shape, m, n, d - 1, **tkwargs),
                torch.arange(m, **tkwargs)[:, None, None].repeat(*batch_shape, 1, n, 1),
            ],
            dim=-1,
        )
        Y = (X[..., :-1] * torch.randn(*batch_shape, m, n, d - 1, **tkwargs)).sum(-1)
        X = X.view(*batch_shape, -1, d)
        Y = Y.view(*batch_shape, -1, 1)

        model = models.MultiTaskGP(
            train_X=X,
            train_Y=Y,
            task_feature=-1,
            rank=m,
            covar_module=covar_module,
            outcome_transform=Standardize(m=Y.shape[-1], batch_shape=batch_shape),
        )

    return model.to(**tkwargs)
