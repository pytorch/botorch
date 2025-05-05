#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.pathwise.features.maps import (
    DirectSumFeatureMap,
    FourierFeatureMap,
    IndexKernelFeatureMap,
    KernelFeatureMap,
    LinearKernelFeatureMap,
    MultitaskKernelFeatureMap,
    OuterProductFeatureMap,
)
from botorch.sampling.pathwise.utils import get_kernel_num_inputs, transforms
from botorch.utils.dispatcher import Dispatcher
from botorch.utils.sampling import draw_sobol_normal_samples
from gpytorch import kernels
from torch import Size, Tensor
from torch.distributions import Gamma

# IMPLEMENTATION NOTE: This type definition specifies the interface for feature map
# generators.
# It defines a callable that takes a kernel and dimension parameters and returns a
# KernelFeatureMap.
TKernelFeatureMapGenerator = Callable[[kernels.Kernel, int, int], KernelFeatureMap]

# IMPLEMENTATION NOTE: We use a Dispatcher pattern to register different handlers for
# various
# kernel types. This allows for extensibility - new kernel types can be supported by
# adding
# new handler functions registered to this dispatcher.
GenKernelFeatureMap = Dispatcher("gen_kernel_feature_map")


def gen_kernel_feature_map(
    kernel: kernels.Kernel,
    num_random_features: int = 1024,
    num_ambient_inputs: Optional[int] = None,
    **kwargs: Any,
) -> KernelFeatureMap:
    # IMPLEMENTATION NOTE: This function serves as the main entry point for generating
    # feature maps from kernels. It uses the dispatcher to call the appropriate handler
    # based on the kernel type. The function has been updated from the original
    # implementation
    # to use more descriptive parameter names (num_ambient_inputs instead of num_inputs,
    # and num_random_features instead of num_outputs) to better reflect their purpose.
    return GenKernelFeatureMap(
        kernel,
        num_ambient_inputs=num_ambient_inputs,
        num_random_features=num_random_features,
        **kwargs,
    )


def _gen_fourier_features(
    kernel: kernels.Kernel,
    weight_generator: Callable[[Size], Tensor],
    num_random_features: int,
    num_inputs: Optional[int] = None,
    random_feature_scale: Optional[float] = None,
    cosine_only: bool = False,
    **ignore: Any,
) -> FourierFeatureMap:
    # IMPLEMENTATION NOTE: This function implements the random Fourier features method
    # from
    # to approximate stationary kernels. It has been enhanced from
    # the original implementation to support the cosine_only option, which is critical
    # for
    # the ProductKernel implementation where we need to avoid the tensor product of sine
    # and
    # cosine features.

    if not cosine_only and num_random_features % 2:
        raise UnsupportedError(
            f"Expected an even number of random features, but {num_random_features=}."
        )

    # Get the appropriate number of inputs based on kernel configuration
    num_inputs = get_kernel_num_inputs(kernel, num_ambient_inputs=num_inputs)
    input_transform = transforms.InverseLengthscaleTransform(kernel)

    # Handle active dimensions if specified
    if kernel.active_dims is not None:
        num_inputs = len(kernel.active_dims)
        input_transform = transforms.ChainedTransform(
            input_transform, transforms.FeatureSelector(indices=kernel.active_dims)
        )

    # Calculate the constant scaling factor for the features
    constant = torch.tensor(
        2**0.5 * (random_feature_scale or num_random_features**-0.5),
        device=kernel.device,
        dtype=kernel.dtype,
    )
    output_transforms = [transforms.SineCosineTransform(constant)]

    # Handle the cosine_only case by generating random phase shifts
    if cosine_only:
        # IMPLEMENTATION NOTE: When cosine_only is True, we use cosine features with
        # random phases instead of paired sine and cosine features. This is important
        # for ProductKernel where we need to take element-wise products of features.
        bias = (
            2
            * torch.pi
            * torch.rand(num_random_features, device=kernel.device, dtype=kernel.dtype)
        )
        num_raw_features = num_random_features
    else:
        bias = None
        num_raw_features = num_random_features // 2

    # Generate the weight matrix using the provided weight generator
    weight = weight_generator(
        Size([kernel.batch_shape.numel() * num_raw_features, num_inputs])
    ).reshape(*kernel.batch_shape, num_raw_features, num_inputs)

    # Create and return the FourierFeatureMap with appropriate transforms
    return FourierFeatureMap(
        kernel=kernel,
        weight=weight,
        bias=bias,
        input_transform=input_transform,
        output_transform=transforms.ChainedTransform(*output_transforms),
    )


@GenKernelFeatureMap.register(kernels.RBFKernel)
def _gen_kernel_feature_map_rbf(
    kernel: kernels.RBFKernel,
    **kwargs: Any,
) -> KernelFeatureMap:
    # IMPLEMENTATION NOTE: This handler generates Fourier features for the RBF kernel.
    # The RBF (Radial Basis Function) kernel is a stationary kernel, so we can use
    # random Fourier features to approximate it. The weight generator uses normal
    # distributions as specified in Rahimi & Recht (2007).
    def _weight_generator(shape: Size) -> Tensor:
        try:
            n, d = shape
        except ValueError:
            raise UnsupportedError(
                f"Expected `shape` to be 2-dimensional, but {len(shape)=}."
            )

        return draw_sobol_normal_samples(
            n=n,
            d=d,
            device=kernel.device,
            dtype=kernel.dtype,
        )

    return _gen_fourier_features(
        kernel=kernel,
        weight_generator=_weight_generator,
        **kwargs,
    )


@GenKernelFeatureMap.register(kernels.MaternKernel)
def _gen_kernel_feature_map_matern(
    kernel: kernels.MaternKernel,
    **kwargs: Any,
) -> KernelFeatureMap:
    # smoothness parameter nu. The spectral density guides weight sampling.
    # For Matern kernels, we use a different weight generator that incorporates the
    # smoothness parameter nu. Weights follow a distribution based on nu.
    # This follows the Matern kernel's spectral density.
    def _weight_generator(shape: Size) -> Tensor:
        try:
            n, d = shape
        except ValueError:
            raise UnsupportedError(
                f"Expected `shape` to be 2-dimensional, but {len(shape)=}."
            )

        dtype = kernel.dtype
        device = kernel.device
        nu = torch.tensor(kernel.nu, device=device, dtype=dtype)
        normals = draw_sobol_normal_samples(n=n, d=d, device=device, dtype=dtype)
        # For Matern kernels, we sample from a Gamma distribution based on nu
        return Gamma(nu, nu).rsample((n, 1)).rsqrt() * normals

    return _gen_fourier_features(
        kernel=kernel,
        weight_generator=_weight_generator,
        **kwargs,
    )


@GenKernelFeatureMap.register(kernels.ScaleKernel)
def _gen_kernel_feature_map_scale(
    kernel: kernels.ScaleKernel,
    *,
    num_ambient_inputs: Optional[int] = None,
    **kwargs: Any,
) -> KernelFeatureMap:
    active_dims = kernel.active_dims
    num_scale_kernel_inputs = get_kernel_num_inputs(
        kernel=kernel,
        num_ambient_inputs=num_ambient_inputs,
        default=None,
    )
    kwargs_copy = kwargs.copy()
    kwargs_copy["num_ambient_inputs"] = num_scale_kernel_inputs
    feature_map = gen_kernel_feature_map(
        kernel.base_kernel,
        **kwargs_copy,
    )

    if active_dims is not None and active_dims is not kernel.base_kernel.active_dims:
        feature_map.input_transform = transforms.ChainedTransform(
            feature_map.input_transform, transforms.FeatureSelector(indices=active_dims)
        )

    feature_map.output_transform = transforms.ChainedTransform(
        transforms.OutputscaleTransform(kernel), feature_map.output_transform
    )
    return feature_map


@GenKernelFeatureMap.register(kernels.ProductKernel)
def _gen_kernel_feature_map_product(
    kernel: kernels.ProductKernel,
    **kwargs: Any,
) -> KernelFeatureMap:
    feature_maps = []
    for sub_kernel in kernel.kernels:
        feature_map = gen_kernel_feature_map(sub_kernel, **kwargs)
        feature_maps.append(feature_map)
    return OuterProductFeatureMap(feature_maps=feature_maps)


@GenKernelFeatureMap.register(kernels.AdditiveKernel)
def _gen_kernel_feature_map_additive(
    kernel: kernels.AdditiveKernel,
    **kwargs: Any,
) -> KernelFeatureMap:
    feature_maps = []
    for sub_kernel in kernel.kernels:
        feature_map = gen_kernel_feature_map(sub_kernel, **kwargs)
        feature_maps.append(feature_map)
    return DirectSumFeatureMap(feature_maps=feature_maps)


@GenKernelFeatureMap.register(kernels.IndexKernel)
def _gen_kernel_feature_map_index(
    kernel: kernels.IndexKernel,
    **kwargs: Any,
) -> KernelFeatureMap:
    return IndexKernelFeatureMap(kernel=kernel)


@GenKernelFeatureMap.register(kernels.LinearKernel)
def _gen_kernel_feature_map_linear(
    kernel: kernels.LinearKernel,
    *,
    num_inputs: Optional[int] = None,
    **kwargs: Any,
) -> KernelFeatureMap:
    num_features = get_kernel_num_inputs(kernel=kernel, num_ambient_inputs=num_inputs)
    return LinearKernelFeatureMap(kernel=kernel, raw_output_shape=Size([num_features]))


@GenKernelFeatureMap.register(kernels.MultitaskKernel)
def _gen_kernel_feature_map_multitask(
    kernel: kernels.MultitaskKernel,
    **kwargs: Any,
) -> KernelFeatureMap:
    data_feature_map = gen_kernel_feature_map(kernel.data_covar_module, **kwargs)
    return MultitaskKernelFeatureMap(kernel=kernel, data_feature_map=data_feature_map)


@GenKernelFeatureMap.register(kernels.LCMKernel)
def _gen_kernel_feature_map_lcm(
    kernel: kernels.LCMKernel,
    **kwargs: Any,
) -> KernelFeatureMap:
    return _gen_kernel_feature_map_additive(
        kernel=kernel, sub_kernels=kernel.covar_module_list, **kwargs
    )
