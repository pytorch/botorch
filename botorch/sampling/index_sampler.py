#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from botorch.sampling.base import MCSampler
from botorch.posteriors import Posterior
from torch import Tensor
from botorch.utils.sampling import manual_seed
import torch


class BaseSampler(MCSampler):
    def forward(self, posterior: Posterior) -> Tensor:
        return super().forward(posterior)

    def _construct_base_samples(self, posterior: Posterior):
        if self.base_samples is None or self.base_samples.shape != self.sample_shape:
            with manual_seed(seed=self.seed):
                base_samples = torch.multinomial(
                    torch.ones(self.size) / self.size,
                    num_samples=self.sample_shape.numel(),
                    replacement=True,
                ).reshape(self.sample_shape)
            self.register_buffer("base_samples", base_samples)
        if self.base_samples.device != posterior.device:
            self.to(device=posterior.device)  # pragma: nocover
        if self.base_samples.dtype != posterior.dtype:
            self.to(dtype=posterior.dtype)
