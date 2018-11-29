#! /usr/bin/env python3

from typing import Optional

import torch
from gpytorch import fast_pred_var
from gpytorch.distributions import MultivariateNormal
from torch import Tensor

from .posterior import Posterior


class GPyTorchPosterior(Posterior):
    def __init__(self, mvn: MultivariateNormal) -> None:
        self.mvn = mvn

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        if sample_shape is not None and base_samples is not None:
            raise RuntimeError("Use either sample_shape or base_samples, not both.")
        if sample_shape is None and base_samples is None:
            kwargs = {}
        elif sample_shape is not None:
            kwargs = {"sample_shape": sample_shape}
        elif base_samples is not None:
            kwargs = {"base_samples": base_samples}
        with fast_pred_var():
            return self.mvn.rsample(**kwargs)

    @property
    def mean(self):
        return self.mvn.mean

    @property
    def variance(self):
        return self.mvn.variance

    def zero_mean_mvn_samples(self, num_samples: int) -> Tensor:
        return self.mvn.lazy_covariance_matrix.zero_mean_mvn_samples(num_samples)
