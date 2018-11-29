#! /usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class Posterior(ABC):
    @property
    def mean(self) -> Tensor:
        raise NotImplementedError

    @property
    def variance(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        pass

    def sample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, base_samples=base_samples)

    def zero_mean_mvn_samples(self, num_samples: int) -> Tensor:
        raise NotImplementedError
