#! /usr/bin/env python3

from abc import ABC, abstractmethod, abstractproperty
from typing import Optional

import torch
from torch import Tensor


class Posterior(ABC):
    @abstractproperty
    def device(self) -> torch.device:
        pass

    @abstractproperty
    def dtype(self) -> torch.dtype:
        pass

    @abstractproperty
    def event_shape(self) -> torch.Size:
        """Return the event shape (i.e. the shape of a single sample)"""
        pass

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
        """
            If both optional arguments are included,
            base_samples must take priority.
        """
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

    def get_base_samples(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        raise NotImplementedError
