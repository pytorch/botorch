#! /usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Optional

from torch import Tensor

from ..posteriors import Posterior


class Model(ABC):
    @abstractmethod
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        **kwargs
    ) -> Posterior:
        pass

    def add_observation_noise(
        self,
        posterior: Posterior,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        **kwargs
    ) -> Posterior:
        raise NotImplementedError

    def fantasize(
        self, X: Tensor, num_samples: int, base_samples: Optional[Tensor] = None
    ) -> "Model":
        raise NotImplementedError
