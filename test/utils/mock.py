#!/usr/bin/env python3

from typing import List, Optional

import torch
from botorch.models import Model
from botorch.posteriors import Posterior
from torch import Tensor


EMPTY_SIZE = torch.Size()


class MockPosterior(Posterior):
    """Mock object that implements dummy methods and feeds through specified outputs"""

    def __init__(self, mean=None, variance=None, samples=None):
        self._mean = mean
        self._variance = variance
        self._samples = samples

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        """Mock sample by repeating only the 0th dimension of self._samples."""
        if base_samples is not None:
            raise RuntimeError("base_samples are not supported in MockPosterior")
        if sample_shape is None:
            sample_shape = torch.Size()
        # create list with a one for each dimension of self._samples
        size = torch.ones(self._samples.ndimension(), dtype=torch.int).tolist()
        # set the number of times to repeat the 0th dimension
        size[0] = sample_shape.numel()
        return self._samples.repeat(torch.Size(size))


class MockModel(Model):
    """Mock object that implements dummy methods and feeds through specified outputs"""

    def __init__(self, posterior: MockPosterior):
        self._posterior = posterior

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
    ) -> MockPosterior:
        return self._posterior


class MockBatchAcquisitionModule:
    """Mock batch acquisition module that returns the sum of the input."""

    def __init__(self):
        self.model = MockModel(None)

    def __call__(self, X):
        return X.sum((1, 2))
