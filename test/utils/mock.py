#!/usr/bin/env python3

import torch
from gpytorch.lazy import LazyTensor, NonLazyTensor


class MockLikelihood(object):
    """Mock object that implements dummy methods and feeds through specified outputs"""

    def __init__(self, mean=None, covariance=None, samples=None):
        self._mean = mean
        self._covariance = covariance
        self._samples = samples

    @property
    def mean(self):
        return self._mean

    @property
    def covariance_matrix(self):
        if isinstance(self._covariance, LazyTensor):
            return self._covariance.evaluate()
        return self._covariance

    @property
    def lazy_covariance_matrix(self):
        if not isinstance(self._covariance, LazyTensor):
            return NonLazyTensor(self._covariance)
        return self._covariance

    def rsample(self, sample_shape=torch.Size()):
        return self._samples.repeat(sample_shape.numel() or 1, 1, 1)


class MockModel(object):
    """Mock object that implements dummy methods and feeds through specified outputs"""

    def __init__(self, output):
        self.output = output

    def __call__(self, test_x):
        return self.output

    def eval(self):
        pass
