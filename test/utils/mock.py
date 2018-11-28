#!/usr/bin/env python3

import torch
from gpytorch.lazy import LazyTensor, NonLazyTensor


empty_size = torch.Size()


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

    def rsample(self, sample_shape=empty_size):
        """
        Mock rsample by repeating only the 0th dimension of self._samples.
        """
        # create list with a one for each dimension of self._samples
        size = torch.ones(self._samples.ndimension(), dtype=torch.int).tolist()
        # set the number of times to repeat the 0th dimension
        size[0] = sample_shape.numel()
        return self._samples.repeat(torch.Size(size))


class MockModel(object):
    """Mock object that implements dummy methods and feeds through specified outputs"""

    def __init__(self, output):
        self.output = output

    def __call__(self, test_x):
        return self.output

    def eval(self):
        pass

    def train(self):
        pass


class MockBatchAcquisitionModule(object):
    """Mock batch acquisition module that returns the sum of the input"""

    def __init__(self):
        self.model = MockModel(None)

    def __call__(self, X):
        return X.sum()
