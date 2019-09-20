#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import warnings
from collections import OrderedDict
from typing import List, Optional
from unittest import TestCase

import torch
from torch import Tensor

from ..models.model import Model
from ..posteriors import Posterior


EMPTY_SIZE = torch.Size()


class BotorchTestCase(TestCase):
    r"""Basic test case for Botorch.

    This
        1. sets the default device to be `torch.device("cpu")`
        2. ensures that no warnings are suppressed by default.
    """

    device = torch.device("cpu")

    def setUp(self):
        warnings.simplefilter("always")


class MockPosterior(Posterior):
    r"""Mock object that implements dummy methods and feeds through specified outputs"""

    def __init__(self, mean=None, variance=None, samples=None):
        self._mean = mean
        self._variance = variance
        self._samples = samples

    @property
    def device(self) -> torch.device:
        for t in (self._mean, self._variance, self._samples):
            if torch.is_tensor(t):
                return t.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        for t in (self._mean, self._variance, self._samples):
            if torch.is_tensor(t):
                return t.dtype
        return torch.float32

    @property
    def event_shape(self) -> torch.Size:
        if self._samples is not None:
            return self._samples.shape
        if self._mean is not None:
            return self._mean.shape
        if self._variance is not None:
            return self._variance.shape
        return torch.Size()

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
        """Mock sample by repeating self._samples. If base_samples is provided,
        do a shape check but return the same mock samples."""
        if sample_shape is None:
            sample_shape = torch.Size()
        if sample_shape is not None and base_samples is not None:
            # check the base_samples shape is consistent with the sample_shape
            if base_samples.shape[: len(sample_shape)] != sample_shape:
                raise RuntimeError("sample_shape disagrees with base_samples.")
        return self._samples.expand(sample_shape + self._samples.shape)


class MockModel(Model):
    r"""Mock object that implements dummy methods and feeds through specified outputs"""

    def __init__(self, posterior: MockPosterior) -> None:
        super(Model, self).__init__()
        self._posterior = posterior

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
    ) -> MockPosterior:
        return self._posterior

    @property
    def num_outputs(self) -> int:
        event_shape = self._posterior.event_shape
        return event_shape[-1] if len(event_shape) > 0 else 0

    def state_dict(self) -> None:
        pass

    def load_state_dict(
        self, state_dict: Optional[OrderedDict] = None, strict: bool = False
    ) -> None:
        pass


class MockAcquisitionFunction:
    r"""Mock acquisition function object that implements dummy methods."""

    def __init__(self):
        self.model = None
        self.X_pending = None

    def __call__(self, X):
        return X[..., 0].max(dim=-1)[0]

    def set_X_pending(self, X_pending: Optional[Tensor] = None):
        self.X_pending = X_pending
