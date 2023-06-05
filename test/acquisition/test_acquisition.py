#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    MCSamplerMixin,
    MultiModelAcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.models.model import ModelDict
from botorch.sampling.normal import IIDNormalSampler
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class DummyMCAcqf(AcquisitionFunction, MCSamplerMixin):
    def __init__(self, model, sampler):
        r"""Dummy acqf for testing MCSamplerMixin."""
        super().__init__(model)
        MCSamplerMixin.__init__(self, sampler)

    def forward(self, X):
        raise NotImplementedError


class DummyMultiModelAcqf(MultiModelAcquisitionFunction):
    def forward(self, X):
        raise NotImplementedError


class TestAcquisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            AcquisitionFunction()


class TestOneShotAcquisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            OneShotAcquisitionFunction()


class TestMCSamplerMixin(BotorchTestCase):
    def test_mc_sampler_mixin(self):
        mm = MockModel(MockPosterior(samples=torch.rand(1, 2)))
        acqf = DummyMCAcqf(model=mm, sampler=None)
        self.assertIsNone(acqf.sampler)
        samples = acqf.get_posterior_samples(mm._posterior)
        self.assertEqual(samples.shape, torch.Size([512, 1, 2]))
        self.assertIsInstance(acqf.sampler, StochasticSampler)
        sampler = IIDNormalSampler(sample_shape=torch.Size([2]))
        acqf.sampler = sampler
        self.assertIs(acqf.sampler, sampler)


class TestMultiModelAcquisitionFunction(BotorchTestCase):
    def test_multi_model_acquisition_function(self):
        model_dict = ModelDict(
            m1=MockModel(MockPosterior()),
            m2=MockModel(MockPosterior()),
        )
        with self.assertRaises(TypeError):
            MultiModelAcquisitionFunction(model_dict=model_dict)
        acqf = DummyMultiModelAcqf(model_dict=model_dict)
        self.assertIs(acqf.model_dict, model_dict)
