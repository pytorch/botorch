#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from botorch.models.model import Model


class NotSoAbstractBaseModel(Model):
    def posterior(self, X, output_indices, observation_noise, **kwargs):
        pass


class TestBaseModel(unittest.TestCase):
    def test_abstract_base_model(self):
        with self.assertRaises(TypeError):
            Model()

    def test_not_so_abstract_base_model(self):
        model = NotSoAbstractBaseModel()
        with self.assertRaises(NotImplementedError):
            model.condition_on_observations(None, None)
