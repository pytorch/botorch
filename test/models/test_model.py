#! /usr/bin/env python3

import unittest

from botorch.models.model import Model


class TestAbstractBaseModel(unittest.TestCase):
    def test_abstract_base_model(self):
        with self.assertRaises(TypeError):
            Model()
