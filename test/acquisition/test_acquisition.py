#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from botorch.acquisition.acquisition import AcquisitionFunction


class TestAcquisitionFunction(unittest.TestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            AcquisitionFunction()
