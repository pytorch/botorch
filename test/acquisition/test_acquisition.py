#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.testing import BotorchTestCase


class TestAcquisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            AcquisitionFunction()
