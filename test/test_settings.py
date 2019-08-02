#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from botorch import settings


class TestSettings(unittest.TestCase):
    def test_propagate_grads(self):
        pgrads = settings.propagate_grads
        self.assertFalse(pgrads.on())
        self.assertTrue(pgrads.off())
        with settings.propagate_grads(True):
            self.assertTrue(pgrads.on())
            self.assertFalse(pgrads.off())
        self.assertFalse(pgrads.on())
        self.assertTrue(pgrads.off())
