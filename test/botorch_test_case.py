#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import warnings
from unittest import TestCase


class BotorchTestCase(TestCase):
    r"""Basic test case for Botorch.

    Currently, this just ensures that no warnings are suppressed by default.
    """

    def setUp(self):
        warnings.simplefilter("always")
