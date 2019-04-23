#! /usr/bin/env python3

import unittest

import torch


class TestTorchVersion(unittest.TestCase):
    def test_torch_version(self):
        raise Exception(f"torch version is {torch.__version__}")
