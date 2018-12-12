#! /usr/bin/env python3

import unittest

from botorch.models.model import Model


class AbstractTestModel(Model):
    pass


class BaseModelTest(unittest.TestCase):
    def test_abstract_base_model(self):
        with self.assertRaises(TypeError):
            AbstractTestModel()


if __name__ == "__main__":
    unittest.main()
