#! /usr/bin/env python3

import unittest

import torch
from botorch.optim.converter import numpy_to_state_dict, state_dict_to_numpy


class TestNumpyTorchParameterConversion(unittest.TestCase):
    def setUp(self):
        self.state_dict = {
            "likelihood.log_noise": torch.tensor([0.0], dtype=torch.double),
            "mean_module.constant": torch.tensor([[0.0, 1.0, 2.0]]),
            "covar_module.log_outputscale": torch.tensor([[0.0], [2.0]]),
            "covar_module.base_kernel.log_lengthscale": torch.tensor([[[0.0]]]),
        }

    def test_torch_to_numpy_and_back(self):
        back_and_forth = numpy_to_state_dict(state_dict_to_numpy(self.state_dict))
        # check that all parameters are there
        self.assertEqual(set(back_and_forth), set(self.state_dict))
        # check that the tensors are the same
        for name, tsr in back_and_forth.items():
            self.assertTrue(torch.equal(tsr, self.state_dict[name]))


if __name__ == "__main__":
    unittest.main()
