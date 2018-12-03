#! /usr/bin/env python3

import unittest
from collections import OrderedDict

import numpy as np
import torch
from botorch.optim.converter import module_to_array, set_params_with_array
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.models.exact_gp import ExactGP


class TestNumpyTorchParameterConversion(unittest.TestCase):
    def test_set_parameters_with_numpy(self):
        # Get an example module with parameters
        likelihood = GaussianLikelihood()
        model = ExactGP(
            torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([4.0]), likelihood
        )
        model.covar_module = RBFKernel(3)
        model.mean_module = ConstantMean()
        mll = ExactMarginalLogLikelihood(likelihood, model)

        bounds_dict = {"likelihood.noise_covar.raw_noise": (0.0, None)}
        x, property_dict, bounds = module_to_array(module=mll, bounds=bounds_dict)
        self.assertTrue(np.array_equal(x, np.zeros(5)))
        self.assertEqual(
            set(property_dict.keys()),
            {
                "likelihood.noise_covar.raw_noise",
                "model.covar_module.raw_lengthscale",
                "model.mean_module.constant",
            },
        )
        sizes = [torch.Size([1, 1]), torch.Size([1, 1, 3]), torch.Size([1, 1])]
        for i, val in enumerate(property_dict.values()):
            self.assertEqual(val.dtype, torch.float32)
            self.assertEqual(val.shape, sizes[i])
            self.assertEqual(val.device, torch.device("cpu"))

        # check bound parsing
        self.assertIsInstance(bounds, tuple)
        lower_exp = np.full_like(x, -np.inf)
        idx = 0
        for p_name, ta in property_dict.items():
            if p_name == "likelihood.noise_covar.raw_noise":
                break
            idx += ta.shape.numel()
        lower_exp[idx] = 0.0
        self.assertTrue(np.equal(bounds[0], lower_exp).all())
        self.assertTrue(np.equal(bounds[1], np.full_like(x, np.inf)).all())

        # Set parameters
        mll = set_params_with_array(
            mll, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), property_dict
        )
        z = OrderedDict(mll.named_parameters())
        self.assertTrue(
            torch.equal(
                z["likelihood.noise_covar.raw_noise"].data, torch.tensor([[1.0]])
            )
        )
        self.assertTrue(
            torch.equal(
                z["model.covar_module.raw_lengthscale"].data,
                torch.tensor([[[2.0, 3.0, 4.0]]]),
            )
        )
        self.assertTrue(
            torch.equal(z["model.mean_module.constant"].data, torch.tensor([[5.0]]))
        )

        # Extract again
        x2, property_dict2, bounds2 = module_to_array(module=mll, bounds=bounds_dict)
        self.assertTrue(np.array_equal(x2, np.array([1.0, 2.0, 3.0, 4.0, 5.0])))


if __name__ == "__main__":
    unittest.main()
