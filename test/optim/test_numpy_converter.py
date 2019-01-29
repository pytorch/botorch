#! /usr/bin/env python3

import unittest
from collections import OrderedDict

import numpy as np
import torch
from botorch.optim.numpy_converter import module_to_array, set_params_with_array
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.models.exact_gp import ExactGP


def _get_index(property_dict, parameter_name):
    idx = 0
    for p_name, ta in property_dict.items():
        if p_name == parameter_name:
            break
        idx += ta.shape.numel()
    return idx


class TestNumpyTorchParameterConversion(unittest.TestCase):
    def test_set_parameters_with_numpy(self):
        # Get an example module with parameters
        likelihood = GaussianLikelihood()
        model = ExactGP(
            torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([4.0]), likelihood
        )
        model.covar_module = RBFKernel(3)
        model.mean_module = ConstantMean()

        model.parameter_bounds = {"mean_module.constant": (None, 10.0)}
        likelihood.parameter_bounds = {"noise_covar.raw_noise": (0.0, None)}
        mll = ExactMarginalLogLikelihood(likelihood, model)
        bounds_dict = {"model.covar_module.raw_lengthscale": (0.1, None)}

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
        self.assertIsInstance(bounds, np.ndarray)

        lower_exp = np.full_like(x, 0.1)
        lower_exp[_get_index(property_dict, "likelihood.noise_covar.raw_noise")] = 0.0
        lower_exp[_get_index(property_dict, "model.mean_module.constant")] = -np.inf
        self.assertTrue(np.equal(bounds[0], lower_exp).all())

        upper_exp = np.full_like(x, np.inf)
        upper_exp[_get_index(property_dict, "model.mean_module.constant")] = 10.0
        self.assertTrue(np.equal(bounds[1], upper_exp).all())

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
