import unittest
import torch
from torch import nn
from torch.distributions import Normal
from botorch_community.acquisition.latent_information_gain import LatentInformationGain
from botorch_community.models.np_regression import NeuralProcessModel

class TestLatentInformationGain(unittest.TestCase):
    def setUp(self):
        self.x_dim = 2
        self.y_dim = 1
        self.r_dim = 8
        self.z_dim = 3
        self.r_hidden_dims = [16, 16]
        self.z_hidden_dims = [32, 32]
        self.decoder_hidden_dims = [16, 16]
        self.num_samples = 10
        self.model = NeuralProcessModel(
            r_hidden_dims = self.r_hidden_dims,
            z_hidden_dims = self.z_hidden_dims,
            decoder_hidden_dims = self.decoder_hidden_dims,
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            r_dim=self.r_dim,
            z_dim=self.z_dim,
        )
        self.acquisition_function = LatentInformationGain(
            model=self.model,
            num_samples=self.num_samples,
        )
        self.context_x = torch.rand(10, self.x_dim)
        self.context_y = torch.rand(10, self.y_dim)
        self.candidate_x = torch.rand(5, self.x_dim)

    def test_initialization(self):
        self.assertEqual(self.acquisition_function.num_samples, self.num_samples)
        self.assertEqual(self.acquisition_function.model, self.model)

    def test_acquisition_shape(self):
        lig_score = self.acquisition_function.acquisition(
            candidate_x=self.candidate_x,
            context_x=self.context_x,
            context_y=self.context_y,
        )
        self.assertTrue(torch.is_tensor(lig_score))
        self.assertEqual(lig_score.shape, ())

    def test_acquisition_kl(self):
        lig_score = self.acquisition_function.acquisition(
            candidate_x=self.candidate_x,
            context_x=self.context_x,
            context_y=self.context_y,
        )
        self.assertGreaterEqual(lig_score.item(), 0)

    def test_acquisition_samples(self):
        lig_1 = self.acquisition_function.acquisition(
            candidate_x=self.candidate_x,
            context_x=self.context_x,
            context_y=self.context_y,
        )
        
        self.acquisition_function.num_samples = 20
        lig_2 = self.acquisition_function.acquisition(
            candidate_x=self.candidate_x,
            context_x=self.context_x,
            context_y=self.context_y,
        )
        self.assertTrue(lig_2.item() < lig_1.item())
        self.assertTrue(abs(lig_2.item() - lig_1.item()) < 0.2)

    def test_acquisition_invalid_inputs(self):
        invalid_context_x = torch.rand(10, self.x_dim + 5)  
        with self.assertRaises(Exception):
            self.acquisition_function.acquisition(
                candidate_x=self.candidate_x,
                context_x=invalid_context_x,
                context_y=self.context_y,
            )

        invalid_candidate_x = torch.rand(5, self.x_dim + 5)  
        with self.assertRaises(Exception):
            self.acquisition_function.acquisition(
                candidate_x=invalid_candidate_x,
                context_x=self.context_x,
                context_y=self.context_y,
            )


if __name__ == "__main__":
    unittest.main()
