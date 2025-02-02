import unittest
import torch
from botorch_community.acquisition.latent_information_gain import LatentInformationGain
from botorch_community.models.np_regression import NeuralProcessModel

class TestLatentInformationGain(unittest.TestCase):
    def setUp(self):
        self.x_dim = 2
        self.y_dim = 1
        self.r_dim = 8
        self.z_dim = 3
        self.context_x = torch.rand(10, self.x_dim)
        self.context_y = torch.rand(10, self.y_dim)
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
            context_x=self.context_x,
            context_y=self.context_y,
            model=self.model,
            num_samples=self.num_samples,
        )
        self.candidate_x = torch.rand(5, self.x_dim)

    def test_initialization(self):
        self.assertEqual(self.acquisition_function.num_samples, self.num_samples)
        self.assertEqual(self.acquisition_function.model, self.model)

    def test_acquisition_shape(self):
        lig_score = self.acquisition_function.forward(
            candidate_x=self.candidate_x
        )
        self.assertTrue(torch.is_tensor(lig_score))
        self.assertEqual(lig_score.shape, (1, 5))

    def test_acquisition_kl(self):
        lig_score = self.acquisition_function.forward(
            candidate_x=self.candidate_x
        )
        self.assertGreaterEqual(lig_score.mean().item(), 0)

if __name__ == "__main__":
    unittest.main()
