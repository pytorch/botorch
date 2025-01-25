import unittest
import numpy as np
import torch
from botorch_community.models.np_regression import NeuralProcessModel
from botorch.posteriors import GPyTorchPosterior

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestNeuralProcessModel(unittest.TestCase):
    def initialize(self):
        self.r_hidden_dims = [16, 16]
        self.z_hidden_dims = [32, 32]
        self.decoder_hidden_dims = [16, 16]
        self.x_dim = 2
        self.y_dim = 1
        self.r_dim = 8
        self.z_dim = 8
        self.model = NeuralProcessModel(
            self.r_hidden_dims, 
            self.z_hidden_dims,
            self.decoder_hidden_dims,
            self.x_dim,
            self.y_dim,
            self.r_dim,
            self.z_dim,
        )
        self.x_data = np.random.rand(100, self.x_dim)
        self.y_data = np.random.rand(100, self.y_dim)

    def test_r_encoder(self):
        self.initialize()
        input = torch.rand(10, self.x_dim + self.y_dim)
        output = self.model.r_encoder(input)
        self.assertEqual(output.shape, (10, self.r_dim))
        self.assertTrue(torch.is_tensor(output))

    def test_z_encoder(self):
        self.initialize()
        input = torch.rand(10, self.r_dim)
        mean, logvar = self.model.z_encoder(input)
        self.assertEqual(mean.shape, (10, self.z_dim))
        self.assertEqual(logvar.shape, (10, self.z_dim))
        self.assertTrue(torch.is_tensor(mean))
        self.assertTrue(torch.is_tensor(logvar))

    def test_decoder(self):
        self.initialize()
        x_pred = torch.rand(10, self.x_dim)
        z = torch.rand(self.z_dim)
        output = self.model.decoder(x_pred, z)
        self.assertEqual(output.shape, (10, self.y_dim))
        self.assertTrue(torch.is_tensor(output))

    def test_sample_z(self):
        self.initialize()
        mu = torch.rand(self.z_dim)
        logvar = torch.rand(self.z_dim)
        samples = self.model.sample_z(mu, logvar, n=5)
        self.assertEqual(samples.shape, (5, self.z_dim))
        self.assertTrue(torch.is_tensor(samples))

    def test_KLD_gaussian(self):
        self.initialize()
        self.model.z_mu_all = torch.rand(self.z_dim)
        self.model.z_logvar_all = torch.rand(self.z_dim)
        self.model.z_mu_context = torch.rand(self.z_dim)
        self.model.z_logvar_context = torch.rand(self.z_dim)
        kld = self.model.KLD_gaussian()
        self.assertGreaterEqual(kld.item(), 0)
        self.assertTrue(torch.is_tensor(kld))

    def test_data_to_z_params(self):
        self.initialize()
        x = torch.rand(10, self.x_dim)
        y = torch.rand(10, self.y_dim)
        mu, logvar = self.model.data_to_z_params(x, y)
        self.assertEqual(mu.shape, (self.z_dim,))
        self.assertEqual(logvar.shape, (self.z_dim,))
        self.assertTrue(torch.is_tensor(mu))
        self.assertTrue(torch.is_tensor(logvar))

    def test_forward(self):
        self.initialize()
        x_t = torch.rand(5, self.x_dim)
        x_c = torch.rand(10, self.x_dim)
        y_c = torch.rand(10, self.y_dim)
        y_t = torch.rand(5, self.y_dim)
        output = self.model(x_t, x_c, y_c, y_t)
        self.assertEqual(output.shape, (5, self.y_dim))

    def test_random_split_context_target(self):
        self.initialize()
        x_c, y_c, x_t, y_t = self.model.random_split_context_target(
            self.x_data[:, 0], self.y_data, 20, 0
        )
        self.assertEqual(x_c.shape[0], 20)
        self.assertEqual(y_c.shape[0], 20)
        self.assertEqual(x_t.shape[0], 80)
        self.assertEqual(y_t.shape[0], 80)
    
    def test_posterior(self):
        self.initialize()
        x_t = torch.rand(5, self.x_dim)
        x_c = torch.rand(10, self.x_dim)
        y_c = torch.rand(10, self.y_dim)
        y_t = torch.rand(5, self.y_dim)
        output = self.model(x_t, x_c, y_c, y_t)
        posterior = self.model.posterior(x_t, 0.1, 0.01, observation_noise=True)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        mvn = posterior.mvn
        self.assertEqual(mvn.covariance_matrix.size(), (5, 5, 5))
    
    def test_transform_inputs(self):
        self.initialize()
        X = torch.rand(5, 3)
        self.assertTrue(torch.equal(self.model.transform_inputs(X), X.to(device)))
    

if __name__ == "__main__":
    unittest.main()
