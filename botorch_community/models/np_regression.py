r"""
Neural Process Regression models based on PyTorch models.

References:

.. [Wu2023arxiv]
   Wu, D., Niu, R., Chinazzi, M., Vespignani, A., Ma, Y.-A., & Yu, R. (2023).
   Deep Bayesian Active Learning for Accelerating Stochastic Simulation.
   arXiv preprint arXiv:2106.02770. Retrieved from https://arxiv.org/abs/2106.02770

Contributor: eibarolle
"""

import torch
import torch.nn as nn
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from botorch.acquisition.objective import PosteriorTransform
from typing import Callable, List, Optional, Tuple
from torch.nn import Module
from gpytorch.distributions import MultivariateNormal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Account for different acquisitions

#reference: https://chrisorm.github.io/NGP.html
class MLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int], 
        activation: Callable = nn.Sigmoid,
        init_func: Optional[Callable] = nn.init.normal_
    ) -> None:
        r"""
        A modular implementation of a Multilayer Perceptron (MLP).
        
        Args:
            input_dim: An int representing the total input dimensionality.
            output_dim: An int representing the total encoded dimensionality.
            hidden_dims: A list of integers representing the # of units in each hidden dimension.
            activation: Activation function applied between layers, defaults to nn.Sigmoid.
            init_func: A function initializing the weights, defaults to nn.init.normal_.
        """
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layer = nn.Linear(prev_dim, hidden_dim).to(device)
            if init_func is not None:
                init_func(layer.weight)
            layers.append(layer)
            layers.append(activation())
            prev_dim = hidden_dim

        final_layer = nn.Linear(prev_dim, output_dim).to(device)
        if init_func is not None:
            init_func(final_layer.weight)
        layers.append(final_layer)
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(device))
    

class REncoder(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int],
        activation: Callable = nn.Sigmoid,
        init_func: Optional[Callable] = nn.init.normal_
    ) -> None:
        r"""Encodes inputs of the form (x_i,y_i) into representations, r_i.

        Args:
            input_dim: An int representing the total input dimensionality.
            output_dim: An int representing the total encoded dimensionality.
            hidden_dims: A list of integers representing the # of units in each hidden dimension.
            activation: Activation function applied between layers, defaults to nn.Sigmoid.
            init_func: A function initializing the weights, defaults to nn.init.normal_.
        """
        super().__init__()
        self.mlp = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, activation=activation, init_func=init_func).to(device)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward pass for representation encoder.

        Args:
            inputs: Input tensor

        Returns:
            torch.Tensor: Encoded representations
        """
        return self.mlp(inputs.to(device))

class ZEncoder(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: Callable = nn.Sigmoid,
        init_func: Optional[Callable] = nn.init.normal_
    ) -> None:
        r"""Takes an r representation and produces the mean & standard 
        deviation of the normally distributed function encoding, z.
        
        Args:
            input_dim: An int representing r's aggregated dimensionality.
            output_dim: An int representing z's latent dimensionality.
            hidden_dims: A list of integers representing the # of units in each hidden dimension.
            activation: Activation function applied between layers, defaults to nn.Sigmoid.
            init_func: A function initializing the weights, defaults to nn.init.normal_.
        """
        super().__init__()
        self.mean_net = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, activation=activation, init_func=init_func).to(device)
        self.logvar_net = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, activation=activation, init_func=init_func).to(device)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward pass for latent encoder.

        Args:
            inputs: Input tensor
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Mean of the latent Gaussian distribution.
                - Log variance of the latent Gaussian distribution.
        """
        inputs = inputs.to(device)
        return self.mean_net(inputs), self.logvar_net(inputs)
    
class Decoder(torch.nn.Module):
    def __init__(
        self,
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int],
        activation: Callable = nn.Sigmoid,
        init_func: Optional[Callable] = nn.init.normal_
    ) -> None:
        r"""Takes the x star points, along with a 'function encoding', z, and makes predictions.
        
        Args:
            input_dim: An int representing the total input dimensionality.
            output_dim: An int representing the total encoded dimensionality.
            hidden_dims: A list of integers representing the # of units in each hidden dimension.
            activation: Activation function applied between layers, defaults to nn.Sigmoid.
            init_func: A function initializing the weights, defaults to nn.init.normal_.
        """
        super().__init__()
        self.mlp = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, activation=activation, init_func=init_func).to(device)
        
    def forward(self, x_pred: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        r"""Forward pass for decoder.

        Args:
            x_pred: Input points of shape (n x d_x), representing # of data points by x_dim.
            z: Latent encoding of shape (num_samples x d_z), representing # of samples by z_dim.

        Returns:
            torch.Tensor: Predicted target values of shape (n, z_dim), representing # of data points by z_dim.
        """
        z = z.to(device)
        z_expanded = z.unsqueeze(0).expand(x_pred.size(0), -1).to(device)
        x_pred = x_pred.to(device)
        xz = torch.cat([x_pred, z_expanded], dim=-1)
        return self.mlp(xz)

def MAE(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    r"""Mean Absolute Error loss function.

    Args:
        pred: The predicted values tensor.
        target: The target values tensor.

    Returns:
        torch.Tensor: A tensor representing the MAE.
    """
    loss = torch.abs(pred-target)
    return loss.mean()

class NeuralProcessModel(Model):
    def __init__(
        self,
        r_hidden_dims: List[int], 
        z_hidden_dims: List[int], 
        decoder_hidden_dims: List[int],
        x_dim: int,
        y_dim: int,
        r_dim: int,
        z_dim: int,
        activation: Callable = nn.Sigmoid,
        init_func: Optional[Callable] = torch.nn.init.normal_,
    ) -> None:
        r"""Diffusion Convolutional Recurrent Neural Network Model Implementation.

        Args:
            r_hidden_dims: Hidden Dimensions/Layer list for REncoder
            z_hidden_dims: Hidden Dimensions/Layer list for ZEncoder
            decoder_hidden_dims: Hidden Dimensions/Layer for Decoder
            x_dim: Int dimensionality of input data x.
            y_dim: Int dimensionality of target data y.
            r_dim: Int dimensionality of representation r.
            z_dim: Int dimensionality of latent variable z.
            activation: Activation function applied between layers, defaults to nn.Sigmoid.
            init_func: A function initializing the weights, defaults to nn.init.normal_.
        """
        super().__init__()
        self.r_encoder = REncoder(x_dim+y_dim, r_dim, r_hidden_dims, activation=activation, init_func=init_func).to(device) 
        self.z_encoder = ZEncoder(r_dim, z_dim, z_hidden_dims, activation=activation, init_func=init_func).to(device)
        self.decoder = Decoder(x_dim + z_dim, y_dim, decoder_hidden_dims, activation=activation, init_func=init_func).to(device) 
        self.z_dim = z_dim
        self.z_mu_all = None
        self.z_logvar_all = None
        self.z_mu_context = None
        self.z_logvar_context = None
    
    def data_to_z_params(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        xy_dim: int = 1,
        r_dim: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute latent parameters from inputs as a latent distribution.

        Args:
            x: Input tensor
            y: Target tensor
            xy_dim: Combined Input Dimension as int, defaults as 1
            r_dim: Combined Target Dimension as int, defaults as 0.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                - x_c: Context input data.
                - y_c: Context target data.
                - x_t: Target input data.
                - y_t: Target target data.
        """
        x = x.to(device)
        y = y.to(device)
        xy = torch.cat([x,y], dim=xy_dim).to(device).to(device)
        rs = self.r_encoder(xy)
        r_agg = rs.mean(dim=r_dim).to(device)
        return self.z_encoder(r_agg) 
    
    def sample_z(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        n: int = 1,
        min_std: float = 0.01,
        scaler: float = 0.5
    ) -> torch.Tensor:
        r"""Reparameterization trick for z's latent distribution.

        Args:
            mu: Tensor representing the Gaussian distribution mean.
            logvar: Tensor representing the log variance of the Gaussian distribution.
            n: Int representing the # of samples, defaults to 1.
            min_std: Float representing the minimum possible standardized std, defaults to 0.1.
            scaler: Float scaling the std, defaults to 0.9.

        Returns:
            torch.Tensor: Samples from the Gaussian distribution.
    """
        if min_std <= 0 or scaler <= 0:
            raise ValueError()
        
        shape = [n, self.z_dim]
        if n == 1:
            shape = shape[1:]
        eps = torch.autograd.Variable(logvar.data.new(*shape).normal_()).to(device)
        
        std = min_std + scaler * torch.sigmoid(logvar) 
        std = std.to(device)
        mu = mu.to(device)
        return mu + std * eps

    def KLD_gaussian(
        self,
        min_std: float = 0.1,
        scaler: float = 0.9
    ) -> torch.Tensor:
        r"""Analytical KLD between 2 Gaussian Distributions.

        Args:
            min_std: Float representing the minimum possible standardized std, defaults to 0.1.
            scaler: Float scaling the std, defaults to 0.9.
            
        Returns:
            torch.Tensor: A tensor representing the KLD.
        """
        
        if min_std <= 0 or scaler <= 0:
            raise ValueError()
        std_q = min_std + scaler * torch.sigmoid(self.z_logvar_all).to(device)
        std_p = min_std + scaler * torch.sigmoid(self.z_logvar_context).to(device)
        p = torch.distributions.Normal(self.z_mu_context.to(device), std_p)
        q = torch.distributions.Normal(self.z_mu_all.to(device), std_q)
        return torch.distributions.kl_divergence(p, q).sum()
    
    def posterior(
        self, 
        X: torch.Tensor, 
        covariance_multiplier: float,
        observation_constant: float,
        observation_noise: bool = False, 
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> GPyTorchPosterior:
        r"""Computes the model's posterior distribution for given input tensors.

        Args:
            X: Input Tensor
            covariance_multiplier: Float scaling the covariance.
            observation_constant: Float representing the noise constant.
            observation_noise: Adds observation noise to the covariance if True, defaults to False.
            posterior_transform: An optional posterior transformation, defaults to None.

        Returns:
            GPyTorchPosterior: The posterior distribution object 
            utilizing MultivariateNormal.
        """
        X = X.to(device)
        mean = self.decoder(X.to(device), self.sample_z(self.z_mu_all, self.z_logvar_all))
        covariance = torch.eye(X.size(0)) * covariance_multiplier
        if (observation_noise):
            covariance = covariance + observation_constant
        mvn = MultivariateNormal(mean, covariance)
        posterior = GPyTorchPosterior(mvn)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def transform_inputs(
        self,
        X: torch.Tensor,
        input_transform: Optional[Module] = None,
    ) -> torch.Tensor:
        r"""Transform inputs.

        Args:
            X: A tensor of inputs
            input_transform: A Module that performs the input transformation.

        Returns:
            torch.Tensor: A tensor of transformed inputs
        """
        X = X.to(device)
        if input_transform is not None:
            input_transform.to(X)
            return input_transform(X)
        try:
            return self.input_transform(X)
        except AttributeError:
            return X

    def forward(
        self,
        x_t: torch.Tensor,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        y_t: torch.Tensor,
        input_dim: int = 0,
        target_dim: int = 0
    ) -> torch.Tensor:
        r"""Forward pass for the model.

        Args:
            x_t: Target input data.
            x_c: Context input data.
            y_c: Context target data.
            y_t: Target output data.
            input_dim: Input dimension concatenated
            target_dim: Target dimension concatendated

        Returns:
            torch.Tensor: Predicted target values.
        """
        if any(tensor.numel() == 0 for tensor in [x_t, x_c, y_c]):
            raise ValueError()
        if input_dim not in [0, 1]:
            raise ValueError()
        if x_c.size(1 - input_dim) != x_t.size(1 - input_dim):
            raise ValueError()
        if y_c.size(1 - target_dim) != y_t.size(1 - target_dim):
            raise ValueError()
        
        x_t = x_t.to(device)
        x_c = x_c.to(device)
        y_c = y_c.to(device)
        y_t = y_t.to(device)
        
        self.z_mu_all, self.z_logvar_all = self.data_to_z_params(torch.cat([x_c, x_t], dim = input_dim), torch.cat([y_c, y_t], dim = target_dim))
        self.z_mu_context, self.z_logvar_context = self.data_to_z_params(x_c, y_c)
        z = self.sample_z(self.z_mu_all, self.z_logvar_all)
        return self.decoder(x_t, z)
    
    def random_split_context_target(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_context: int,
        axis: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Helper function to split randomly into context and target.

        Args:
            x: Input data tensor.
            y: Target data tensor.
            n_context (int): Number of context points.
            axis: Dimension axis as int

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                - x_c: Context input data.
                - y_c: Context target data.
                - x_t: Target input data.
                - y_t: Target target data.
        """
        mask = torch.randperm(x.shape[0])[:n_context]
        x_c = torch.from_numpy(x[mask]).to(device)
        y_c = torch.from_numpy(y[mask]).to(device)
        splitter = torch.zeros(x.shape[0], dtype=torch.bool)
        splitter[mask] = True
        x_t = torch.from_numpy(x[~splitter]).to(device)
        y_t = torch.from_numpy(y[~splitter]).to(device)
        return x_c, y_c, x_t, y_t
    