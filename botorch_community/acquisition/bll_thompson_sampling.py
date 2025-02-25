from typing import Optional

import numpy as np

import scipy

import torch
from torch.func import grad

from botorch_community.models.vblls import AbstractBLLModel

torch.set_default_dtype(torch.float64)


class BLLMaxPosteriorSampling:
    """
    Implements Maximum Posterior Sampling for Bayesian Linear Last (VBLL) models.

    This class provides functionality to sample from the posterior distribution of a BLL model,
    with optional optimization to refine the sampling process.

    Args:
        model (AbstractBLLModel):
            The VBLL model from which posterior samples are drawn. Must be an instance of `AbstractBLLModel`.
        num_restarts (int, optional):
            Number of restarts for optimization-based sampling. Defaults to 10.
        bounds (torch.Tensor, optional):
            Tensor of shape (2, num_inputs) specifying the lower and upper bounds for sampling.
            If None, defaults to [(0, 1)] for each input dimension.
        discrete (bool, optional):
            If True, assumes the input space is discrete. Defaults to False.

    Raises:
        ValueError:
            If the provided `model` is not an instance of `AbstractBLLModel`.

    Notes:
        - If `bounds` is not provided, the default range [0,1] is assumed for each input dimension.
        - The lower (`lb`) and upper (`ub`) bounds are stored as CPU tensors for compatibility
          with initial condition generation and `scipy.optimize.minimize`.
    """

    def __init__(
        self,
        model,
        num_restarts: int = 10,
        bounds: Optional[torch.Tensor] = None,
        discrete: bool = False,
    ):
        if not isinstance(model, AbstractBLLModel):
            raise ValueError("Model must be an instance of AbstractBLLModel")

        self.model = model
        self.device = model.device
        self.discrete = discrete
        self.num_restarts = num_restarts

        if bounds is None:
            # Default bounds [0,1] for each input dimension
            self.bounds = [(0, 1)] * self.model.num_inputs
            self.lb = torch.zeros(self.model.num_inputs).cpu()
            self.ub = torch.ones(self.model.num_inputs).cpu()
        else:
            # Ensure bounds are on CPU for compatibility with scipy.optimize.minimize
            self.lb = bounds[0, :].cpu()
            self.ub = bounds[1, :].cpu()
            self.bounds = [tuple(bound) for bound in bounds.T.cpu().tolist()]

    def __call__(self, X_cand: torch.Tensor = None, num_samples: int = 1):
        if self.discrete and X_cand is None:
            raise ValueError("X_cand must be provided if `discrete` is True.")

        sampled_functions = [self.model.sample() for _ in range(num_samples)]
        X_next = torch.empty(num_samples, self.model.num_inputs, device=self.device)

        # get max of sampled functions at candidate points for each function
        for i, f in enumerate(sampled_functions):
            # Note that optimization overwrites sampling-based approach
            if not self.discrete:
                X_cand = torch.empty(
                    self.num_restarts, self.model.num_inputs, device=self.device
                )
                Y_cand = torch.empty(
                    self.num_restarts, self.model.num_outputs, device=self.device
                )

                # create numpy wrapper around the sampled function, note we aim to maximize
                def func(x):
                    return (
                        -f(torch.from_numpy(x).to(self.device)).detach().cpu().numpy()
                    )

                grad_f = grad(lambda x: f(x).mean())

                def grad_func(x):
                    return (
                        -grad_f(torch.from_numpy(x).to(self.device))
                        .detach()
                        .cpu()
                        .numpy()
                    )

                for j in range(self.num_restarts):
                    # generate initial condition from within the bounds, necessary for TuRBO
                    x0 = np.random.rand(self.model.num_inputs)
                    x0 = self.lb + (self.ub - self.lb) * x0

                    # optimize sample path
                    res = scipy.optimize.minimize(
                        func,
                        x0,
                        jac=grad_func,
                        bounds=self.bounds,
                        method="L-BFGS-B",
                    )

                    if not res.success:
                        print(f"Optimization failed with message: {res.message}")

                    # store the candidate
                    X_cand[j, :] = torch.from_numpy(res.x)
                    Y_cand[j] = torch.tensor([-res.fun])

                # select the best candidate
                X_next[i, :] = X_cand[Y_cand.argmax()]
            else:
                # sampling based approach on candidate points
                Y_cand = f(X_cand)
                X_next[i, :] = X_cand[Y_cand.argmax()]

        # ensure that the next point is within the bounds, scipy minimize can sometimes return points outside the bounds
        X_next = torch.clamp(X_next, self.lb.to(self.device), self.ub.to(self.device))
        return X_next
