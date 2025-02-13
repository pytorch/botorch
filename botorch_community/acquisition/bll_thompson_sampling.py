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


# class NeuralNetworkProblemWrapper(Problem):
#     def __init__(self, net, input_size=4, output_size=2, maximize=True):
#         super().__init__(
#             n_var=input_size, n_obj=output_size, xl=0, xu=1
#         )  # Define bounds for inputs
#         self.net = net
#         self.maximize = maximize

#     def _evaluate(self, X, out, *args, **kwargs):
#         # Convert the input 'X' (a population of solutions) from numpy to torch tensor
#         X_tensor = torch.Tensor(X)

#         # Perform forward pass through the neural network
#         outputs = self.net(X_tensor).detach().numpy()

#         if self.maximize:
#             # maximize the objective
#             outputs = -outputs

#         # Objective functions can be based on the network's output
#         out["F"] = outputs


# class MultiObjectiveVBLLParetoSampling:
#     def __init__(
#         self,
#         model,
#         Y,
#         ref_point,
#         Y_stats,
#         bounds: torch.Tensor = None,
#         batch_size: int = 1,
#         plot_result: bool = False,
#     ):
#         self.model = model
#         self.device = model.device
#         self.maximize = True
#         self.plot_result = plot_result

#         # need current Y for hypervolume calculation
#         self.Y = Y
#         self.Y_mean = Y_stats["mean"].detach().numpy()
#         self.Y_std = Y_stats["std"].detach().numpy()

#         # need reference point for hypervolume calculation
#         self.ref_point = ref_point

#         self.batch_size = batch_size
#         if bounds is None:
#             self.bounds = [(0, 1)] * self.model.num_inputs
#             self.lb = torch.zeros(self.model.num_inputs).cpu()
#             self.ub = torch.ones(self.model.num_inputs).cpu()
#         else:
#             # need bounds on the cpu for generation of initial conditions and scipy minimize
#             print("WARNING: Bounds are not used in this implementation")
#             # TODO: Add bounds
#             self.lb = bounds[0, :].cpu()
#             self.ub = bounds[1, :].cpu()
#             self.bounds = [tuple(bound) for bound in bounds.T.cpu().tolist()]

#     def __call__(self, X_cand: torch.Tensor = None, num_samples: int = 1):
#         sampled_functions = [self.model.sample() for _ in range(num_samples)]
#         X_next = torch.empty(num_samples, self.model.num_inputs, device=self.device)

#         # get max of sampled functions at candidate points for each function
#         for i, f in enumerate(sampled_functions):
#             # create pymoo problem
#             problem = NeuralNetworkProblemWrapper(
#                 net=f,
#                 input_size=self.model.num_inputs,
#                 output_size=self.model.num_outputs,
#                 maximize=self.maximize,
#             )

#             # Define the algorithm
#             algorithm = NSGA2(pop_size=100)

#             # Run the optimization
#             res = minimize(problem, algorithm, ("n_gen", 200), seed=1, verbose=False)

#             # if we are maximizing, then re-flip the sign of the objective
#             if self.maximize:
#                 res.F = -res.F

#             # iterate over the pareto front and compute the hypervolume for each candidate
#             hv_volume = torch.zeros(len(res.F))
#             for j, candidate in enumerate(res.F):
#                 # stack candidate with current Y
#                 Y_augmented_pareto = torch.cat(
#                     [self.Y, torch.tensor(candidate).unsqueeze(0)], dim=0
#                 )

#                 # create dominated partitioning
#                 bd = DominatedPartitioning(
#                     ref_point=self.ref_point, Y=Y_augmented_pareto
#                 )

#                 # compute hypervolume for candidate
#                 hv_volume[j] = bd.compute_hypervolume().item()

#             # if the hypervolume is the same for all candidates, then select a random candidate of the approximate front
#             if hv_volume.unique().shape[0] == 1:
#                 index = np.random.randint(0, len(res.X))
#                 print(
#                     "All candidates have the same hypervolume, selecting a random candidate"
#                 )
#             else:
#                 # select the best candidate that maximizes the hypervolume
#                 index = hv_volume.argmax().item()
#                 print(
#                     "Selecting the candidate that maximizes the hypervolume, predicted hypervolume: ",
#                     hv_volume.max().item(),
#                 )
#             X_next[i, :] = torch.tensor(res.X[index])

#         # ensure that the next point is within the bounds
#         X_next = torch.clamp(X_next, self.lb.to(self.device), self.ub.to(self.device))
#         return X_next
