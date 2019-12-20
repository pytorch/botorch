import math
import torch

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

X = torch.rand(20, 2) - 0.5
Y = (torch.sin(2 * math.pi * X[:, 0]) + torch.cos(2 * math.pi * X[:, 1])).unsqueeze(-1)
Y += 0.1 * torch.randn_like(Y)

gp = SingleTaskGP(X, Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll);

from botorch.acquisition import UpperConfidenceBound

UCB = UpperConfidenceBound(gp, beta=0.1)

import cma
import numpy as np

# get initial condition for CMAES in numpy form
# note that CMAES expects a different shape (no explicit q-batch dimension)
x0 = np.random.rand(2)

# create the CMA-ES optimizer
es = cma.CMAEvolutionStrategy(
    x0=x0,
    sigma0=0.2,
    inopts={'bounds': [0, 1], "popsize": 50},
)

# speed up things by telling pytorch not to generate a compute graph in the background
with torch.no_grad():

    # Run the optimization loop using the ask/tell interface -- this uses 
    # PyCMA's default settings, see the PyCMA documentation for how to modify these
    while not es.stop():
        xs = es.ask()  # as for new points to evaluate
        # convert to Tensor for evaluating the acquisition function
        X = torch.tensor(xs, device=X.device, dtype=X.dtype)
        # evaluate the acquisition function (optimizer assumes we're minimizing)
        Y = - UCB(X.unsqueeze(-2))  # acquisition functions require an explicit q-batch dimension
        y = Y.view(-1).double().numpy()  # convert result to numpy array
        es.tell(xs, y)  # return the result to the optimizer

# convert result back to a torch tensor
best_x = torch.from_numpy(es.best.x).to(X)

best_x
