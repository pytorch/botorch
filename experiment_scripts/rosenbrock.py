import os
import math
import torch

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qKnowledgeGradient
from botorch.optim import optimize_acqf
from botorch.utils.sampling import manual_seed
from botorch.acquisition import PosteriorMean
from botorch.test_functions import Rosenbrock
from botorch.utils.transforms import unnormalize

SMOKE_TEST = os.environ.get("SMOKE_TEST")

#defining box optimisation bounds
bounds = torch.stack([torch.zeros(2), torch.ones(2)])

#generate XY data
train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(20, 2)
fun = Rosenbrock(dim=2, negate=True)
fun.bounds[0, :].fill_(-5)
fun.bounds[1, :].fill_(10)


def eval_objective(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return fun(unnormalize(x, fun.bounds)).unsqueeze(-1)

train_Y = standardize(eval_objective(train_X))


#fit GP model
model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)

#Defining the qKnowledgeGradient acquisition function (One-Shot KG)
NUM_FANTASIES = 128
qKG = qKnowledgeGradient(model, num_fantasies=NUM_FANTASIES)

#Optimize acquisition function
NUM_RESTARTS = 10
RAW_SAMPLES = 512


with manual_seed(1234):
    candidates, acq_value = optimize_acqf(
        acq_function=qKG,
        bounds=bounds,
        q=2,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

print("candidates: ", candidates, "acq_value ", acq_value)