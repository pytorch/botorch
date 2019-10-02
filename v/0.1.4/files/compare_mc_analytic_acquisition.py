import torch

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.test_functions import neg_hartmann6
from gpytorch.mlls import ExactMarginalLogLikelihood

train_x = torch.rand(10, 6)
train_obj = neg_hartmann6(train_x).unsqueeze(-1)
model = SingleTaskGP(train_X=train_x, train_Y=train_obj)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)

from botorch.acquisition import ExpectedImprovement

best_value = train_obj.max()
EI = ExpectedImprovement(model=model, best_f=best_value)

from botorch.optim import optimize_acqf

new_point_analytic, _ = optimize_acqf(
    acq_function=EI,
    bounds=torch.tensor([[0.0] * 6, [1.0] * 6]),
    q=1,
    num_restarts=20,
    raw_samples=100,
    options={},
)

new_point_analytic

from botorch.acquisition import qExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler


sampler = SobolQMCNormalSampler(num_samples=500, seed=0, resample=False)        
MC_EI = qExpectedImprovement(
    model, best_f=best_value, sampler=sampler
)
torch.manual_seed(seed=0) # to keep the restart conditions the same
new_point_mc, _ = optimize_acqf(
    acq_function=MC_EI,
    bounds=torch.tensor([[0.0] * 6, [1.0] * 6]),
    q=1,
    num_restarts=20,
    raw_samples=100,
    options={},
)

new_point_mc

torch.norm(new_point_mc - new_point_analytic)

from botorch.gen import get_best_candidates, gen_candidates_torch
from botorch.optim import gen_batch_initial_conditions

resampler = SobolQMCNormalSampler(num_samples=500, seed=0, resample=True)        
MC_EI_resample = qExpectedImprovement(
    model, best_f=best_value, sampler=resampler
)
bounds = torch.tensor([[0.0] * 6, [1.0] * 6])

batch_initial_conditions = gen_batch_initial_conditions(
    acq_function=MC_EI_resample,
    bounds=bounds,
    q=1,
    num_restarts=20,
    raw_samples=100,
)
batch_candidates, batch_acq_values = gen_candidates_torch(
    initial_conditions=batch_initial_conditions,
    acquisition_function=MC_EI_resample,
    lower_bounds=bounds[0],
    upper_bounds=bounds[1],
    optimizer=torch.optim.Adam,
    verbose=False,
    options={"maxiter": 100},
)
new_point_torch_Adam = get_best_candidates(
    batch_candidates=batch_candidates, batch_values=batch_acq_values
).detach()

new_point_torch_Adam

torch.norm(new_point_torch_Adam - new_point_analytic)

batch_candidates, batch_acq_values = gen_candidates_torch(
    initial_conditions=batch_initial_conditions,
    acquisition_function=MC_EI_resample,
    lower_bounds=bounds[0],
    upper_bounds=bounds[1],
    optimizer=torch.optim.SGD,
    verbose=False,
    options={"maxiter": 350},
)
new_point_torch_SGD = get_best_candidates(
    batch_candidates=batch_candidates, batch_values=batch_acq_values
).detach()

new_point_torch_SGD

torch.norm(new_point_torch_SGD - new_point_analytic)


