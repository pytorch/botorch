import torch

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

d = 5

bounds = torch.stack([-torch.ones(d), torch.ones(d)])

train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(50, d)
train_Y = 1 - torch.norm(train_X, dim=-1, keepdim=True)

model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll);

from botorch.acquisition import qExpectedImprovement
from botorch.sampling import IIDNormalSampler

sampler = IIDNormalSampler(num_samples=100, resample=True)
qEI = qExpectedImprovement(model, best_f=train_Y.max(), sampler=sampler)

N = 5
q = 2

from botorch.optim.initializers import initialize_q_batch_nonneg

# generate a large number of random q-batches
Xraw = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(100 * N, q, d)
Yraw = qEI(Xraw)  # evaluate the acquisition function on these q-batches

# apply the heuristic for sampling promising initial conditions
X = initialize_q_batch_nonneg(Xraw, Yraw, N)

# we'll want gradients for the input
X.requires_grad_(True);

# set up the optimizer, make sure to only pass in the candidate set here
optimizer = torch.optim.Adam([X], lr=0.01)
X_traj = []  # we'll store the results

# run a basic optimization loop
for i in range(75):
    optimizer.zero_grad()
    # this performs batch evaluation, so this is an N-dim tensor
    losses = - qEI(X)  # torch.optim minimizes
    loss = losses.sum()
    
    loss.backward()  # perform backward pass
    optimizer.step()  # take a step
    
    # clamp values to the feasible set
    for j, (lb, ub) in enumerate(zip(*bounds)):
        X.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
    
    # store the optimization trajecatory
    X_traj.append(X.detach().clone())
    
    if (i + 1) % 15 == 0:
        print(f"Iteration {i+1:>3}/75 - Loss: {loss.item():>4.3f}")
    
    # use your favorite convergence criterion here...


