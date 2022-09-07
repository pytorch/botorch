#!/usr/bin/env python3
# coding: utf-8

# # Scalable Constrained Bayesian Optimization (SCBO)
# In this tutorial, we show how to implement Scalable Constrained Bayesian Optimization (SCBO) [1] in a closed loop in BoTorch.
# 
# We optimize the 20𝐷 Ackley function on the domain [−5,10]^20. This implementation uses two simple constraint functions c1 and c2. Our goal is to find values x which maximize Ackley(x) subject to the constraints c1(x) <= 0 and c2(x) <= 0.
# 
# [1]: David Eriksson and Matthias Poloczek. Scalable constrained Bayesian optimization. In International Conference on Artificial Intelligence and Statistics, pages 730–738. PMLR, 2021.
# (https://doi.org/10.48550/arxiv.2002.08526)
# 
# Since SCBO is essentially a constrained version of Trust Region Bayesian Optimization (TuRBO), this tutorial shares much of the same code as the TuRBO Tutorial (https://botorch.org/tutorials/turbo_1) with small modifications made to implement SCBO.

# In[3]:


import math
from dataclasses import dataclass

import torch
from torch import Tensor
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.model_list_gp_regression import ModelListGP

# Constrained Max Posterior Sampling 
# is a new sampling class, similar to MaxPosteriorSampling, 
# which implements the constrained version of Thompson Sampling described in [1]
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
import os
SMOKE_TEST = os.environ.get("SMOKE_TEST")


# ## Demonstration with 20-dimensional Ackley function and Two Simple Constraint Functions

# In[5]:


# Here we define the example 20D Ackley function 
fun = Ackley(dim=20, negate=True).to(dtype=dtype, device=device)
fun.bounds[0, :].fill_(-5)
fun.bounds[1, :].fill_(10)
dim = fun.dim
lb, ub = fun.bounds

batch_size = 4
n_init = 2 * dim
max_cholesky_size = float("inf")  # Always use Cholesky

# When evaluating the function, we must first unnormalize the inputs since 
# we will use normalized inputs x in the main optimizaiton loop
def eval_objective(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return fun(unnormalize(x, fun.bounds))


# ### Defining two simple constraint functions
# 
# #### We'll use two constraints functions: c1 and c2 
# We want to find solutions which maximize the above Ackley objective subject to the constraint that 
# c1(x) <= 0 and c2(x) <= 0 
# Note that SCBO expects all constraints to be of the for c(x) <= 0, so any other desired constraints must be modified to fit this form. 
# 
# Note also that while the below constraints are very simple functions, the point of this tutorial is to show how to use SCBO, and this same implementation could be applied in the same way if c1, c2 were actually complex black-box functions. 
# 

# In[6]:


def c1(x): # Equivalent to enforcing that x[0] >= 0
    return -x[0] 

def c2(x): # Equivalent to enforcing that x[1] >= 0
    return -x[1] 

# We assume c1, c2 have same bounds as the Ackley function above 
def eval_c1(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return c1(unnormalize(x, fun.bounds))

def eval_c2(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return c2(unnormalize(x, fun.bounds))


# ## Define TuRBO Class
# 
# Just as in the TuRBO Tutorial (https://botorch.org/tutorials/turbo_1), we'll define a class to hold the turst region state and a method update_state() to update the side length of the trust region hyper-cube during optimization. We'll update the side length according to the number of sequential successes or failures as discussed in the original TuRBO paper. 

# In[7]:


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = torch.ones(2,)*torch.inf
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

def update_tr_length(state):
    # Update the length of the trust region according to 
    # success and failure counters 
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min: # Restart when trust region becomes too small 
        state.restart_triggered = True
    
    return state


def update_state(state, Y_next, C_next): 
    ''' Method used to update the TuRBO state after each
        step of optimization. 
        
        Success and failure counters are updated accoding to 
        the objective values (Y_next) and constraint values (C_next) 
        of the batch of candidate points evaluated on the optimization step. 
        
        As in the original TuRBO paper, a success is counted whenver 
        any one of the new candidate points imporves upon the incumbent 
        best point. The key difference for SCBO is that we only compare points 
        by their objective values when both points are valid (meet all constraints). 
        If exactly one of the two points beinc compared voliates a constraint, the 
        other valid point is automatically considered to be better. If both points 
        violate some constraints, we compare them inated by their constraint values. 
        The better point in this case is the one with minimum total constraint violation 
        (the minimum sum over constraint values)'''

    # Determine which candidates meet the constraints (are valid)
    bool_tensor = C_next <= 0 
    bool_tensor = torch.all(bool_tensor, dim=-1)
    Valid_Y_next = Y_next[bool_tensor] 
    Valid_C_next = C_next[bool_tensor]
    if Valid_Y_next.numel() == 0: # if none of the candidates are valid
        # pick the point with minimum violation
        sum_violation = C_next.sum(dim=-1)
        min_violation = sum_violation.min()
        # if the minimum voilation candidate is smaller than the violation of the incumbent
        if min_violation < state.best_constraint_values.sum():
            # count a success and update the current best point and constraint values
            state.success_counter += 1
            state.failure_counter = 0
            # new best is min violator
            state.best_value = Y_next[sum_violation.argmin()].item()
            state.best_constraint_values = C_next[sum_violation.argmin()]
        else:
            # otherwise, count a failure
            state.success_counter = 0
            state.failure_counter += 1
    else: # if at least one valid candidate was suggested, 
          # throw out all invalid candidates 
          # (a valid candidate is always better than an invalid one)

        # Case 1: if best valid candidate found has a higher obj value that incumbent best
            # count a success, the obj valuse has been improved
        imporved_obj = max(Valid_Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value)
        # Case 2: if incumbent best violates constraints
            # count a success, we now have suggested a point which is valid and therfore better
        obtained_validity = torch.all(state.best_constraint_values > 0)
        if imporved_obj or obtained_validity: # If Case 1 or Case 2
            # count a success and update the best value and constraint values 
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = max(Valid_Y_next).item()
            state.best_constraint_values = Valid_C_next[Valid_Y_next.argmax()]
        else:
            # otherwise, count a fialure
            state.success_counter = 0
            state.failure_counter += 1

    # Finally, update the length of the trust region according to the
    # updated success and failure counts
    state = update_tr_length(state) 

    return state   


# Define example state 
state = TurboState(dim=dim, batch_size=batch_size)
print(state)


# ### Generate Initial Points
# 
# Here we define a simple method to generate a set of random initial datapoints that we will use to kick-off optimization. 

# In[8]:


def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init


# ### Generating a batch of candidates for SCBO 
# 
# Just as in the TuRBO Tutorial (https://botorch.org/tutorials/turbo_1), we'll define a method generate_batch to generate a new batch of candidate points within the TuRBO trust region using thompson sampling. 
# 
# The key difference here from TuRBO is that, instead of using MaxPosteriorSampling to simply grab the candidates within the trust region with the maximum posterior values, we use ConstrainedMaxPosteriorSampling to instead grab the candidates within the trust region with the maximum posterior values subject to the constrain that the posteriors for the constraint models for c1(x) and c2(x) must be less than or equal to 0 for each candidate. In otherwords, we use additional GPs ('constraiant models') to model each black-box constraint (c1 and c2), and throw out all candidates for which the posterior prediction of these constraint models is greater than 0 (throw out all predicted constraint violators). According to [1], in the special case when all of the candidaates arae predicted to be constraint violators, we select the candidate with the minimum predicted violation. (See botorch.generation.sampling.ConstrainedMaxPosteriorSampling for implementation details).

# In[9]:


def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    constraint_model=None
):
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    # Thompson Sampling w/ Constraints (SCBO) 
    dim = X.shape[-1]
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = (
        torch.rand(n_candidates, dim, dtype=dtype, device=device)
        <= prob_perturb
    )
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    # Create candidate points from the perturbations and the mask        
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Sample on the candidate points using Constrained Max Posterior Sampling
    constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(model=model, constraint_model=constraint_model, replacement=False)
    with torch.no_grad():
        X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

    return X_next


# ## Main Optimization Loop

# In[10]:


# Get initial data 
# Must get initial values for both objective and constraints 
train_X = get_initial_points(dim, n_init)
train_Y = torch.tensor(
    [eval_objective(x) for x in train_X], dtype=dtype, device=device
).unsqueeze(-1)
C1 = torch.tensor(
    [eval_c1(x) for x in train_X], dtype=dtype, device=device
).unsqueeze(-1)
C2 = torch.tensor(
    [eval_c2(x) for x in train_X], dtype=dtype, device=device
).unsqueeze(-1)

C1.min(), C1.max(), C2.min(), C2.max(), train_Y.min(), train_Y.max()


# In[11]:


# Initialize TuRBO state 
from botorch.models.transforms.outcome import Standardize 
state = TurboState(dim, batch_size=batch_size)
N_CANDIDATES = min(5000, max(2000, 200 * dim)) if not SMOKE_TEST else 4


def get_fitted_model(X,Y):
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
        )
        model = SingleTaskGP(X, Y, covar_module=covar_module, 
                            likelihood=likelihood, outcome_transform=Standardize(m=1) )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            fit_gpytorch_model(mll)

        return model


while not state.restart_triggered:  # Run until TuRBO converges
    # Fit GP models for objective and constraints 
    model = get_fitted_model(train_X,train_Y)
    c1_model = get_fitted_model(train_X,C1)
    c2_model = get_fitted_model(train_X,C2)
    
    # Generate a batch of candidates
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        X_next = generate_batch(
            state=state,
            model=model,
            X=train_X,
            Y=train_Y,
            batch_size=batch_size,
            n_candidates=N_CANDIDATES,
            constraint_model=ModelListGP(c1_model, c2_model)    
            )

    # Evaluate both the objective and constraints for the selected candidaates 
    Y_next = torch.tensor(
        [eval_objective(x) for x in X_next], dtype=dtype, device=device
    ).unsqueeze(-1) 

    C1_next = torch.tensor(
        [eval_c1(x) for x in X_next], dtype=dtype, device=device
    ).unsqueeze(-1) 

    C2_next = torch.tensor( 
        [eval_c2(x) for x in X_next], dtype=dtype, device=device
    ).unsqueeze(-1) 

    C_next = torch.cat([C1_next, C2_next], dim=-1) 

    # Update TuRBO state
    state = update_state(state, Y_next, C_next)

    # Append data
    #   Notice we append all data, even points that violate
    #   the constriants, this is so our constraint models
    #   can learn more about the constranit functions and 
    #   gain confidence about where violation occurs
    train_X = torch.cat((train_X, X_next), dim=0)
    train_Y = torch.cat((train_Y, Y_next), dim=0)
    C1 = torch.cat((C1, C1_next), dim=0)
    C2 = torch.cat((C2, C2_next), dim=0)

    # Print current status 
    #   Note: state.best_value is always the best objective value
    #   found so far which meets the constraints, or in the case
    #   that no points have been found yet which meet the constraints,
    #   it is the objective value of the point with the 
    #   minimum constraint violation
    print(
        f"{len(train_X)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
    )


# In[12]:


#  Valid samples must have BOTH c1 <= 0 and c2 <= 0  
constraint_vals = torch.cat([C1, C2], dim=-1) 
bool_tensor = constraint_vals <= 0 
bool_tensor = torch.all(bool_tensor, dim=-1).unsqueeze(-1) 
Valid_Y = train_Y[bool_tensor] 

print(f"With constraints, the best value we found is: {Valid_Y.max().item():.4f}") 


# ### Plot Results
# 
# Notice that with these two simple constraints, SCBO preforms about the same as to TuRBO (see TuRBO 1 tutorial notebok)

# In[13]:


# Plot Optimization Results

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(8, 6))

fx = np.maximum.accumulate(train_Y.cpu())
plt.plot(fx, marker="", lw=3)

plt.plot([0, len(train_Y)], [fun.optimal_value, fun.optimal_value], "k--", lw=3)
plt.ylabel("Function value", fontsize=18)
plt.xlabel("Number of evaluations", fontsize=18)
plt.title("20D Ackley", fontsize=24)
plt.xlim([0, len(train_Y)])
plt.ylim([-15, 1])

plt.grid(True)
plt.show()


# In[ ]:




