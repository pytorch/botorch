#!/usr/bin/env python
# coding: utf-8

# ## Analytic and MC-based Expected Improvement (EI) acquisition
# 
# In this tutorial, we compare the analytic and MC-based EI acquisition functions and show both `scipy`- and `torch`-based optimizers for optimizing the acquisition. This tutorial highlights the modularity of botorch and the ability to easily try different acquisition functions and accompanying optimization algorithms on the same fitted model.

# ### Comparison of analytic and MC-based EI

# In[1]:


import torch

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.test_functions import neg_hartmann6
from gpytorch.mlls import ExactMarginalLogLikelihood


# First, we generate some random data and fit a SingleTaskGP for a 6-dimensional synthetic test function 'Hartmann6'.

# In[2]:


train_x = torch.rand(10, 6)
train_obj = neg_hartmann6(train_x)
model = SingleTaskGP(train_X=train_x, train_Y=train_obj)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll);


# Initialize an analytic EI acquisition function on the fitted model.
# 

# In[3]:


from botorch.acquisition import ExpectedImprovement

best_value = train_obj.max()
EI = ExpectedImprovement(model=model, best_f=best_value)


# Next, we optimize the analytic EI acquisition function using 50 random restarts chosen from 100 initial raw samples.

# In[4]:


from botorch.optim import joint_optimize

new_point_analytic = joint_optimize(
    acq_function=EI,
    bounds=torch.tensor([[0.0] * 6, [1.0] * 6]),
    q=1,
    num_restarts=20,
    raw_samples=100,
    options={},
)


# In[5]:


new_point_analytic


# Now, let's swap out the analytic acquisition function and replace it with an MC version. Note that we are in the `q = 1` case; for `q > 1`, an analytic version does not exist.

# In[6]:


from botorch.acquisition import qExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler


sampler = SobolQMCNormalSampler(num_samples=500, seed=0, resample=False)        
MC_EI = qExpectedImprovement(
    model, best_f=best_value, sampler=sampler
)
torch.manual_seed(seed=0) # to keep the restart conditions the same
new_point_mc = joint_optimize(
    acq_function=MC_EI,
    bounds=torch.tensor([[0.0] * 6, [1.0] * 6]),
    q=1,
    num_restarts=20,
    raw_samples=100,
    options={},
)


# In[7]:


new_point_mc


# Check that the two generated points are close.

# In[8]:


torch.norm(new_point_mc - new_point_analytic)


# ### Using a torch optimizer on a stochastic acquisition function
# We could also optimize using a `torch` optimizer. This is particularly useful for the case of a stochastic acquisition function, which we can obtain by setting `resample=True`. First, we illustrate the usage of `torch.optim.Adam`. In the code snippet below, `gen_batch_initial_candidates` uses a heuristic to select a set of restart locations, `gen_candidates_torch` is a wrapper to the `torch` optimizer for maximizing the acquisition value, and `get_best_candidates` finds the best result amongst the random restarts.

# In[9]:


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


# In[10]:


new_point_torch_Adam


# In[11]:


torch.norm(new_point_torch_Adam - new_point_analytic)


# By changing the `optimizer` parameter to `gen_candidates_torch`, we can also try `torch.optim.SGD`. Note that we are allowing `SGD` more iterations than `Adam` to find the best point.

# In[12]:


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


# In[13]:


new_point_torch_SGD


# In[14]:


torch.norm(new_point_torch_SGD - new_point_analytic)

