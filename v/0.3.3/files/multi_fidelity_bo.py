#!/usr/bin/env python3
# coding: utf-8

# ## Multi-Fidelity BO in BoTorch with Knowledge Gradient
# 
# In this tutorial, we show how to perform multi-fidelity Bayesian optimization (BO) in BoTorch using the Multi-fidelity Knowledge Gradient (qMFKG) acquisition function [1, 2].
# 
# [1] [J. Wu, P.I. Frazier. Continuous-Fidelity Bayesian Optimization with Knowledge Gradient. NIPS Workshop on Bayesian Optimization, 2017.](https://bayesopt.github.io/papers/2017/20.pdf)
# [2] [J. Wu, S. Toscano-Palmerin, P.I. Frazier, A.G. Wilson. Practical Multi-fidelity Bayesian Optimization for Hyperparameter Tuning. Conference on Uncertainty in Artificial Intelligence (UAI), 2019](https://arxiv.org/pdf/1903.04703.pdf)

# ### Set dtype and device

# In[1]:


import torch
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


# ### Problem setup
# 
# We'll consider the Augmented Hartmann multi-fidelity synthetic test problem. This function is a version of the Hartmann6 test function with an additional dimension representing the fidelity parameter; details are in [2]. The function takes the form $f(x,s)$ where $x \in [0,1]^6$ and $s \in [0,1]$. The target fidelity is 1.0, which means that our goal is to solve $\max_x f(x,1.0)$ by making use of cheaper evaluations $f(x,s)$ for $s < 1.0$. In this example, we'll assume that the cost function takes the form $5.0 + s$, illustrating a situation where the fixed cost is $5.0$.

# In[2]:


from botorch.test_functions.multi_fidelity import AugmentedHartmann

problem = AugmentedHartmann(negate=True).to(**tkwargs)


# #### Model initialization
# 
# We use a `SingleTaskMultiFidelityGP` as the surrogate model, which uses a kernel from [2] that is well-suited for multi-fidelity applications.

# In[3]:


from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples

def generate_initial_data(n=16):
    # generate training data
    train_x = torch.rand(n, 7, **tkwargs)
    train_obj = problem(train_x).unsqueeze(-1) # add output dimension
    return train_x, train_obj
    
def initialize_model(train_x, train_obj):
    # define a surrogate model suited for a "training data"-like fidelity parameter
    # in dimension 6, as in [2]
    model = SingleTaskMultiFidelityGP(
        train_x, 
        train_obj, 
        outcome_transform=Standardize(m=1),
        data_fidelity=6
    )   
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


# #### Define a helper function to construct the MFKG acquisition function
# The helper function illustrates how one can initialize a $q$MFKG acquisition function. In this example, we assume that the affine cost is known. We then use the notion of a `CostAwareUtility` in BoTorch to scalarize the competing objectives of information gain and cost. The MFKG acquisition function optimizes the ratio of information gain to cost, which is captured by the `InverseCostWeightedUtility`.
# 
# In order for MFKG to evaluate the information gain, it uses the model to predict the function value at the highest fidelity after conditioning on the observation. This is handled by the `project` argument, which specifies how to transform a tensor `X` to its target fidelity. We use a default helper function called `project_to_target_fidelity` to achieve this.
# 
# An important point to keep in mind: in the case of standard KG, one can ignore the current value and simply optimize the expected maximum posterior mean of the next stage. However, for MFKG, since the goal is optimize information *gain* per cost, it is important to first compute the current value (i.e., maximum of the posterior mean at the target fidelity). To accomplish this, we use a `FixedFeatureAcquisitionFunction` on top of a `PosteriorMean`.

# In[4]:


from botorch import fit_gpytorch_model
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity

bounds = torch.tensor([[0.0] * problem.dim, [1.0] * problem.dim], **tkwargs)
target_fidelities = {6: 1.0}

cost_model = AffineFidelityCostModel(fidelity_weights={6: 1.0}, fixed_cost=5.0)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)


def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

def get_mfkg(model):
    
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=7,
        columns=[6],
        values=[1],
    )
    
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:,:-1],
        q=1,
        num_restarts=10,
        raw_samples=1024,
        options={"batch_limit": 10, "maxiter": 200},
    )
        
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )


# #### Define a helper function that performs the essential BO step
# This helper function optimizes the acquisition function and returns the batch $\{x_1, x_2, \ldots x_q\}$ along with the observed function values. 

# In[5]:


from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
torch.set_printoptions(precision=3, sci_mode=False)

def optimize_mfkg_and_get_observation(mfkg_acqf):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""
    
    X_init = gen_one_shot_kg_initial_conditions(
        acq_function = mfkg_acqf,
        bounds=bounds,
        q=4,
        num_restarts=10,
        raw_samples=512,
    )
    candidates, _ = optimize_acqf(
        acq_function=mfkg_acqf,
        bounds=bounds,
        q=4,
        num_restarts=10,
        raw_samples=512,
        batch_initial_conditions=X_init,
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x).unsqueeze(-1)
    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, cost


# ### Perform a few steps of multi-fidelity BO
# First, let's generate some initial random data and fit a surrogate model.

# In[6]:


train_x, train_obj = generate_initial_data(n=16)


# We can now use the helper functions above to run a few iterations of BO.

# In[7]:


cumulative_cost = 0.0

for _ in range(6):
    mll, model = initialize_model(train_x, train_obj)
    fit_gpytorch_model(mll)
    mfkg_acqf = get_mfkg(model)
    new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    cumulative_cost += cost


# ### Make a final recommendation
# In multi-fidelity BO, there are usually fewer observations of the function at the target fidelity, so it is important to use a recommendation function that uses the correct fidelity. Here, we maximize the posterior mean with the fidelity dimension fixed to the target fidelity of 1.0.

# In[8]:


def get_recommendation(model):
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=7,
        columns=[6],
        values=[1],
    )

    final_rec, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:,:-1],
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
    )
    
    final_rec = rec_acqf._construct_X_full(final_rec)
    
    objective_value = problem(final_rec)
    print(f"recommended point:\n{final_rec}\n\nobjective value:\n{objective_value}")
    return final_rec


# In[9]:


final_rec = get_recommendation(model)
print(f"\ntotal cost: {cumulative_cost}\n")


# ### Comparison to standard EI (always use target fidelity)
# Let's now repeat the same steps using a standard EI acquisition function (note that this is not a rigorous comparison as we are only looking at one trial in order to keep computational requirements low).

# In[10]:


from botorch.acquisition import qExpectedImprovement

def get_ei(model, best_f):
           
    return FixedFeatureAcquisitionFunction(
        acq_function=qExpectedImprovement(model=model, best_f=best_f),
        d=7,
        columns=[6],
        values=[1],
    ) 

def optimize_ei_and_get_observation(ei_acqf):
    """Optimizes EI and returns a new candidate, observation, and cost."""
    
    candidates, _ = optimize_acqf(
        acq_function=ei_acqf,
        bounds=bounds[:,:-1],
        q=4,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
    )
    
    # add the fidelity parameter
    candidates = ei_acqf._construct_X_full(candidates)
    
    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x).unsqueeze(-1)
    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, cost


# In[11]:


cumulative_cost = 0.0

train_x, train_obj = generate_initial_data(n=16)

for _ in range(6):
    mll, model = initialize_model(train_x, train_obj)
    fit_gpytorch_model(mll)
    ei_acqf = get_ei(model, best_f=train_obj.max())
    new_x, new_obj, cost = optimize_ei_and_get_observation(ei_acqf)
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    cumulative_cost += cost


# In[12]:


final_rec = get_recommendation(model)
print(f"\ntotal cost: {cumulative_cost}\n")

