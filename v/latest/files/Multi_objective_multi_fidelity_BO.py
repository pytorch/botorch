#!/usr/bin/env python3
# coding: utf-8

# # Simultaneous multi-objective multi-fidelity optimization  

# In this tutorial notebook we demonstrate how to perform multi-objective multi-fidelity (MOMF) optimization in BoTORCH as described in [1]. The main concept in MOMF is introducing a 'fidelity objective' that is optimized along with the problem objectives. This fidelity objective can be thought of as a trust objective that rewards the optimization when going to higher fidelity. This emulates a real world scenario where high fidelity data may sometime yield similar values for other objectives but is still considered more trustworthy. Thus the MOMF explicitly optimizes for getting more trustworthy data while taking into account the higher computational costs associated with it.
# 
# We will optimize a synthetic function that is a modified multi-fidelity Branin-Currin. This is a 3 x 2 dimensional problem with one of the input dimension being the fidelity. For the MOMF this results in a 3 x 3 optimization since it also takes into account the fidelity objective. In this case the fidelity objective is taken to be a linear function of fidelity, $ f(s)=s$, where $s$ is the fidelity. The MOMF algorithm can accept any discrete or continuous cost functions as an input. In this example, we choose an exponential dependency of the form $C(s)=\exp(s)$. The goal of the optimization is to find the Pareto front, which is a trade-off solution set for Multi-objective problems, at the highest fidelity. 
# 
# In the second part of this tutorial we compare the method with a multi-objective only optimization using qEHVI [2] with q set to 1 (note that MOMF also supports q>1 if the underlying MO acquisition function supports it). The MO-only optimization runs only at the highest fidelity while MOMF makes use of lower fidelity data to estimate the Pareto front at the highest fidelity.
# 
# [1] [Irshad, Faran, Stefan Karsch, and Andreas Döpp. "Expected hypervolume improvement for simultaneous multi-objective and multi-fidelity optimization." arXiv preprint arXiv:2112.13901 (2021).](https://arxiv.org/abs/2112.13901)
# 
# [2] [S. Daulton, M. Balandat, and E. Bakshy. Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020.](https://arxiv.org/abs/2006.05078)

# In[1]:


import os

import matplotlib.pyplot as plt
import numpy as np
import torch


# ### Set dtype and device 
# Setting up the global variable that determine the device to run the optimization. The optimization is much faster when it runs on GPU.

# In[2]:


tkwargs = (
    {  # Tkwargs is a dictionary contaning data about data type and data device
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
)
SMOKE_TEST = os.environ.get("SMOKE_TEST")


# ### Here we define different variables that define the problem setting and optimization settings

# In[3]:


##############################-----------------------------------------------------##############################################

from botorch.test_functions.multi_objective_multi_fidelity import MOMFBraninCurrin

##############################-----------------------------------------------------##############################################
BC = MOMFBraninCurrin(negate=True)
dim_x = BC.dim  # Input Dimension for MO only optimization
dim_yMO = BC.num_objectives  # Output Dimension for MO only optimization
dim_xMO = dim_x - 1  # Input Dimension for MOMF optimization
dim_y = dim_yMO + 1  # Output Dimesnion for MOMF optimization

ref_pointMO = [0] * dim_xMO  # Reference point for MO only Hypervolume calculation
ref_point = [0] * dim_x  # Reference point for MOMF Hypervolume calculation

n_TRIALS = 3
BATCH_SIZE = 2  # For Batch-optimization BATCH_SIZE should be greater than 1
n_BATCH = 5 if SMOKE_TEST else 30  # Number of iterations within one optimization loop
n_INIT = 5  # Number of initial points for MOMF
n_INITMO = 1  # Number of initial points for MO only optimization
MC_SAMPLES = 8 if SMOKE_TEST else 128  # Number of Monte Carlo samples
NUM_RESTARTS = 2 if SMOKE_TEST else 10  # Number of restart points for multi-start optimization
RAW_SAMPLES = 4 if SMOKE_TEST else 512  # Number of raw samples for initial point selection heuristic

standard_boundsMO = torch.tensor(
    [[0.0] * dim_xMO, [1.0] * dim_xMO], **tkwargs
)  # Bounds for MO only optimization
standard_bounds = torch.tensor(
    [[0.0] * dim_x, [1.0] * dim_x], **tkwargs
)  # Bounds for MOMF optimization
verbose = True


# ### Problem Setup 
# The problem as described before is a modified multi-fidelity version of Branin-Currin (BC) function that results in a 3 x 2 problem. A simple fidelity objective is also defined here which is a linear function of the input fidelity. We also design a wrapper function around the BC that takes care of interfacing torch with numpy and appends the fidelity objective with the BC functions.
# 

# In[4]:


def fid_obj(X):
    # A Fidelity Objective that can be thought of as a trust objective. Higher Fidelity simulations are rewarded as being more
    # trustworthy. Here we consider just a linear fidelity objective.
    fid_obj = 1 * X[..., -1]
    return fid_obj


def problem(x, dim_y):
    # Wrapper around the Objective function to take care of fid_obj stacking
    y = BC(x)  # The Branin-Currin is called
    fid = fid_obj(x)  # Getting the fidelity objective values
    fid_out = fid.unsqueeze(-1)
    y_out = torch.cat(
        [y, fid_out], -1
    )  # Concatenating objective values with fid_objective
    return y_out


# ### Helper functions to define Cost 
# 
# The cost_func function returns an exponential cost from the fidelity. The cost_callable is a wrapper around it that takes care of the input output shapes. This is given as a callable function to MOMF that internally divides the hypervolume by cost.

# In[5]:


exp_arg = torch.tensor(4,**tkwargs)


def cost_func(x, A):
    # A simple exponential cost function.
    val = torch.exp(A * x)
    return val


print(
    "Min Cost:", cost_func(0, exp_arg)
)  # Displaying the min cost for this optimization
print(
    "Max Cost:", cost_func(1, exp_arg)
)  # Displaying the max cost for this optimization


def cost_callable(X):
    r"""Wrapper for the cost function that takes care of shaping input and output arrays for interfacing with cost_func.
    This is passed as a callable function to MOMF.

    Args:
        X: A `batch_shape x q x d`-dim Tensor
    Returns:
        Cost `batch_shape x q x m`-dim Tensor of cost generated from fidelity dimension using cost_func.
    """

    cost = cost_func(torch.flatten(X), exp_arg).reshape(X.shape)
    cost = cost[..., [-1]]
    return cost


# ### Model Initialization 
# We use a multi-output SingleTaskGP to model the problem with a homoskedastic Gaussian likelihood with an inferred noise level. 
# The model is initialized with 5 random points where the fidelity dimension of the initial points is sampled from a probability distribution : $p(s)=C(s)^{-1}$ 

# In[6]:


from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


def gen_init_data(dim_x, points, dim_y):
    # Generates training data with Fidelity dimension sampled from a probability distribution that depends on Cost function
    train_x = torch.rand(points, dim_x,**tkwargs)
    fid_samples = torch.linspace(
        0, 1, 101,**tkwargs) # Array from which fidelity values are sampled  
    prob = 1 / cost_func(
        fid_samples, exp_arg)  # Probability calculated from the Cost function
    prob = prob / torch.sum(prob)  # Normalizing
    idx = prob.multinomial(
        num_samples=points,
        replacement=True)  # Generating indices to choose fidelity samples
    train_x[:, -1] = fid_samples[idx]
    train_obj = problem(
        train_x, dim_y)  # Calls the Problem wrapper to generate train_obj
    return train_x, train_obj


def initialize_model(train_x, train_obj):
    # Initializes a SingleTaskGP with Matern 5/2 Kernel and returns the model and its MLL
    model = SingleTaskGP(
        train_x,
        train_obj,
        outcome_transform=Standardize(m=train_obj.shape[-1])
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


# ### Helper function to optimize acquisition function 
# This is a helper function that initializes, optimizes the acquisition function MOMF and returns the new_x and new_obj. The problem is called from within this helper function.
# 
# A simple initialization heuristic is used to select the 20 restart initial locations from a set of 1024 random points. Multi-start optimization of the acquisition function is performed using LBFGS-B with exact gradients computed via auto-differentiation.

# In[7]:


from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.multi_fidelity import MOMF
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import unnormalize


def optimize_MOMF_and_get_obs(
    model,
    train_obj,
    sampler,
    dim_y,
    ref_point,
    standard_bounds,
    BATCH_SIZE,
    cost_call,
):
    """Wrapper to call MOMF and optimizes it in a sequential greedy fashion returning a new candidate and evaluation"""
    partitioning = FastNondominatedPartitioning(
        ref_point=torch.tensor(ref_point,**tkwargs), Y=train_obj
    )
    acq_func = MOMF(
        model=model,
        ref_point=ref_point,  # use known reference point
        partitioning=partitioning,
        sampler=sampler,
        cost_call=cost_call,
    )
    # Optimization
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=standard_bounds)
    new_obj = problem(new_x, dim_y)
    return new_x, new_obj


# ### Running MOMF optimization 
# 
# We run a single trial of 30 iterations to optimize the multi-fidelity versions of the Branin-Currin functions. The optimization loop works in the following sequence. 
# 
# 1. At the start an initial data of 5 random points is generated and a model initialized using this data.
# 2. The models are used to generate an acquisition function that is optimized to select new input parameters
# 3. The objective function is evaluated at the suggested new_x and returns a new_obj.
# 4. The models are updated with the new points and then are used again to make the next prediction.
# 

# In[8]:


# MOMF Opt run
from botorch import fit_gpytorch_mll
from botorch.sampling.samplers import SobolQMCNormalSampler

verbose = True
train_x = torch.zeros(n_INIT + n_BATCH * BATCH_SIZE, dim_x,**tkwargs)  # Intializing train_x to zero
train_obj = torch.zeros(n_INIT + n_BATCH * BATCH_SIZE, dim_y,**tkwargs)  # Intializing train_obj to zero
train_x[:n_INIT, :], train_obj[:n_INIT, :] = gen_init_data(dim_x, n_INIT, dim_y)  # Generate Initial Data
mll, model = initialize_model(train_x[:n_INIT, :], train_obj[:n_INIT, :])  # Initialize Model
for iteration in range(0, n_BATCH):
    # run N_BATCH rounds of BayesOpt after the initial random batch
    fit_gpytorch_mll(mll)  # Fit the model
    momf_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)  # Generate Sampler
    # Updating indices used to store new observations
    lower_index = n_INIT + iteration * BATCH_SIZE
    upper_index = n_INIT + iteration * BATCH_SIZE + BATCH_SIZE
    # optimize acquisition functions and get new observations
    new_x, new_obj = optimize_MOMF_and_get_obs(
        model,
        train_obj[: upper_index, :],
        momf_sampler,
        dim_y,
        ref_point,
        standard_bounds,
        BATCH_SIZE,
        cost_call=cost_callable,
    )
    if verbose:
        print(f"Iteration: {iteration + 1}")
    # Updating train_x and train_obj
    train_x[lower_index : upper_index, :] = new_x
    train_obj[lower_index : upper_index, :] = new_obj
    # reinitialize the models so they are ready for fitting on next iteration
    mll, model = initialize_model(
        train_x[: upper_index, :], train_obj[: upper_index, :]
    )


# ### Result:  Evaluating the Pareto front at the highest fidelity from MOMF
# 
# After the optimization we are interested in evaluating the final Pareto front. For this we train a GP model with the data acquired by the MOMF optimization. After this we generate $10^4$ random test points between between [0,1] with the fidelity dimension set to 1 to approximate the Pareto front. Two helper functions are defined to achieve this objective where one function generates the test data and the other extracts the Pareto front at the highest fidelity for a given training and testing data.
# 
# **Note: This works reasonably well only for lower dimensional search spaces**

# In[9]:


from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning


def gen_test_points(n_points, dim_x):
    # Function to generate random points with fidelity dimension set to 1. Used to evaluate Pareto front from MOMF
    test_x = torch.rand(size=(n_points, dim_x), **tkwargs)
    test_x[:, -1] = 1
    return test_x


def get_pareto(train_x, train_obj, test_x, ref_point):
    # Function that takes in training and testing data with a reference point. It initializes a model and takes out the
    # Posterior mean at the testing points. From these points the non-dominated set is calculated and used to computer
    # the hypervolume.
    mll, model = initialize_model(train_x, train_obj)
    fit_gpytorch_mll(mll)
    with torch.no_grad():
        # Compute posterior mean over outputs at testing data
        means = model.posterior(test_x).mean
    pareto_mask = is_non_dominated(means)  # Calculating Non-dominated points
    pareto_f = means[pareto_mask]  # Saving the Pareto front
    # Computing Hypervolume
    box_decomp = DominatedPartitioning(
        torch.tensor(ref_pointMO,**tkwargs), pareto_f
    )
    hyper_volume = box_decomp.compute_hypervolume().item()
    return hyper_volume, pareto_f


# In[10]:


# Using the above two functions to generate the final Pareto front.
n_points = 10**4
test_x = gen_test_points(n_points, dim_x)
_hv, final_PF = get_pareto(train_x[:, :], train_obj[:, :-1], test_x, ref_point)


# Plotting the final Pareto front.

# In[11]:


fig, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=200)
axes.plot(
    final_PF[:, 0].detach().cpu().numpy(),
    final_PF[:, 1].detach().cpu().numpy(),
    "o",
    markersize=3.5,
    label="MOMF",
)
axes.set_title("Branin-Currin_Pareto Front", fontsize="12")
axes.set_xlabel("Branin", fontsize="10")
axes.set_ylabel("Currin", fontsize="10")
axes.set_xlim(0, 1)
axes.set_ylim(0, 1)
axes.tick_params(labelsize=10)
axes.legend(loc="lower right", fontsize="7", frameon=True, ncol=1)
plt.tight_layout()


# # Comparison of MOMF with single-fidelity multi-objective optimization using qEHVI 
# 
# In this section we draw a comparison of the MOMF with qEHVI. For this purpose we will now run 3 trials of the MOMF and qEHVI with 80 iterations for both. For this we define some helper functions for qEHVI similar to what was done for MOMF. 
# 
# **Note: Most of the material for qEHVI example has been taken from [3]**
# 
# [3] [Constrained, Parallel, Multi-Objective BO in BoTorch with qNEHVI, and qParEGO](https://botorch.org/tutorials/constrained_multi_objective_bo)

# Here we define a wrapper function for single-fidelity multi-objective qEHVI optimization that appends a column of ones (representing highest fidelity) around the Branin-Currin function.

# In[12]:


def problem_MO(X, dim_y):
    # Since MO-only optimization only evaluates at highest fidelity so a column of ones is added to the input to be consistent
    # with the same objective function definition.
    h_fid = torch.ones(*X.shape[:-1], 1, **tkwargs)
    X = torch.cat([X, h_fid], dim=-1)
    y = BC(X)
    return y


# ### Data Initialization qEHVI 
# 
# For qEHVI we initialize with 1 point to keep the initial cost low. We do not aim to make the initial costs the same but for all cases the initial costs of the MOMF are lower when the fidelity is drawn in a probabilistic fashion.

# In[13]:


def gen_init_data_MO(dim_x, points, dim_y):
    # generates random training data.
    train_x = torch.rand(size=(points, dim_x), **tkwargs)
    train_obj = problem_MO(train_x, dim_y)
    return train_x, train_obj


# ### Helper function to optimize acquisition function 
# This is a helper function that initializes and optimizes the acquisition function qEHVI and returns the new_x and new_obj. The problem is called from within this helper function.
# 
# A simple initialization heuristic is used to select the 20 restart initial locations from a set of 1024 random points. Multi-start optimization of the acquisition function is performed using LBFGS-B with exact gradients computed via auto-differentiation.

# In[14]:


from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement


def optimize_MO_and_get_obs(
    model, train_obj, sampler, dim_y, ref_point, standard_bounds, BATCH_SIZE_MO
):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    partitioning = FastNondominatedPartitioning(
        ref_point=torch.tensor(ref_point,**tkwargs), Y=train_obj
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,  # use a known reference point
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE_MO,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=standard_bounds)
    new_obj = problem_MO(new_x, dim_y)
    return new_x, new_obj


# ### Running MOMF and qEHVI optimization for multiple Trials
# 
# We run 5 trials of 30 iterations each to optimize the multi-fidelity versions of the Brannin-Currin functions using MOMF and qEHVI. The Bayesian loop works in the following sequence. 
# 
# 1. At the start of each trial an initial data is generated and a model initialized using this data.
# 2. The models are used to generated an acquisition function that gives us a suggestion for new input parameters
# 3. The objective function is evaluated at the suggested new_x and returns a new_obj.
# 4. The models are updated with the new points and then are used again to make the next prediction.
# 
# **Note: The following code cell would take some time to run**

# In[15]:


# MOMF and qEHVI
verbose = True
train_x = torch.zeros(n_INIT + n_BATCH * BATCH_SIZE, dim_x, n_TRIALS,**tkwargs)          # Intializing train_x to zero
train_obj = torch.zeros(n_INIT + n_BATCH * BATCH_SIZE, dim_y, n_TRIALS,**tkwargs)        # Intializing train_obj to zero

train_xMO = torch.zeros(n_INITMO + n_BATCH * BATCH_SIZE, dim_xMO, n_TRIALS,**tkwargs)    # Intializing train_xMO to zero
train_objMO = torch.zeros(n_INITMO + n_BATCH * BATCH_SIZE, dim_yMO, n_TRIALS,**tkwargs)  # Intializing train_objMO to zero

for trial in range(0, n_TRIALS):
    torch.manual_seed(trial)
    print("***************************************************************************")
    print("Trial is ", trial)
    # Generate Data and initialize model
    train_x[:n_INIT, :, trial], train_obj[:n_INIT, :, trial] = gen_init_data(dim_x, n_INIT, dim_y)
    mll, model = initialize_model(train_x[:n_INIT, :, trial], train_obj[:n_INIT, :, trial])

    train_xMO[:n_INITMO, :, trial], train_objMO[:n_INITMO, :, trial] = gen_init_data_MO(dim_xMO, n_INITMO, dim_yMO)
    mll_MO, model_MO = initialize_model(train_xMO[:n_INITMO, :, trial], train_objMO[:n_INITMO, :, trial])
    for _iter in range(0, n_BATCH):
        # run N_BATCH rounds of BayesOpt after the initial random batch
        fit_gpytorch_mll(mll)

        fit_gpytorch_mll(mll_MO)  # fit the models
        sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
        # Updating indices used to store new observations
        lower_index = n_INIT + _iter * BATCH_SIZE
        upper_index = n_INIT + _iter * BATCH_SIZE + BATCH_SIZE
        
        lower_indexMO = n_INITMO + _iter * BATCH_SIZE
        upper_indexMO = n_INITMO + _iter * BATCH_SIZE + BATCH_SIZE
        # optimize acquisition function and get new observations
        new_x, new_obj = optimize_MOMF_and_get_obs(
            model,
            train_obj[: upper_index, :, trial],
            sampler,
            dim_y,
            ref_point,
            standard_bounds,
            BATCH_SIZE,
            cost_call=cost_callable,
        )

        new_x_MO, new_obj_MO = optimize_MO_and_get_obs(
            model_MO,
            train_objMO[: upper_indexMO, :, trial],
            sampler,
            dim_yMO,
            ref_pointMO,
            standard_boundsMO,
            BATCH_SIZE,
        )
        # Updating train_x and train_obj
        train_x[lower_index : upper_index, :, trial] = new_x
        train_obj[lower_index : upper_index, :, trial] = new_obj
        
        train_xMO[lower_indexMO : upper_indexMO, :, trial] = new_x_MO
        train_objMO[lower_indexMO : upper_indexMO, :, trial] = new_obj_MO
        # reinitialize the models so they are ready for fitting on next _iter
        mll, model = initialize_model(
            train_x[: upper_index, :, trial],
            train_obj[: upper_index, :, trial],
        )

        mll_MO, model_MO = initialize_model(
            train_xMO[: upper_indexMO, :, trial],
            train_objMO[: upper_indexMO, :, trial],
        )
        if verbose:
            print(f"Iteration: {_iter + 1}")


# ### Evaluating the hypervolume for each iteration for MOMF
# 
# As before we are interested in the Pareto front at the highest fidelity after the MOMF. In this section we only show the evolution of the mean hypervolume for both MOMF and qEHVI. The calculation for the MOMF is done in a similar fashion as before but now for each trial and iteration to get the evolution of hypervolume.

# The following code loops over the number of trials and the number of iterations within a single trial. It works in the following steps: 
# 1. At the start of each iteration a model is trained on the MOMF data.
# 2. We generate a posterior mean from the model and calculated the dominated set from these 10000 points.
# 3. This dominated set is used to estimate the hypervolume and then the next iteration is started where the model now takes n+1 data points
# 
# **The test points used here are already generated before for evaluation of the single trial of MOMF**
# 
# **Note: The following cell would take some time**

# In[16]:


# Generating the final Pareto front by fitting a GP to MOMF data and evaluating the
# GP posterior at the highest fidelity with 10000 random points
verbose = True
hv_MOMF = np.zeros(
    (train_obj.shape[0], n_TRIALS)
)  # Array that contains hypervolume for n_TRIALS
for trial in range(0, n_TRIALS):
    # Loop to run over all trials
    print("***************************************************************************")
    print("Trial is ", trial)
    for num, i in enumerate(range(n_INIT, train_obj.shape[0], 1)):
        # Loop to get evolution of hypervolume during a single trial of MOMF optimization.
        hv_MOMF[num, trial], _ = get_pareto(
            train_x[:i, :, trial], train_obj[:i, :-1, trial], test_x, ref_point
        )
        if verbose:
            print(f"Iteration: {num + 1}")


# For the qEHVI the hypervolume calculation is more straightforward. We calculated at each iteration the set of non-dominated points from the training data and use that to estimate the hypervolume.

# In[17]:


# Since the MO data is already at highest fidelity we can calculate the evolution of hypervolume directly from data
verbose = False
hv_MO = np.zeros((train_objMO.shape[0], n_TRIALS))
for trial in range(0, n_TRIALS):
    # Loop to run over all the trials
    print("***************************************************************************")
    print("Trial is ", trial)
    for num, i in enumerate(range(0, train_objMO.shape[0], 1)):
        # Loop to get evolution of hypervolume within a single trial
        pareto_maskMO = is_non_dominated(
            train_objMO[:i, :, trial]
        )  # Calculating Non-dominated points
        box_decomp = DominatedPartitioning(
            torch.tensor(ref_pointMO,**tkwargs),
            train_objMO[:i, :, trial][pareto_maskMO],
        )  # Used to calculate hypervolume
        hv_MO[num, trial] = box_decomp.compute_hypervolume().item()
        if verbose:
            print("Iteration", num)


# In[18]:


# Cost calculation MOMF and MO
cost_single = cost_func(train_x[:, -1], exp_arg)
cost_MOMF = torch.cumsum(cost_single, axis=0)
# Generating ones equal to number of iterations for MO only-optimization
ones = torch.ones(train_objMO.shape[0],**tkwargs)
cost_singleMO = cost_func(ones, exp_arg)
cost_MO = torch.cumsum(cost_singleMO, axis=0)


# In[19]:


# Converting to Numpy arrays for plotting
cost_MOMF = cost_MOMF.cpu().numpy()
cost_MO = cost_MO.cpu().numpy()
true_hv = 0.5235514158034145  # The approximate max hypervolume taken by evaluating the BC function offline with random 50000 points


# For plotting purposes we calculate the cost of both MOMF and MO using the last dimension of the training input data.

# ### Results
# 
# The following plot shows the hypervolume as a percetage of the true hypervolume for both MOMF and qEHVI. The hypervolume is plotted at each iteration on a log scale. We also plot the confidence intervals derived from running multiple trials for the MOMF and MO.
# 

# In[20]:


MOMF_mean = np.mean(hv_MOMF[n_INIT : n_BATCH * BATCH_SIZE, :] / true_hv * 100, axis=1)
MOMF_CI = (
    1.96
    * np.std(hv_MOMF[n_INIT : n_BATCH * BATCH_SIZE, :] / true_hv * 100, axis=1)
    / np.sqrt(n_TRIALS)
)
mean_costMOMF = np.mean(cost_MOMF[n_INIT : n_BATCH * BATCH_SIZE], axis=1)
MO_mean = np.mean(hv_MO[n_INITMO:, :] / true_hv * 100, axis=1)
MO_CI = 1.96 * np.std(hv_MO[n_INITMO:, :] / true_hv * 100, axis=1) / np.sqrt(n_TRIALS)

fig, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=200)
axes.errorbar(mean_costMOMF, MOMF_mean, yerr=MOMF_CI, markersize=3.5, label="MOMF")
axes.errorbar(cost_MO[n_INITMO:], MO_mean, yerr=MO_CI, markersize=3.5, label="EHVI")
axes.set_title("Branin-Currin", fontsize="12")
axes.set_xlabel("Total Cost Log Scale", fontsize="10")
axes.set_ylabel("Hypervolume (%)", fontsize="10")
# axes.set_xlim(1,5000)
axes.set_ylim(0, 100)
axes.tick_params(labelsize=10)
axes.legend(loc="lower right", fontsize="7", frameon=True, ncol=1)
plt.xscale("log")
plt.tight_layout()

