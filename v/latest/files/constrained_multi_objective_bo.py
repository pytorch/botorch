#!/usr/bin/env python3
# coding: utf-8

# ## Constrained, Parallel, Multi-Objective BO in BoTorch with qEHVI and qParEGO
# 
# In this tutorial, we illustrate how to implement a constrained multi-objective (MO) Bayesian Optimization (BO) closed loop in BoTorch.
# 
# We use the parallel ParEGO ($q$ParEGO) [1] and parallel Expected Hypervolume Improvement ($q$EHVI) [1]  acquisition functions to optimize a synthetic C2-DTLZ2 test function with $M=2$ objectives, $V=1$ constraint, and $d=12$ parameters. The two objectives are
# $$f_1(\mathbf x) = (1+ g(\mathbf x_M))\cos\big(\frac{\pi}{2}x_1\big)$$
# $$f_2(\mathbf x) = (1+ g(\mathbf x_M))\sin\big(\frac{\pi}{2}x_1\big)$$
# where $g(\mathbf x) = \sum_{x_i \in \mathbf x_M} (x_i - 0.5)^2, \mathbf x \in [0,1]^d,$ and $\mathbf x_M$ represents the last $d - M +1$ elements of $\mathbf x$. Additionally, the C2-DTLZ2 problem uses the following constraint:
# 
# $$c(\mathbf x) = - \min \bigg[\min_{i=1}^M\bigg((f_i(\mathbf x) -1 )^2 + \sum_{j=1, j=i}^M (f_j^2 - r^2) \bigg), \bigg(\sum_{i=1}^M \big((f_i(\mathbf x) - \frac{1}{\sqrt{M}})^2 - r^2\big)\bigg)\bigg]\geq 0$$
# 
# where $\mathbf x \in [0,1]^d$ and $r=0.2$. 
# 
# The goal here is to *minimize* both objectives. Since BoTorch assumes maximization, we maximize the negative of each objective. Since there typically is no single best solution in multi-objective optimization problems, we seek to find the pareto frontier, the set of optimal trade-offs where improving one metric means deteriorating another.
# 
# [1] [S. Daulton, M. Balandat, and E. Bakshy. Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020.](https://arxiv.org/abs/2006.05078)

# ### Set dtype and device
# Note: $q$EHVI aggressively exploits parallel hardware and is much faster when run on a GPU. See [1] for details.

# In[1]:


import torch
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


# ### Problem setup
# 

# In[2]:


from botorch.test_functions.multi_objective import C2DTLZ2
d = 12
M = 2
problem = C2DTLZ2(dim=d, num_objectives=M, negate=True).to(**tkwargs)


# #### Model initialization
# 
# We use a multi-output `SingleTaskGP` to model the two objectives with a homoskedastic Gaussian likelihood with an inferred noise level.
# 
# The models are initialized with $2(d+1)=26$ points drawn randomly from $[0,1]^{12}$.

# In[3]:


from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples

def generate_initial_data(n):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds,n=1, q=n, seed=torch.randint(1000000, (1,)).item()).squeeze(0)
    train_obj = problem(train_x)
    # negative values imply feasibility in botorch
    train_con = -problem.evaluate_slack(train_x)
    return train_x, train_obj, train_con
    
def initialize_model(train_x, train_obj, train_con):
    # define models for objective and constraint
    train_y = torch.cat([train_obj, train_con], dim=-1)
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


# #### Define a helper function that performs the essential BO step for $q$EHVI
# The helper function below initializes the $q$EHVI acquisition function, optimizes it, and returns the batch $\{x_1, x_2, \ldots x_q\}$ along with the observed function values. 
# 
# For this example, we'll use a small batch of $q=2$. Passing the keyword argument `sequential=True` to the function `optimize_acqf`specifies that candidates should be optimized in a sequential greedy fashion (see [1] for details why this is important). A simple initialization heuristic is used to select the 20 restart initial locations from a set of 1024 random points. Multi-start optimization of the acquisition function is performed using LBFGS-B with exact gradients computed via auto-differentiation.
# 
# **Reference Point**
# 
# $q$EHVI requires specifying a reference point, which is the lower bound on the objectives used for computing hypervolume. In this tutorial, we assume the reference point is known. In practice the reference point can be set 1) using domain knowledge to be slightly worse than the lower bound of objective values, where the lower bound is the minimum acceptable value of interest for each objective, or 2) using a dynamic reference point selection strategy.
# 
# **Partitioning the Non-dominated Space into disjoint rectangles**
# 
# $q$EHVI requires partitioning the non-dominated space into disjoint rectangles (see [1] for details). 
# 
# *Note:* `NondominatedPartitioning` *will be very slow when 1) there are a lot of points on the pareto frontier and 2) there are >3 objectives.*

# In[6]:


from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex

BATCH_SIZE = 2
standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
standard_bounds[1] = 1


def optimize_qehvi_and_get_observation(model, train_obj, train_con, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # compute feasible observations
    is_feas = (train_con <= 0).all(dim=-1)
    # compute points that are better than the known reference point
    better_than_ref = (train_obj > problem.ref_point).all(dim=-1)
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(
        num_outcomes=problem.num_objectives, 
        # use observations that are better than the specified reference point and feasible
        Y=train_obj[better_than_ref & is_feas],  
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),  # use known reference point 
        partitioning=partitioning,
        sampler=sampler,
        # define an objective that specifies which outcomes are the objectives
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
        # specify that the constraint is on the last outcome
        constraints=[lambda Z: Z[..., -1]],
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=20,
        raw_samples=1024,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    # observe new values 
    new_x =  unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj = problem(new_x)
    # negative values imply feasibility in botorch
    new_con = -problem.evaluate_slack(new_x)
    return new_x, new_obj, new_con


# #### Define a helper function for creating constrained objectives for $q$ParEGO
# The helper function below initializes a ConstrainedMCObjective for $q$ParEGO. It creates the `objective` which fetches the outcomes required for the scalarized objective and applies the scalarization and the constraints, which are modeled outcomes.
# 

# In[7]:


from botorch.acquisition.objective import ConstrainedMCObjective

def get_constrained_mc_objective(train_obj,train_con,scalarization):
    """Initialize a ConstrainedMCObjective for qParEGO"""
    n_obj = train_obj.shape[-1]
    # assume first outcomes of the model are the objectives, the rest constraints
    def objective(Z):
        return scalarization(Z[..., :n_obj])

    constrained_obj = ConstrainedMCObjective(
        objective=objective,
        constraints=[lambda Z: Z[..., -1]], # index the constraint
    )
    return constrained_obj


# #### Define a helper function that performs the essential BO step for $q$ParEGO
# The helper function below similarly initializes $q$ParEGO, optimizes it, and returns the batch $\{x_1, x_2, \ldots x_q\}$ along with the observed function values. 
# 
# $q$ParEGO uses random augmented chebyshev scalarization with the `qExpectedImprovement` acquisition function. In the parallel setting ($q>1$), each candidate is optimized in sequential greedy fashion using a different random scalarization (see [1] for details).
# 
# To do this, we create a list of `qExpectedImprovement` acquisition functions, each with different random scalarization weights. The `optimize_acqf_list` method sequentially generates one candidate per acquisition function and conditions the next candidate (and acquisition function) on the previously selected pending candidates.

# In[8]:


def optimize_qparego_and_get_observation(model, train_obj, train_con, sampler):
    """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization 
    of the qParEGO acquisition function, and returns a new candidate and observation."""
    acq_func_list = []
    for _ in range(BATCH_SIZE):
        # sample random weights
        weights = sample_simplex(problem.num_objectives, **tkwargs).squeeze()
        # construct augmented Chebyshev scalarization
        scalarization = get_chebyshev_scalarization(weights=weights, Y=train_obj)
        # initialize ConstrainedMCObjective
        constrained_objective = get_constrained_mc_objective(train_obj=train_obj, train_con=train_con, scalarization=scalarization)
        train_y = torch.cat([train_obj, train_con], dim=-1)
        acq_func = qExpectedImprovement(  # pyre-ignore: [28]
            model=model,
            objective=constrained_objective,
            best_f=constrained_objective(train_y).max(),
            sampler=sampler,
        )
        acq_func_list.append(acq_func)
    # optimize
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds=standard_bounds,
        num_restarts=20,
        raw_samples=1024,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values 
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj = problem(new_x)
    # negative values imply feasibility in botorch
    new_con = -problem.evaluate_slack(new_x)
    return new_x, new_obj, new_con


# ### Perform Bayesian Optimization loop with $q$EHVI and $q$ParEGO
# The Bayesian optimization "loop" for a batch size of $q$ simply iterates the following steps:
# 1. given a surrogate model, choose a batch of points $\{x_1, x_2, \ldots x_q\}$
# 2. observe $f(x)$ for each $x$ in the batch 
# 3. update the surrogate model. 
# 
# 
# Just for illustration purposes, we run three trials each of which do `N_BATCH=50` rounds of optimization. The acquisition function is approximated using `MC_SAMPLES=128` samples.
# 
# *Note*: Running this may take a little while.

# In[9]:


from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
import time

import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

N_TRIALS = 3
N_BATCH = 50
MC_SAMPLES = 128

verbose = False

hvs_qparego_all, hvs_qehvi_all, hvs_random_all = [], [], []

hv = Hypervolume(ref_point=problem.ref_point)

# average over multiple trials
for trial in range(1, N_TRIALS + 1):
    torch.manual_seed(trial)
    
    print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
    hvs_qparego, hvs_qehvi, hvs_random = [], [], []
    
    # call helper functions to generate initial training data and initialize model
    train_x_qparego, train_obj_qparego, train_con_qparego = generate_initial_data(n=2*(d+1))
    mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego, train_con_qparego)
    
    train_x_qehvi, train_obj_qehvi, train_con_qehvi = train_x_qparego, train_obj_qparego, train_con_qparego
    train_x_random, train_obj_random, train_con_random = train_x_qparego, train_obj_qparego, train_con_qparego
    # compute hypervolume 
    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, train_con_qehvi)
    
    # compute pareto front
    is_feas = (train_con_qparego <= 0).all(dim=-1)
    feas_train_obj = train_obj_qparego[is_feas]
    pareto_mask = is_non_dominated(feas_train_obj)
    pareto_y = feas_train_obj[pareto_mask]
    # compute hypervolume
    
    volume = hv.compute(pareto_y)
    
    hvs_qparego.append(volume)
    hvs_qehvi.append(volume)
    hvs_random.append(volume)
    
    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):    
        
        t0 = time.time()
        
        # fit the models
        fit_gpytorch_model(mll_qparego)
        fit_gpytorch_model(mll_qehvi)
        
        # define the qEI and qNEI acquisition modules using a QMC sampler
        qparego_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
        qehvi_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
        
        # optimize acquisition functions and get new observations
        new_x_qparego, new_obj_qparego, new_con_qparego = optimize_qparego_and_get_observation(
            model_qparego, train_obj_qparego, train_con_qparego, qparego_sampler
        )
        new_x_qehvi, new_obj_qehvi, new_con_qehvi = optimize_qehvi_and_get_observation(
            model_qehvi, train_obj_qehvi, train_con_qehvi, qehvi_sampler
        )
        new_x_random, new_obj_random, new_con_random = generate_initial_data(n=BATCH_SIZE)
                
        # update training points
        train_x_qparego = torch.cat([train_x_qparego, new_x_qparego])
        train_obj_qparego = torch.cat([train_obj_qparego, new_obj_qparego])
        train_con_qparego = torch.cat([train_con_qparego, new_con_qparego])

        train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
        train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])
        train_con_qehvi = torch.cat([train_con_qehvi, new_con_qehvi])
    
        train_x_random = torch.cat([train_x_random, new_x_random])
        train_obj_random = torch.cat([train_obj_random, new_obj_random])
        train_con_random = torch.cat([train_con_random, new_con_random])
        

        # update progress
        for hvs_list, train_obj, train_con in zip(
            (hvs_random, hvs_qparego, hvs_qehvi), 
            (train_obj_random, train_obj_qparego, train_obj_qehvi),
            (train_con_random, train_con_qparego, train_con_qehvi),
        ):
            # compute pareto front
            is_feas = (train_con <= 0).all(dim=-1)
            feas_train_obj = train_obj[is_feas]
            pareto_mask = is_non_dominated(feas_train_obj)
            pareto_y = feas_train_obj[pareto_mask]
            # compute feasible hypervolume
            volume = hv.compute(pareto_y)
            hvs_list.append(volume)

        # reinitialize the models so they are ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration
        mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego, train_con_qparego)
        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, train_con_qehvi)
        
        t1 = time.time()
        
        if verbose:
            print(
                f"\nBatch {iteration:>2}: Hypervolume (random, qParEGO, qEHVI) = "
                f"({hvs_random[-1]:>4.2f}, {hvs_qparego[-1]:>4.2f}, {hvs_qehvi[-1]:>4.2f}), "
                f"time = {t1-t0:>4.2f}.", end=""
            )
        else:
            print(".", end="")
   
    hvs_qparego_all.append(hvs_qparego)
    hvs_qehvi_all.append(hvs_qehvi)
    hvs_random_all.append(hvs_random)


# #### Plot the results
# The plot below shows the log feasible hypervolume difference: the log difference between the hypervolume of the true feasible pareto front and the hypervolume of the observed (feasible) pareto front identified by each algorithm. The log feasible hypervolume difference is plotted at each step of the optimization for each of the algorithms. The confidence intervals represent the variance at that step in the optimization across the trial runs. The variance across optimization runs is quite high, so in order to get a better estimate of the average performance one would have to run a much larger number of trials `N_TRIALS` (we avoid this here to limit the runtime of this tutorial). 
# 
# The plot show that $q$EHVI vastly outperforms the $q$ParEGO and Sobol baselines.

# In[10]:


import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def ci(y):
    return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

iters = np.arange(N_BATCH + 1) * BATCH_SIZE
log_hv_difference_qparego = np.log10(problem.max_hv - np.asarray(hvs_qparego_all))
log_hv_difference_qehvi = np.log10(problem.max_hv - np.asarray(hvs_qehvi_all))
log_hv_difference_rnd = np.log10(problem.max_hv - np.asarray(hvs_random_all))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.errorbar(iters, log_hv_difference_rnd.mean(axis=0), yerr=ci(log_hv_difference_rnd), label="Sobol", linewidth=1.5)
ax.errorbar(iters, log_hv_difference_qparego.mean(axis=0), yerr=ci(log_hv_difference_qparego), label="qParEGO", linewidth=1.5)
ax.errorbar(iters, log_hv_difference_qehvi.mean(axis=0), yerr=ci(log_hv_difference_qehvi), label="qEHVI", linewidth=1.5)
ax.set(xlabel='number of observations (beyond initial points)', ylabel='Log Hypervolume Difference')
ax.legend(loc="lower right")


# #### plot the observations colored by iteration
# 
# To examine optimization process from another perspective, we plot the collected observations under each algorithm where the color corresponds to the BO iteration at which the point was collected. The plot on the right for $q$EHVI shows that the $q$EHVI quickly identifies the pareto front and most of its evaluations are very close to the pareto front. $q$ParEGO also identifies has many observations close to the pareto front, but relies on optimizing random scalarizations, which is a less principled way of optimizing the pareto front compared to $q$EHVI, which explicitly attempts focuses on improving the pareto front. Sobol generates random points and has few points close to the pareto front

# In[11]:


from matplotlib.cm import ScalarMappable
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
algos = ["Sobol", "qParEGO", "qEHVI"]
cm = plt.cm.get_cmap('viridis')

batch_number = torch.cat([torch.zeros(2*(d+1)), torch.arange(1, N_BATCH+1).repeat(BATCH_SIZE, 1).t().reshape(-1)]).numpy()
for i, train_obj in enumerate((train_obj_random, train_obj_qparego, train_obj_qehvi)):
    sc = axes[i].scatter(train_obj[:, 0].cpu().numpy(), train_obj[:,1].cpu().numpy(), c=batch_number, alpha=0.8)
    axes[i].set_title(algos[i])
    axes[i].set_xlabel("Objective 1")
    axes[i].set_xlim(-2.5, 0)
    axes[i].set_ylim(-2.5, 0)
axes[0].set_ylabel("Objective 2")
norm = plt.Normalize(batch_number.min(), batch_number.max())
sm =  ScalarMappable(norm=norm, cmap=cm)
sm.set_array([])
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title("Iteration")


# In[ ]:




