#!/usr/bin/env python3
# coding: utf-8

# ## Meta-Learning with the Rank-Weighted GP Ensemble (RGPE)
# 
# BoTorch is designed in to be model-agnostic and only requries that a model conform to a minimal interface. This tutorial walks through an example of implementing the rank-weighted Gaussian process ensemble (RGPE) [Feurer, Letham, Bakshy ICML 2018 AutoML Workshop] and using the RGPE in BoTorch to do meta-learning across related optimization tasks.
# 
# * Original paper: https://arxiv.org/pdf/1802.02219.pdf

# In[1]:


import os
import torch
import math


torch.manual_seed(29)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")


# ### Toy Problem
# * We consider optimizing the following 1-D synthetic function
# $$f(x, s_i) = \frac{1}{10}\bigg(x-1\bigg)\bigg(\sin(x+s_i)+\frac{1}{10}\bigg)$$
# where
# $$s_i = \frac{(i+9)\pi}{8}$$
# is a task-dependent shift parameter and $i$ is the task index $i \in [1, t]$.
# 
# * In this tutorial, we will consider the scenario where we have collected data from 5 prior tasks (referred to as base tasks), which with a different task dependent shift parameter $s_i$.
# 
# * The goal now is use meta-learning to improve sample efficiency when optimizing a 6th task.

# #### Toy Problem Setup
# 
# First let's define a function for compute the shift parameter $s_i$ and set the shift amount for the target task.

# In[2]:


NUM_BASE_TASKS = 5 if not SMOKE_TEST else 2


def task_shift(task):
    """
    Fetch shift amount for task.
    """
    return math.pi * task / 12.0


# set shift for target task

TARGET_SHIFT = 0.0


# Then, let's define our function $f(x, s_i)$ and set bounds on $x$.

# In[3]:


BOUNDS = torch.tensor([[-10.0], [10.0]], dtype=dtype, device=device)


def f(X, shift=TARGET_SHIFT):
    """
    Torch-compatible objective function for the target_task
    """
    f_X = X * torch.sin(X + math.pi + shift) + X / 10.0
    return f_X


# #### Sample training data for prior base tasks

# We sample data from a Sobol sequence to help ensure numerical stability when using a small amount of 1-D data. Sobol sequences help prevent us from sampling a bunch of training points that are close together.

# In[4]:


from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize


noise_std = 0.05

# Sample data for each base task
data_by_task = {}
for task in range(NUM_BASE_TASKS):
    num_training_points = 20
    # draw points from a sobol sequence
    raw_x = draw_sobol_samples(
        bounds=BOUNDS,
        n=num_training_points,
        q=1,
        seed=task + 5397923,
    ).squeeze(1)
    # get observed values
    f_x = f(raw_x, task_shift(task + 1))
    train_y = f_x + noise_std * torch.randn_like(f_x)
    train_yvar = torch.full_like(train_y, noise_std**2)
    # store training data
    data_by_task[task] = {
        # scale x to [0, 1]
        "train_x": normalize(raw_x, bounds=BOUNDS),
        "train_y": train_y,
        "train_yvar": train_yvar,
    }


# #### Let's plot the base tasks and the target task function along with the observed points

# In[5]:


from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


fig, ax = plt.subplots(1, 1, figsize=(12, 8))
x = torch.linspace(-10, 10, 51)
for task in data_by_task:
    # plot true function and observed values for base runs
    t = ax.plot(
        unnormalize(data_by_task[task]["train_x"], bounds=BOUNDS).cpu().numpy(),
        data_by_task[task]["train_y"].cpu().numpy(),
        ".",
        markersize=10,
        label=f"Observed task {task}",
    )
    ax.plot(
        x.detach().numpy(),
        f(x, task_shift(task + 1)).cpu().numpy(),
        label=f"Base task {task}",
        color=t[0].get_color(),
    )
# plot true target function
ax.plot(
    x.detach().numpy(),
    f(x, TARGET_SHIFT).detach().numpy(),
    "--",
    label="Target task",
)
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()


# ### Fit base task models

# First, let's define a helper function to fit a FixedNoiseGP with an fixed observed noise level.

# In[6]:


from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_mll


def get_fitted_model(train_X, train_Y, train_Yvar, state_dict=None):
    """
    Get a single task GP. The model will be fit unless a state_dict with model
        hyperparameters is provided.
    """
    Y_mean = train_Y.mean(dim=-2, keepdim=True)
    Y_std = train_Y.std(dim=-2, keepdim=True)
    model = FixedNoiseGP(train_X, (train_Y - Y_mean) / Y_std, train_Yvar)
    model.Y_mean = Y_mean
    model.Y_std = Y_std
    if state_dict is None:
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
        fit_gpytorch_mll(mll)
    else:
        model.load_state_dict(state_dict)
    return model


# #### Now let's fit a FixedNoiseGP for each base task

# In[7]:


# Fit base model
base_model_list = []
for task in range(NUM_BASE_TASKS):
    print(f"Fitting base model {task}")
    model = get_fitted_model(
        data_by_task[task]["train_x"],
        data_by_task[task]["train_y"],
        data_by_task[task]["train_yvar"],
    )
    base_model_list.append(model)


# ### Implement the RGPE
# 
# The main idea of the RGPE is to estimate the target function as weighted sum of the target model and the base models:
# $$\bar f(\mathbf x | \mathcal D) =
# \sum_{i=1}^{t} w_if^i(\mathbf x |\mathcal D_i)$$
# Importantly, the ensemble model is also a GP:
# $$\bar f(\mathbf x | \mathcal D) \sim \mathcal N\bigg(\sum_{i=1}^{t} w_i\mu_i(\mathbf x), \sum_{i=1}^{t}w_i^2\sigma_i^2\bigg)$$
# 
# The weights $w_i$ for model $i$ are based on the the ranking loss between a draw from the model's posterior and the targets. Specifically, the ranking loss for model $i$ is:
# $$\mathcal L(f^i, \mathcal D_t) = \sum_{j=1}^{n_t}\sum_{k=1}^{n_t}\mathbb 1\bigg[\bigg(f^i\big(\mathbf x^t_j\big) < f^i\big(\mathbf x_k^t\big)\bigg)\oplus \big(y_j^t < y_k^t\big)\bigg]$$
# where $\oplus$ is exclusive-or.
# 
# The loss for the target model is computing using leave-one-out cross-validation (LOOCV) and is given by:
# $$\mathcal L(f^t, \mathcal D_t) = \sum_{j=1}^{n_t}\sum_{k=1}^{n_t}\mathbb 1\bigg[\bigg(f^t_{-j}\big(\mathbf x^t_j\big) < f^t_{-j}\big(\mathbf x_k^t\big)\bigg)\oplus \big(y_j^t < y_k^t\big)\bigg]$$
# where $f^t_{-j}$ model fitted to all data from the target task except training example $j$.
# 
# The weights are then computed as:
# $$w_i = \frac{1}{S}\sum_{s=1}^S\mathbb 1\big(i = \text{argmin}_{i'}l_{i', s}\big)$$

# In[8]:


def roll_col(X, shift):
    """
    Rotate columns to right by shift.
    """
    return torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1)


# In[9]:


def compute_ranking_loss(f_samps, target_y):
    """
    Compute ranking loss for each sample from the posterior over target points.

    Args:
        f_samps: `n_samples x (n) x n`-dim tensor of samples
        target_y: `n x 1`-dim tensor of targets
    Returns:
        Tensor: `n_samples`-dim tensor containing the ranking loss across each sample
    """
    n = target_y.shape[0]
    if f_samps.ndim == 3:
        # Compute ranking loss for target model
        # take cartesian product of target_y
        cartesian_y = torch.cartesian_prod(
            target_y.squeeze(-1),
            target_y.squeeze(-1),
        ).view(n, n, 2)
        # the diagonal of f_samps are the out-of-sample predictions
        # for each LOO model, compare the out of sample predictions to each in-sample prediction
        rank_loss = (
            (
                (f_samps.diagonal(dim1=1, dim2=2).unsqueeze(-1) < f_samps)
                ^ (cartesian_y[..., 0] < cartesian_y[..., 1])
            )
            .sum(dim=-1)
            .sum(dim=-1)
        )
    else:
        rank_loss = torch.zeros(
            f_samps.shape[0], dtype=torch.long, device=target_y.device
        )
        y_stack = target_y.squeeze(-1).expand(f_samps.shape)
        for i in range(1, target_y.shape[0]):
            rank_loss += (
                (roll_col(f_samps, i) < f_samps) ^ (roll_col(y_stack, i) < y_stack)
            ).sum(dim=-1)
    return rank_loss


# Define a function to:
# 1. Create a batch mode-gp LOOCV GP using the hyperparameters from `target_model`
# 2. Draw a joint sample across all points from the target task (in-sample and out-of-sample)

# In[10]:


def get_target_model_loocv_sample_preds(
    train_x, train_y, train_yvar, target_model, num_samples
):
    """
    Create a batch-mode LOOCV GP and draw a joint sample across all points from the target task.

    Args:
        train_x: `n x d` tensor of training points
        train_y: `n x 1` tensor of training targets
        target_model: fitted target model
        num_samples: number of mc samples to draw

    Return: `num_samples x n x n`-dim tensor of samples, where dim=1 represents the `n` LOO models,
        and dim=2 represents the `n` training points.
    """
    batch_size = len(train_x)
    masks = torch.eye(len(train_x), dtype=torch.uint8, device=device).bool()
    train_x_cv = torch.stack([train_x[~m] for m in masks])
    train_y_cv = torch.stack([train_y[~m] for m in masks])
    train_yvar_cv = torch.stack([train_yvar[~m] for m in masks])
    state_dict = target_model.state_dict()
    # expand to batch size of batch_mode LOOCV model
    state_dict_expanded = {
        name: t.expand(batch_size, *[-1 for _ in range(t.ndim)])
        for name, t in state_dict.items()
    }
    model = get_fitted_model(
        train_x_cv, train_y_cv, train_yvar_cv, state_dict=state_dict_expanded
    )
    with torch.no_grad():
        posterior = model.posterior(train_x)
        # Since we have a batch mode gp and model.posterior always returns an output dimension,
        # the output from `posterior.sample()` here `num_samples x n x n x 1`, so let's squeeze
        # the last dimension.
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        return sampler(posterior).squeeze(-1)


# In[11]:


def compute_rank_weights(train_x, train_y, base_models, target_model, num_samples):
    """
    Compute ranking weights for each base model and the target model (using
        LOOCV for the target model). Note: This implementation does not currently
        address weight dilution, since we only have a small number of base models.

    Args:
        train_x: `n x d` tensor of training points (for target task)
        train_y: `n` tensor of training targets (for target task)
        base_models: list of base models
        target_model: target model
        num_samples: number of mc samples

    Returns:
        Tensor: `n_t`-dim tensor with the ranking weight for each model
    """
    ranking_losses = []
    # compute ranking loss for each base model
    for task in range(len(base_models)):
        model = base_models[task]
        # compute posterior over training points for target task
        posterior = model.posterior(train_x)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        base_f_samps = sampler(posterior).squeeze(-1).squeeze(-1)
        # compute and save ranking loss
        ranking_losses.append(compute_ranking_loss(base_f_samps, train_y))
    # compute ranking loss for target model using LOOCV
    # f_samps
    target_f_samps = get_target_model_loocv_sample_preds(
        train_x,
        train_y,
        train_yvar,
        target_model,
        num_samples,
    )
    ranking_losses.append(compute_ranking_loss(target_f_samps, train_y))
    ranking_loss_tensor = torch.stack(ranking_losses)
    # compute best model (minimum ranking loss) for each sample
    best_models = torch.argmin(ranking_loss_tensor, dim=0)
    # compute proportion of samples for which each model is best
    rank_weights = (
        best_models.bincount(minlength=len(ranking_losses)).type_as(train_x)
        / num_samples
    )
    return rank_weights


# In[12]:


from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import GP
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import PsdSumLazyTensor
from gpytorch.likelihoods import LikelihoodList
from torch.nn import ModuleList


class RGPE(GP, GPyTorchModel):
    """
    Rank-weighted GP ensemble. Note: this class inherits from GPyTorchModel which provides an
        interface for GPyTorch models in botorch.
    """

    _num_outputs = 1  # metadata for botorch

    def __init__(self, models, weights):
        super().__init__()
        self.models = ModuleList(models)
        for m in models:
            if not hasattr(m, "likelihood"):
                raise ValueError(
                    "RGPE currently only supports models that have a likelihood (e.g. ExactGPs)"
                )
        self.likelihood = LikelihoodList(*[m.likelihood for m in models])
        self.weights = weights
        self.to(weights)

    def forward(self, x):
        weighted_means = []
        weighted_covars = []
        # filter model with zero weights
        # weights on covariance matrices are weight**2
        non_zero_weight_indices = (self.weights**2 > 0).nonzero()
        non_zero_weights = self.weights[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()

        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            model = self.models[raw_idx]
            posterior = model.posterior(x)
            # unstandardize predictions
            posterior_mean = posterior.mean.squeeze(-1) * model.Y_std + model.Y_mean
            posterior_cov = posterior.mvn.lazy_covariance_matrix * model.Y_std.pow(2)
            # apply weight
            weight = non_zero_weights[non_zero_weight_idx]
            weighted_means.append(weight * posterior_mean)
            weighted_covars.append(posterior_cov * weight**2)
        # set mean and covariance to be the rank-weighted sum the means and covariances of the
        # base models and target model
        mean_x = torch.stack(weighted_means).sum(dim=0)
        covar_x = PsdSumLazyTensor(*weighted_covars)
        return MultivariateNormal(mean_x, covar_x)


# ### Optimize target function using RGPE + qNEI

# In[13]:


from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf

# suppress GPyTorch warnings about adding jitter
import warnings


warnings.filterwarnings("ignore", "^.*jitter.*", category=RuntimeWarning)


best_rgpe_all = []
best_random_all = []
best_vanilla_nei_all = []
N_BATCH = 10 if not SMOKE_TEST else 2
NUM_POSTERIOR_SAMPLES = 256 if not SMOKE_TEST else 16
RANDOM_INITIALIZATION_SIZE = 3
N_TRIALS = 10 if not SMOKE_TEST else 2
MC_SAMPLES = 512 if not SMOKE_TEST else 32
N_RESTART_CANDIDATES = 512 if not SMOKE_TEST else 8
N_RESTARTS = 10 if not SMOKE_TEST else 2
Q_BATCH_SIZE = 1


# Average over multiple trials
for trial in range(N_TRIALS):
    print(f"Trial {trial + 1} of {N_TRIALS}")
    best_rgpe = []
    best_random = []
    best_vanilla_nei = []
    # Initial random observations
    raw_x = draw_sobol_samples(
        bounds=BOUNDS, n=RANDOM_INITIALIZATION_SIZE, q=1, seed=trial
    ).squeeze(1)
    train_x = normalize(raw_x, bounds=BOUNDS)
    train_y_noiseless = f(raw_x)
    train_y = train_y_noiseless + noise_std * torch.randn_like(train_y_noiseless)
    train_yvar = torch.full_like(train_y, noise_std**2)
    vanilla_nei_train_x = train_x.clone()
    vanilla_nei_train_y = train_y.clone()
    vanilla_nei_train_yvar = train_yvar.clone()
    # keep track of the best observed point at each iteration
    best_value = train_y.max().item()
    best_rgpe.append(best_value)
    best_random.append(best_value)
    vanilla_nei_best_value = best_value
    best_vanilla_nei.append(vanilla_nei_best_value)

    # Run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(N_BATCH):
        target_model = get_fitted_model(train_x, train_y, train_yvar)
        model_list = base_model_list + [target_model]
        rank_weights = compute_rank_weights(
            train_x,
            train_y,
            base_model_list,
            target_model,
            NUM_POSTERIOR_SAMPLES,
        )

        # create model and acquisition function
        rgpe_model = RGPE(model_list, rank_weights)
        sampler_qnei = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        qNEI = qNoisyExpectedImprovement(
            model=rgpe_model,
            X_baseline=train_x,
            sampler=sampler_qnei,
            prune_baseline=False,
        )

        # optimize
        candidate, _ = optimize_acqf(
            acq_function=qNEI,
            bounds=torch.tensor([[0.0], [1.0]], dtype=dtype, device=device),
            q=Q_BATCH_SIZE,
            num_restarts=N_RESTARTS,
            raw_samples=N_RESTART_CANDIDATES,
        )

        # fetch the new values
        new_x = candidate.detach()
        new_y_noiseless = f(unnormalize(new_x, bounds=BOUNDS))
        new_y = new_y_noiseless + noise_std * torch.randn_like(new_y_noiseless)
        new_yvar = torch.full_like(new_y, noise_std**2)

        # update training points
        train_x = torch.cat((train_x, new_x))
        train_y = torch.cat((train_y, new_y))
        train_yvar = torch.cat((train_yvar, new_yvar))
        random_candidate = torch.rand(1, dtype=dtype, device=device)
        next_random_noiseless = f(unnormalize(random_candidate, bounds=BOUNDS))
        next_random = next_random_noiseless + noise_std * torch.randn_like(
            next_random_noiseless
        )
        next_random_best = next_random.max().item()
        best_random.append(max(best_random[-1], next_random_best))

        # get the new best observed value
        best_value = train_y.max().item()
        best_rgpe.append(best_value)

        # Run Vanilla NEI for comparison
        vanilla_nei_model = get_fitted_model(
            vanilla_nei_train_x,
            vanilla_nei_train_y,
            vanilla_nei_train_yvar,
        )
        vanilla_nei_sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([MC_SAMPLES])
        )
        vanilla_qNEI = qNoisyExpectedImprovement(
            model=vanilla_nei_model,
            X_baseline=vanilla_nei_train_x,
            sampler=vanilla_nei_sampler,
        )
        vanilla_nei_candidate, _ = optimize_acqf(
            acq_function=vanilla_qNEI,
            bounds=torch.tensor([[0.0], [1.0]], dtype=dtype, device=device),
            q=Q_BATCH_SIZE,
            num_restarts=N_RESTARTS,
            raw_samples=N_RESTART_CANDIDATES,
        )
        # fetch the new values
        vanilla_nei_new_x = vanilla_nei_candidate.detach()
        vanilla_nei_new_y_noiseless = f(unnormalize(vanilla_nei_new_x, bounds=BOUNDS))
        vanilla_nei_new_y = vanilla_nei_new_y_noiseless + noise_std * torch.randn_like(
            new_y_noiseless
        )
        vanilla_nei_new_yvar = torch.full_like(vanilla_nei_new_y, noise_std**2)

        # update training points
        vanilla_nei_train_x = torch.cat([vanilla_nei_train_x, vanilla_nei_new_x])
        vanilla_nei_train_y = torch.cat([vanilla_nei_train_y, vanilla_nei_new_y])
        vanilla_nei_train_yvar = torch.cat(
            [vanilla_nei_train_yvar, vanilla_nei_new_yvar]
        )

        # get the new best observed value
        vanilla_nei_best_value = vanilla_nei_train_y.max().item()
        best_vanilla_nei.append(vanilla_nei_best_value)

    best_rgpe_all.append(best_rgpe)
    best_random_all.append(best_random)
    best_vanilla_nei_all.append(best_vanilla_nei)


# #### Plot best observed value vs iteration

# In[14]:


import numpy as np


best_rgpe_all = np.array(best_rgpe_all)
best_random_all = np.array(best_random_all)
best_vanilla_nei_all = np.array(best_vanilla_nei_all)

x = range(RANDOM_INITIALIZATION_SIZE, RANDOM_INITIALIZATION_SIZE + N_BATCH + 1)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# Plot RGPE - NEI
ax.errorbar(
    x,
    best_rgpe_all.mean(axis=0),
    yerr=1.96 * best_rgpe_all.std(axis=0) / math.sqrt(N_TRIALS),
    label="RGPE - NEI",
    linewidth=3,
    capsize=5,
    capthick=3,
)
# Plot FixedNoiseGP - NEI
ax.errorbar(
    x,
    best_vanilla_nei_all.mean(axis=0),
    yerr=1.96 * best_vanilla_nei_all.std(axis=0) / math.sqrt(N_TRIALS),
    label="FixedNoiseGP - NEI",
    linewidth=3,
    capsize=5,
    capthick=3,
)
# Plot Random
ax.errorbar(
    x,
    best_random_all.mean(axis=0),
    yerr=1.96 * best_random_all.std(axis=0) / math.sqrt(N_TRIALS),
    label="Random",
    linewidth=3,
    capsize=5,
    capthick=3,
)
ax.set_ylim(bottom=0)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Best Observed Value", fontsize=12)
ax.set_title("Best Observed Value by Iteration", fontsize=12)
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()

