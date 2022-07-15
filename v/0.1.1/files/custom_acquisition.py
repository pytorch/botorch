#!/usr/bin/env python
# coding: utf-8

# ## Writing a custom acquisition function and interfacing with Ax
# 
# As seen in the [custom BoTorch model in Ax](./custom_botorch_model_in_ax) tutorial, Ax's `BotorchModel` is flexible in allowing different components of the Bayesian optimization loop to be specified through a functional API. This tutorial walks through the steps of writing a custom acquisition function and then inserting it into Ax. 
# 
# 
# ### Upper Confidence Bound (UCB)
# 
# The Upper Confidence Bound (UCB) acquisition function balances exploration and exploitation by assigning a score of $\mu + \sqrt{\beta} \cdot \sigma$ if the posterior distribution is normal with mean $\mu$ and variance $\sigma^2$. This "analytic" version is implemented in the `UpperConfidenceBound` class. The Monte Carlo version of UCB is implemented in the `qUpperConfidenceBound` class, which also allows for q-batches of size greater than one. (The derivation of q-UCB is given in Appendix A of [Wilson et. al., 2017](https://arxiv.org/pdf/1712.00424.pdf)).

# ### A scalarized version of q-UCB
# 
# Suppose now that we are in a multi-output setting, where, e.g., we model the effects of a design on multiple metrics. We first show a simple extension of the q-UCB acquisition function that accepts a multi-output model and performs q-UCB on a scalarized version of the multiple outputs, achieved via a vector of weights. Implementing a new acquisition function in botorch is easy; one simply needs to implement the constructor and a `forward` method.

# In[1]:


import math

from torch import Tensor
from typing import Optional

from botorch.acquisition import MCAcquisitionObjective
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch.utils import t_batch_mode_transform


class qScalarizedUpperConfidenceBound(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        weights: Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("weights", torch.as_tensor(weights))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate scalarized qUCB on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Upper Confidence Bound values at the
                given design points `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)  # n x b x q x o
        scalarized_samples = samples.matmul(self.weights)  # n x b x q
        mean = posterior.mean  # b x q x o
        scalarized_mean = mean.matmul(self.weights)  # b x q
        ucb_samples = (
            scalarized_mean
            + math.sqrt(self.beta * math.pi / 2)
            * (scalarized_samples - scalarized_mean).abs()
        )
        return ucb_samples.max(dim=-1)[0].mean(dim=0)


# Note that `qScalarizedUpperConfidenceBound` is very similar to `qUpperConfidenceBound` and only requires a few lines of new code to accomodate scalarization of multiple outputs. The `@t_batch_mode_transform` decorator ensures that the input `X` has an explicit t-batch dimension (code comments are added with shapes for clarity).

# #### Ad-hoc testing q-Scalarized-UCB
# 
# Before hooking the newly defined acquisition function into a Bayesian Optimization loop, we should test it. For this we'll just make sure that it properly evaluates on a compatible multi-output model. Here we just define a basic multi-output `SingleTaskGP` model trained on synthetic data.

# In[2]:


import torch

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

# generate synthetic data
X = torch.rand(20, 2)
Y = torch.stack([torch.sin(X[:, 0]), torch.cos(X[:, 1])], -1)

# construct and fit the multi-output model
gp = SingleTaskGP(X, Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll);

# construct the acquisition function
qSUCB = qScalarizedUpperConfidenceBound(gp, beta=0.1, weights=torch.tensor([0.1, 0.5]))


# In[3]:


# evaluate on single q-batch with q=3
qSUCB(torch.rand(3, 2))


# In[4]:


# batch-evaluate on two q-batches with q=3
qSUCB(torch.rand(2, 3, 2))


# ### A scalarized version of analytic UCB (`q=1` only)
# 
# We can also write an *analytic* version of UCB for a multi-output model, assuming a multivariate normal posterior and `q=1`. The new class `ScalarizedUpperConfidenceBound` subclasses `AnalyticAcquisitionFunction` instead of `MCAcquisitionFunction`. In contrast to the MC version, instead of using the weights on the MC samples, we directly scalarize the mean vector $\mu$ and covariance matrix $\Sigma$ and apply standard UCB on the univariate normal distribution, which has mean $w^T \mu$ and variance $w^T \Sigma w$. In addition to the `@t_batch_transform` decorator, here we are also using `expected_q=1` to ensure the input `X` has a `q=1`.

# In[5]:


from botorch.acquisition import AnalyticAcquisitionFunction


class ScalarizedUpperConfidenceBound(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        weights: Tensor,
        maximize: bool = True,
    ) -> None:
        super().__init__(model=model)
        self.maximize = maximize
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("weights", torch.as_tensor(weights))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the Upper Confidence Bound on the candidate set X using scalarization

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
                design points `X`.
        """
        self.beta = self.beta.to(X)
        batch_shape = X.shape[:-2]
        posterior = self.model.posterior(X)
        means = posterior.mean.squeeze(dim=-2)  # b x o
        scalarized_mean = means.matmul(self.weights)  # b
        covs = posterior.mvn.covariance_matrix  # b x o x o
        weights = self.weights.view(1, -1, 1)  # 1 x o x 1 (assume single batch dimension)
        weights = weights.expand(batch_shape + weights.shape[1:])  # b x o x 1
        weights_transpose = weights.permute(0, 2, 1)  # b x 1 x o
        scalarized_variance = torch.bmm(
            weights_transpose, torch.bmm(covs, weights)
        ).view(batch_shape)  # b
        delta = (self.beta.expand_as(scalarized_mean) * scalarized_variance).sqrt()
        if self.maximize:
            return scalarized_mean + delta
        else:
            return scalarized_mean - delta


# #### Ad-hoc testing Scalarized-UCB
# 
# Notice that we pass in an explicit q-batch dimension for consistency, even though `q=1`.

# In[6]:


# construct the acquisition function
SUCB = ScalarizedUpperConfidenceBound(gp, beta=0.1, weights=torch.tensor([0.1, 0.5]))


# In[7]:


# evaluate on single point
SUCB(torch.rand(1, 2))


# In[8]:


# batch-evaluate on 3 points
SUCB(torch.rand(3, 1, 2))


# To use our newly minted acquisition function within Ax, we need to write a custom factory function and pass it to the constructor of Ax's `BotorchModel` as the `acqf_constructor`, which has the call signature:
# 
# ```python
# def acqf_constructor(
#     model: Model,
#     objective_weights: Tensor,
#     outcome_constraints: Optional[Tuple[Tensor, Tensor]],
#     X_observed: Optional[Tensor] = None,
#     X_pending: Optional[Tensor] = None,
#     **kwargs: Any,
# ) -> AcquisitionFunction:
# ```
# 
# The argument `objective_weights` allows for scalarization of multiple objectives, `outcome_constraints` is used to define constraints on multi-output models, `X_observed` contains previously observed points (useful for acquisition functions such as Noisy Expected Improvement), and `X_pending` are the points that are awaiting observations. By default, Ax uses the Noisy Expected Improvement (`qNoisyExpectedImprovement`) acquisition function and so the default value of `acqf_constructor` is `get_NEI` (see documentation for additional details and context).
# 
# Note that there is ample flexibility to how the arguments of `acqf_constructor` are used. In `get_NEI`, they are used in some preprocessing steps *before* constructing the acquisition function. They could also be directly passed to the botorch acquisition function, or not used at all --  all we need to do is return an `AcquisitionFunction`. We now give a bare-bones example of a custom factory function that returns our analytic scalarized-UCB acquisition.
# 
# ```python
# def get_scalarized_UCB(
#     model: Model,
#     objective_weights: Tensor,
#     **kwargs: Any,
# ) -> AcquisitionFunction:
#     return ScalarizedUpperConfidenceBound(model=model, beta=0.2, weights=objective_weights)
# ```

# By following the example shown in the [custom botorch model in ax](./custom_botorch_model_in_ax) tutorial, a `BotorchModel` can be instantiated with `get_scalarized_UCB` and then run in Ax.
