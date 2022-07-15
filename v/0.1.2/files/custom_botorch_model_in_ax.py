#!/usr/bin/env python
# coding: utf-8

# ## Using a custom botorch model with Ax
# 
# In this tutorial, we illustrate how to use a custom BoTorch model within Ax's `SimpleExperiment` API. This allows us to harness the convenience of Ax for running Bayesian Optimization loops, while at the same time maintaining full flexibility in terms of the modeling.
# 
# Acquisition functions and strategies for optimizing acquisitions can be swapped out in much the same fashion. See for example the tutorial for [Implementing a custom acquisition function](./custom_acquisition).
# 
# If you want to do something non-standard, or would like to have full insight into every aspect of the implementation, please see [this tutorial](./closed_loop_botorch_only) for how to write your own full optimization loop in BoTorch.

# ### Implementing the custom model
# 
# For this tutorial, we implement a very simple gpytorch Exact GP Model that uses an RBF kernel (with ARD) and infers a (homoskedastic) noise level.
# 
# Model definition is straightforward - here we implement a gpytorch `ExactGP` that also inherits from `GPyTorchModel` -- this adds all the api calls that botorch expects in its various modules. 
# 
# *Note:* botorch also allows implementing other custom models as long as they follow the minimal `Model` API. For more information, please see the [Model Documentation](../docs/models).

# In[1]:


from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior


class SimpleCustomGP(ExactGP, GPyTorchModel):

    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y, GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# #### Define a factory function to be used with Ax's BotorchModel
# 
# Ax's `BotorchModel` internally breaks down the different components of Bayesian Optimization (model generation & fitting, defining acquisition functions, and optimizing them) into a functional api. 
# 
# Depending on which of these components we want to modify, we can pass in an associated custom factory function to the `BotorchModel` constructor. In order to use a custom model, we have to implement a model factory function that, given data according to Ax's api specification, instantiates and fits a BoTorch Model object.
# 
# The call signature of this factory function is the following: 
# 
# ```python
# def get_and_fit_gpytorch_model(
#     Xs: List[Tensor],
#     Ys: List[Tensor],
#     Yvars: List[Tensor],
#     state_dict: Optional[Dict[str, Tensor]] = None,
#     **kwargs: Any,
# ) -> Model:
# ```
# 
# where
# - the `i`-th element of `Xs` are the training features for the i-th outcome as an `n_i x d` tensor (in our simple example, we only have one outcome)
# - similarly, the `i`-th element of `Ys` and `Yvars` are the observations and associated observation variances for the `i`-th outcome as `n_i x 1` tensors
# - `state_dict` is an optional PyTorch module state dict that can be used to initialize the model's parameters to pre-specified values
# 
# The function must return a botorch `Model` object. What happens inside the function is up to you.
# 
# Using botorch's `fit_gpytorch_model` utility function, model-fitting is straightforward for this simple model (you may have to use your own custom model fitting loop when working with more complex models - see the tutorial for [Fitting a model with torch.optim](fit_model_with_torch_optimizer).

# In[2]:


from botorch.fit import fit_gpytorch_model

def _get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    model = SimpleCustomGP(Xs[0], Ys[0].view(-1))  # collapse trailing dimension
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


# ### Set up the optimization problem in Ax

# Ax's `SimpleExperiment` API requires an evaluation function that is able to compute all the metrics required in the experiment. This function needs to accept a set of parameter values as a dictionary. It should produce a dictionary of metric names to tuples of mean and standard error for those metrics.
# 
# For this tutorial, we use the Branin function, a simple synthetic benchmark function in two dimensions. In an actual application, this could be arbitrarily complicated - e.g. this function could run some costly simulation, conduct some A/B tests, or kick off some ML model training job with the given parameters). 

# In[3]:


import random
import numpy as np

def branin(parameterization, *args):
    x1, x2 = parameterization["x1"], parameterization["x2"]
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    # let's add some synthetic observation noise
    y += random.normalvariate(0, 0.1)
    return {"branin": (y, 0.0)}


# We need to define a search space for our experiment that defines the parameters and the set of feasible values.

# In[4]:


from ax import ParameterType, RangeParameter, SearchSpace

search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        ),
        RangeParameter(
            name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15
        ),
    ]
)


# Third, we make a `SimpleExperiment` — note that the `objective_name` needs to be one of the metric names returned by the evaluation function.

# In[5]:


from ax import SimpleExperiment

exp = SimpleExperiment(
    name="test_branin",
    search_space=search_space,
    evaluation_function=branin,
    objective_name="branin",
    minimize=True,
)


# We use the Sobol generator to create 5 (quasi-) random initial point in the search space. Calling `batch_trial` will cause Ax to evaluate the underlying `branin` function at the generated points, and automatically keep track of the results.

# In[6]:


from ax.modelbridge import get_sobol

sobol = get_sobol(exp.search_space)
exp.new_batch_trial(generator_run=sobol.gen(5))


# To run our custom botorch model inside the Ax optimization loop, we can use the `get_botorch` factory function from `ax.modelbridge.factory`. Any keyword arguments given to this function are passed through to the `BotorchModel` constructor. To use our custom model, we just need to pass our newly minted `_get_and_fit_simple_custom_gp` function to `get_botorch` using the `model_constructor` argument.
# 
# **Note:** `get_botorch` by default automatically applies a number of parameter transformations (e.g. to normalize input data or standardize output data). This is typically what you want for standard use cases with continuous parameters. If your model expects raw parameters, make sure to pass in `transforms=[]` to avoid any transformations to take place. See **TODO: UPDATE LINK** the [Ax documentation](Ax/docs/models.html#transforms) for additional information on how transformations in Ax work.

# #### Run the optimization loop
# 
# We're ready to run the Bayesian Optimization loop.

# In[7]:


from ax.modelbridge.factory import get_botorch

for i in range(5):
    print(f"Running optimization batch {i+1}/5...")
    model = get_botorch(
        experiment=exp,
        data=exp.eval(),
        search_space=exp.search_space,
        model_constructor=_get_and_fit_simple_custom_gp,
    )
    batch = exp.new_trial(generator_run=model.gen(1))
    
print("Done!")

