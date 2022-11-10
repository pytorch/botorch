#!/usr/bin/env python3
# coding: utf-8

# ## Using a custom BoTorch model with Ax
# 
# In this tutorial, we illustrate how to use a custom BoTorch model within Ax's `botorch_modular` API. This allows us to harness the convenience of Ax for running Bayesian Optimization loops, while at the same time maintaining full flexibility in terms of the modeling.
# 
# Acquisition functions and strategies for optimizing acquisitions can be swapped out in much the same fashion. See for example the tutorial for [Implementing a custom acquisition function](./custom_acquisition).
# 
# If you want to do something non-standard, or would like to have full insight into every aspect of the implementation, please see [this tutorial](./closed_loop_botorch_only) for how to write your own full optimization loop in BoTorch.
# 
# Next cell sets up a decorator solely to speed up the testing of the notebook. You can safely ignore this cell and the use of the decorator throughout the tutorial.

# In[1]:


import os
from contextlib import contextmanager

from ax.utils.testing.mock import fast_botorch_optimize_context_manager
import plotly.io as pio
# Ax uses Plotly to produce interactive plots. These are great for viewing and analysis,
# though they also lead to large file sizes, which is not ideal for files living in GH.
# Changing the default to `png` strips the interactive components to get around this.
pio.renderers.default = "png"

SMOKE_TEST = os.environ.get("SMOKE_TEST")
NUM_EVALS = 10 if SMOKE_TEST else 30


@contextmanager
def dummy_context_manager():
    yield


if SMOKE_TEST:
    fast_smoke_test = fast_botorch_optimize_context_manager
else:
    fast_smoke_test = dummy_context_manager


# ### Implementing the custom model
# 
# For this tutorial, we implement a very simple gpytorch Exact GP Model that uses an RBF kernel (with ARD) and infers a (homoskedastic) noise level.
# 
# Model definition is straightforward - here we implement a gpytorch `ExactGP` that also inherits from `GPyTorchModel` -- this adds all the api calls that botorch expects in its various modules. 
# 
# *Note:* botorch also allows implementing other custom models as long as they follow the minimal `Model` API. For more information, please see the [Model Documentation](../docs/models).

# In[2]:


from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.datasets import SupervisedDataset
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP


class SimpleCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# ### Instantiate a `BoTorchModel` in Ax
# 
# A `BoTorchModel` in Ax encapsulates both the surrogate (commonly referred to as `Model` in BoTorch) and an acquisition function. Here, we will only specify the custom surrogate and let Ax choose the default acquisition function.
# 
# Most models should work with the base `Surrogate` in Ax, except for BoTorch `ModelListGP`, which works with `ListSurrogate`.
# Note that the `Model` (e.g., the `SimpleCustomGP`) must implement `construct_inputs`, as this is used to construct the inputs required for instantiating a `Model` instance from the experiment data.
# 
# In case the `Model` requires a complex set of arguments that cannot be constructed using a `construct_inputs` method, one can initialize the `model` and supply it via `Surrogate.from_botorch(model=model, mll_class=<Optional>)`, replacing the `Surrogate(...)` below.

# In[3]:


from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate


ax_model = BoTorchModel(
    surrogate=Surrogate(
        # The model class to use
        botorch_model_class=SimpleCustomGP,
        # Optional, MLL class with which to optimize model parameters
        # mll_class=ExactMarginalLogLikelihood,
        # Optional, dictionary of keyword arguments to model constructor
        # model_options={}
    ),
    # Optional, acquisition function class to use - see custom acquisition tutorial
    # botorch_acqf_class=qExpectedImprovement,
)


# ### Combine with a `ModelBridge`
# 
# `Model`s in Ax require a `ModelBridge` to interface with `Experiment`s. A `ModelBridge` takes the inputs supplied by the `Experiment` and converts them to the inputs expected by the `Model`. For a `BoTorchModel`, we use `TorchModelBridge`. The usage is as follows:
# 
# ```
# from ax.modelbridge import TorchModelBridge
# model_bridge = TorchModelBridge(
#     experiment: Experiment,
#     search_space: SearchSpace,
#     data: Data,
#     model: TorchModel,
#     transforms: List[Type[Transform]],
#     # And additional optional arguments.
# )
# # To generate a trial
# trial = model_bridge.gen(1)
# ```
# 
# For Modular BoTorch interface, we can combine the creation of the `BoTorchModel` and the `TorchModelBridge` into a single step as follows:
# 
# ```
# from ax.modelbridge.registry import Models
# model_bridge = Models.BOTORCH_MODULAR(
#     experiment=experiment,
#     data=data,
#     surrogate=Surrogate(SimpleCustomGP),  # Optional, will use default if unspecified
#     # Optional, will use default if unspecified
#     # botorch_acqf_class=qNoisyExpectedImprovement,  
# )
# # To generate a trial
# trial = model_bridge.gen(1)
# ```
# 

# # Using the custom model in Ax to optimize the Branin function
# 
# We will demonstrate this with both the Service API (simpler, easier to use) and the Developer API (advanced, more customizable).

# ## Optimization with the Service API
# 
# A detailed tutorial on the Service API can be found [here](https://ax.dev/tutorials/gpei_hartmann_service.html).
# 
# In order to customize the way the candidates are created in Service API, we need to construct a new `GenerationStrategy` and pass it into `AxClient`.

# In[4]:


from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models


gs = GenerationStrategy(
    steps=[
        # Quasi-random initialization step
        GenerationStep(
            model=Models.SOBOL,
            num_trials=5,  # How many trials should be produced from this generation step
        ),
        # Bayesian optimization step using the custom acquisition function
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,  # No limitation on how many trials should be produced from this step
            # For `BOTORCH_MODULAR`, we pass in kwargs to specify what surrogate or acquisition function to use.
            model_kwargs={
                "surrogate": Surrogate(SimpleCustomGP),
            },
        ),
    ]
)


# ### Setting up the experiment
# 
# In order to use the `GenerationStrategy` we just created, we will pass it into the `AxClient`.

# In[5]:


import torch
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.test_functions import Branin


# Initialize the client - AxClient offers a convenient API to control the experiment
ax_client = AxClient(generation_strategy=gs)
# Setup the experiment
ax_client.create_experiment(
    name="branin_test_experiment",
    parameters=[
        {
            "name": "x1",
            "type": "range",
            # It is crucial to use floats for the bounds, i.e., 0.0 rather than 0.
            # Otherwise, the parameter would be inferred as an integer range.
            "bounds": [-5.0, 10.0],
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 15.0],
        },
    ],
    objectives={
        "branin": ObjectiveProperties(minimize=True),
    },
)
# Setup a function to evaluate the trials
branin = Branin()


def evaluate(parameters):
    x = torch.tensor([[parameters.get(f"x{i+1}") for i in range(2)]])
    # The GaussianLikelihood used by our model infers an observation noise level,
    # so we pass an sem value of NaN to indicate that observation noise is unknown
    return {"branin": (branin(x).item(), float("nan"))}


# ### Running the BO loop

# In[6]:


with fast_smoke_test():
    for i in range(NUM_EVALS):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))


# ### Viewing the evaluated trials

# In[7]:


ax_client.get_trials_data_frame()


# In[8]:


parameters, values = ax_client.get_best_parameters()
print(f"Best parameters: {parameters}")
print(f"Corresponding mean: {values[0]}, covariance: {values[1]}")


# ### Plotting the response surface and optimization progress

# In[9]:


from ax.utils.notebook.plotting import render

render(ax_client.get_contour_plot())


# In[10]:


best_parameters, values = ax_client.get_best_parameters()
best_parameters, values[0]


# In[11]:


render(ax_client.get_optimization_trace(objective_optimum=0.397887))


# ## Optimization with the Developer API
# 
# A detailed tutorial on the Service API can be found [here](https://ax.dev/tutorials/gpei_hartmann_developer.html).
# 
# ### Set up the Experiment in Ax
# 
# We need 3 inputs for an Ax `Experiment`:
# - A search space to optimize over;
# - An optimization config specifiying the objective / metrics to optimize, and optional outcome constraints;
# - A runner that handles the deployment of trials. For a synthetic optimization problem, such as here, this only returns simple metadata about the trial.

# In[12]:


import pandas as pd
import torch
from ax import (
    Data,
    Experiment,
    Metric,
    Objective,
    OptimizationConfig,
    ParameterType,
    RangeParameter,
    Runner,
    SearchSpace,
)
from ax.utils.common.result import Ok
from botorch.test_functions import Branin


branin_func = Branin()

# For our purposes, the metric is a wrapper that structures the function output.
class BraninMetric(Metric):
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            tensor_params = torch.tensor([params["x1"], params["x2"]])
            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "trial_index": trial.index,
                    "mean": branin_func(tensor_params),
                    "sem": float("nan"),  # SEM (observation noise) - NaN indicates unknown
                }
            )
        return Ok(value=Data(df=pd.DataFrame.from_records(records)))


# Search space defines the parameters, their types, and acceptable values.
search_space = SearchSpace(
    parameters=[
        RangeParameter(name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10),
        RangeParameter(name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15),
    ]
)

optimization_config = OptimizationConfig(
    objective=Objective(
        metric=BraninMetric(name="branin_metric", lower_is_better=True),
        minimize=True,  # This is optional since we specified `lower_is_better=True`
    )
)


class MyRunner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        return trial_metadata


exp = Experiment(
    name="branin_experiment",
    search_space=search_space,
    optimization_config=optimization_config,
    runner=MyRunner(),
)


# ### Run the BO loop
# 
# First, we use the Sobol generator to create 5 (quasi-) random initial point in the search space. Ax controls objective evaluations via `Trial`s. 
# - We generate a `Trial` using a generator run, e.g., `Sobol` below. A `Trial` specifies relevant metadata as well as the parameters to be evaluated. At this point, the `Trial` is at the `CANDIDATE` stage.
# - We run the `Trial` using `Trial.run()`. In our example, this serves to mark the `Trial` as `RUNNING`. In an advanced application, this can be used to dispatch the `Trial` for evaluation on a remote server.
# - Once the `Trial` is done running, we mark it as `COMPLETED`. This tells the `Experiment` that it can fetch the `Trial` data. 
# 
# A `Trial` supports evaluation of a single parameterization. For parallel evaluations, see [`BatchTrial`](https://ax.dev/docs/core.html#trial-vs-batch-trial).

# In[13]:


from ax.modelbridge.registry import Models


sobol = Models.SOBOL(exp.search_space)

for i in range(5):
    trial = exp.new_trial(generator_run=sobol.gen(1))
    trial.run()
    trial.mark_completed()


# Once the initial (quasi-) random stage is completed, we can use our `SimpleCustomGP` with the default acquisition function chosen by `Ax` to run the BO loop.

# In[14]:


with fast_smoke_test():
    for i in range(NUM_EVALS - 5):
        model_bridge = Models.BOTORCH_MODULAR(
            experiment=exp,
            data=exp.fetch_data(),
            surrogate=Surrogate(SimpleCustomGP),
        )
        trial = exp.new_trial(generator_run=model_bridge.gen(1))
        trial.run()
        trial.mark_completed()


# View the trials attached to the `Experiment`.

# In[15]:


exp.trials


# View the evaluation data about these trials.

# In[16]:


exp.fetch_data().df


# ### Plot results
# 
# We can use convenient Ax utilities for plotting the results.

# In[17]:


import numpy as np
from ax.plot.trace import optimization_trace_single_method


# `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple
# optimization runs, so we wrap out best objectives array in another array.
objective_means = np.array([[trial.objective_mean for trial in exp.trials.values()]])
best_objective_plot = optimization_trace_single_method(
    y=np.minimum.accumulate(objective_means, axis=1),
    optimum=0.397887,  # Known minimum objective for Branin function.
)
render(best_objective_plot)

