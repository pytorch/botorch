---
id: models
title: Models
---

Models play an essential role in Bayesian Optimization (BO). A model is used as a
surrogate function for the actual underlying black box function to be optimized.
In BoTorch, a `Model` maps a set of design points to a posterior probability
distribution of its output(s) over the design points.

In BO, the model used is traditionally a Gaussian Process (GP),
in which case the posterior distribution is a multivariate
normal. While BoTorch supports many GP models, **BoTorch makes no
assumption on the model being a GP** or the posterior being multivariate normal.
With the exception of some of the analytic acquisition functions in the
[`botorch.acquisition.analytic`](../api/acquisition.html#analytic-acquisition-function-api)
module, BoTorch’s Monte Carlo-based acquisition functions are compatible with
any model that conforms to the `Model` interface, whether user-implemented or provided.

Under the hood, BoTorch models are PyTorch `Modules` that implement
the light-weight [`Model`](../api/models.html#model-apis) interface.
When working with GPs, [`GPyTorchModel`](../api/models.html#module-botorch.models.gp_regression)
provides a base class for conveniently wrapping GPyTorch models.

Users can extend `Model` and `GPyTorchModel` to generate their own models.
For more on implementing your own models, see
[Implementing Custom Models](#implementing-custom-models) below.


## Terminology

### Multi-Output and Multi-Task
A `Model` (as in the BoTorch object) may have
multiple outputs, multiple inputs, and may exploit correlation
between different inputs. BoTorch uses the following terminology to
distinguish these model types:

* *Multi-Output Model*: a `Model` with multiple
  outputs. Most BoTorch `Model`s are multi-output.
* *Multi-Task Model*: a `Model` making use of a logical grouping of
  inputs/observations (as in the underlying process). For example, there could
  be multiple tasks where each task has a different fidelity.
  In a multi-task model, the relationship between different
  outputs is modeled, with a joint model across tasks.

Note the following:
* A multi-task (MT) model may or may not be a multi-output model.
For example, if a multi-task model uses different tasks for modeling
but only outputs predictions for one of those tasks, it is single-output.
* Conversely, a multi-output (MO) model may or may not be a multi-task model.
For example, multi-output `Model`s that model
different outputs independently rather than
building a joint model are not multi-task.
* If a model is both, we refer to it as a multi-task-multi-output (MTMO) model.

### Noise: Homoskedastic, fixed, and heteroskedastic
Noise can be treated in several different ways:

* *Homoskedastic*: Noise is not provided as an input and is inferred, with a
constant variance that does not depend on `X`. Many models, such as
`SingleTaskGP`, take this approach. Use these models if you know that
your observations are noisy, but not how noisy.

* *Fixed*: Noise is provided as an input and is not fit. In “fixed noise” models
like `FixedNoiseGP`, noise cannot be predicted out-of-sample because it has
not been modeled. Use these models if you have estimates of the noise in
your observations (e.g. observations may be averages over individual samples
in which case you would provide the mean as observation and the standard
error of the mean as the noise estimate), or if you know your observations are
noiseless (by passing a zero noise level).

* *Heteroskedastic*: Noise is provided as an input and is modeled to allow for
predicting noise out-of-sample. Models like `HeteroskedasticSingleTaskGP`
take this approach.

## Standard BoTorch Models

BoTorch provides several GPyTorch models to cover most standard BO use cases:

### Single-Task GPs
These models use the same training data for all outputs and assume conditional
independence of the outputs given the input. If different training data is
required for each output, use a [`ModelListGP`](../api/models.html#module-botorch.models.model_list_gp_regression)
instead.
* [`SingleTaskGP`](../api/models.html#botorch.models.gp_regression.SingleTaskGP): a single-task
  exact GP that infers a homoskedastic noise level (no noise observations).
* [`FixedNoiseGP`](../api/models.html#botorch.models.gp_regression.FixedNoiseGP): a single-task exact GP that
  differs from `SingleTaskGP` in using
  fixed observation noise levels. It requires noise observations.
* [`HeteroskedasticSingleTaskGP`](../api/models.html#botorch.models.gp_regression.HeteroskedasticSingleTaskGP):
  a single-task exact GP that differs from `SingleTaskGP` and `FixedNoiseGP`
  in that it models heteroskedastic noise using an additional
  internal GP model. It requires noise observations.
* [`MixedSingleTaskGP`](../api/models.html#botorch.models.gp_regression_mixed.MixedSingleTaskGP): a single-task exact
  GP that supports mixed search spaces, which combine discrete and continuous features.
* [`SaasFullyBayesianSingleTaskGP`](../api/models.html#botorch.models.fully_bayesian.SaasFullyBayesianSingleTaskGP):
  a fully Bayesian single-task GP with the SAAS prior. This model is suitable for
  sample-efficient high-dimensional Bayesian optimization.

### Model List of Single-Task GPs
* [`ModelListGP`](../api/models.html#module-botorch.models.model_list_gp_regression): A multi-output model in
  which outcomes are modeled independently, given a list of any type of
  single-task GP. This model should be used when the same training data is not
  used for all outputs.

### Multi-Task GPs
* [`MultiTaskGP`](../api/models.html#module-botorch.models.multitask): a Hadamard multi-task,
  multi-output GP using an ICM kernel, inferring a homoskedastic noise level (does not
  require noise observations).
* [`FixedNoiseMultiTaskGP`](../api/models.html#botorch.models.multitask.FixedNoiseMultiTaskGP):
  a Hadamard multi-task, multi-output GP using an ICM kernel, with fixed
  observation noise levels (requires noise observations).
* [`KroneckerMultiTaskGP`](../api/models.html#botorch.models.multitask.KroneckerMultiTaskGP): A multi-task,
  multi-output GP using an ICM kernel, with Kronecker structure. Useful for
  multi-fidelity optimization.
* [`SaasFullyBayesianMultiTaskGP`](../api/models.html#saasfullybayesianmultitaskgp):
  a fully Bayesian multi-task GP using an ICM kernel. The data kernel uses the SAAS
  prior to model high-dimensional parameter spaces.

All of the above models use Matérn 5/2 kernels with Automatic Relevance
Discovery (ARD), and have reasonable priors on hyperparameters that make them
work well in settings where the **input features are normalized to the unit
cube** and the **observations are standardized** (zero mean, unit variance).

## Other useful models

* [`ModelList`](../api/models.html#botorch.models.model.ModelList): a multi-output model container
  in which outcomes are modeled independently by individual `Model`s (as in `ModelListGP`, but the
  component models do not all need to be GPyTorch models).
* [`SingleTaskMultiFidelityGP`](../api/models.html#botorch.models.gp_regression_fidelity.SingleTaskMultiFidelityGP) and
  [`FixedNoiseMultiFidelityGP`](../api/models.html#botorch.models.gp_regression_fidelity.FixedNoiseMultiFidelityGP):
  Models for multi-fidelity optimization.  For more on Multi-Fidelity BO, see the
  [tutorial](../tutorials/discrete_multi_fidelity_bo).
* [`HigherOrderGP`](../api/models.html#botorch.models.higher_order_gp.HigherOrderGP): A GP model with
  matrix-valued predictions, such as images or grids of images.
* [`PairwiseGP`](../api/models.html#module-botorch.models.pairwise_gp): A probit-likelihood GP that
  learns via pairwise comparison data, useful for preference learning.
* [`ApproximateGPyTorchModel`](../api/models.html#botorch.models.approximate_gp.ApproximateGPyTorchModel): for
  efficient computation when data is large or responses are non-Gaussian.
* [Deterministic models](../api/models.html#module-botorch.models.deterministic), such as
  [`AffineDeterministicModel`](../api/models.html#botorch.models.deterministic.AffineDeterministicModel),
  [`AffineFidelityCostModel`](../api/models.html#botorch.models.cost.AffineFidelityCostModel),
  [`GenericDeterministicModel`](../api/models.html#botorch.models.deterministic.GenericDeterministicModel),
  and
  [`PosteriorMeanModel`](../api/models.html#botorch.models.deterministic.PosteriorMeanModel)
  express known input-output relationships; they conform
  to the BoTorch `Model` API, so they can easily be used in conjunction with other
  BoTorch models. Deterministic models are
  useful for multi-objective optimization with known objective
  functions and for encoding cost functions for cost-aware acquisition.
* [`SingleTaskVariationalGP`](../api/models.html#botorch.models.approximate_gp.SingleTaskVariationalGP): an
  approximate model for faster computation when you have a lot of data or your responses
  are non-Gaussian.


## Implementing Custom Models

The configurability of the above models is limited (for instance, it is not
straightforward to use a different kernel). Doing so is an intentional design
decision -- we believe that having a few simple and easy-to-understand models for
basic use cases is more valuable than having a highly complex and configurable
model class whose implementation is difficult to understand.

Instead, we advocate that users implement their own models to cover
more specialized use cases. The light-weight nature of BoTorch's Model API makes
this easy to do. See the
[Using a custom BoTorch model in Ax](../tutorials/custom_botorch_model_in_ax)
tutorial for an example.

The BoTorch `Model` interface is light-weight and easy to extend. The only
requirement for using BoTorch's Monte-Carlo based acquisition functions is that
the model has a `posterior` method. It takes in a Tensor `X` of design points, and
returns a Posterior object describing the (joint) probability distribution of
the model output(s) over the design points in `X`.  The `Posterior` object must
implement an `rsample()` method for sampling from the posterior of the model.
If you wish to use gradient-based optimization algorithms, the model should
allow back-propagating gradients through the samples to the model input.

If you happen to implement a model that would be useful for other
researchers as well (and involves more than just swapping out the Matérn kernel
for an RBF kernel), please consider [contributing](getting_started#contributing)
this model to BoTorch.
