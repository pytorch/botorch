---
id: models
title: Models
---


Models play an essential role in Bayesian Optimization. A model is used as a
surrogate function for the actual underlying black box function that one is
trying to optimize. In botorch, a `Model` maps a set of design points to a
posterior probability distribution of its output(s) over the design points.

In Bayesian Optimization, the model used is traditionally a Gaussian Process,
in which case the posterior distribution, by definition, is a multivariate
normal. However, with the exception of some of the analytic Acquisition
functions in the
[`botorch.acquisition.analytic`](../api/acquisition.html#botorch-acquisition-analytic)
module, **botorch makes no assumption on the model being a GP, or on the
posterior being a multivariate normal**. The only requirement for using
botorch's Monte-Carlo based acquisition functions is that the model returns a
[`Posterior`](../api/api/posteriors.html#posterior) object that implements an
`rsample()` method for sampling from the posterior of the model (if you wish to
use gradient-based optimization algorithms, the model should allow
back-propagating gradients through the samples to the model input).


## The botorch `Model` Interface

botorch models are PyTorch modules that implement the light-weight
[`Model`](../api/models.html#model) interface. A botorch `Model` requires only
a single `posterior()` method that takes in a Tensor `X` of design points,
and returns a [`Posterior`](../api/posteriors.html#posterior) object describing
the (joint) probability distribution of the model output(s) over the design
points in `X`

In addition, it is typically useful to define a `reinitialize()` method that
provides a way to reinitialize a model with new data while optionally keeping
previously learned hyperparameters fixed (so as to speed up subsequent model
fitting).

When working with GPs, [`GPyTorchModel`](../api/models.html#gpytorchmodel)
provides a base class for conveniently wrapping GPyTorch models.

## Terminology

Models may have multiple outputs, multiple inputs,
and may exploit correlation between between different inputs. botorch uses the
following terminology to distinguish these model types:

* *Multi-Output Model*: a `Model` (as in the botorch object) with multiple outputs.
* *Multi-Task Model*: A `Model` making use of a logical grouping of inputs/observations
(as in the underlying process). For example, there could be multiple tasks where
each task has a different fidelity.

Note the following:
* A multi-task model may or may not be a multi-output model.
* Conversely, a multi-output model may or may not be a multi-task model.
* If a model is both, we refer to it as a multi-task-multi-output model.

## Standard botorch Models

botorch provides several GPyTorch models to cover most standard Bayesian
optimization use cases:

### Single-Task GPs
These models use the same training data for all outputs and assume conditional
independence of the outputs given the input. If different training data is required
for each output, use a `ModelListMultiOutputGP` to handle multiple outputs.
* [`SingleTaskGP`](../api/models.html#singletaskgp): a single-task
  exact GP that infers a homoskedastic noise level (no noise observations)
* [`FixedNoiseGP`](../api/models.html#fixednoisegp): a single-task exact GP that
uses fixed observation noise levels (requires noise observations)
* [`HeteroskedasticSingleTaskGP`](../api/models.html#heteropskedasticsingletaskgp):
  a single-task exact GP that models heteroskedastic noise via
  an additional internal GP model (requires noise observations)

### Model List of Single-Task GPs
* [`ModelListGP`](../api/models.html#modellistgp): A multi-output model in
  which outcomes are modeled independently, given a list of any type of
  single-task GP. This model should be used when the same training data is not
  used for all outputs.

### Multi-Task GPs
* [`MultiTaskGP`](../api/models.html#multitaskgp): A Hadamard multi-task,
  multi-output GP using an ICM kernel, inferring the noise level (does not require
     noise
  observations).
* [`FixedNoiseMultiTaskGP`](../api/models.html#fixednoisemultitaskgp): A Hadamard
  multi-task, multi-output GP using an ICM kernel, with fixed observation noise
  levels (requires noise observations).

All of the above models use Matérn 5/2 kernels with Automatic Relevance Discovery
(ARD), and have reasonable priors on hyperparameters that make them work well in
settings where the input features are normalized to the unit cube and the
observations are standardized (zero mean, unit variance).

## Implementing Custom Models

See the [Using a custom botorch model in Ax](../tutorials/custom_botorch_model_in_ax)
tutorial on how to define your own custom models.
