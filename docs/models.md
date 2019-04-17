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
normal.

## Models in botorch

With the exception of some of the analytic Acquisition functions in the
[`botorch.acquisition.analytic`](../api/acquisition.html#botorch-acquisition-analytic)
module, **botorch makes no assumption on the model being a GP, or on the
posterior being a multivariate normal**. The only requirement for using botorch's
Monte-Carlo based acquisition functions is that the model returns a `Posterior`
object that implements an `rsample()` method for sampling from the posterior of
the model (if you wish to use gradient-based optimization algorithms, the model
should allow back-propagating gradients through the samples to the model input).


### The botorch `Model` Interface

botorch models are PyTorch modules that implement the lightweight `Model`
interface. A botorch `Model` requires only two methods:

* A `posterior()` method that returns a
  [`botorch.posteriors.Posterior`](../api/posteriors.html#posterior) over the
  provided points.
* A `reinitialize()` method that provides a way to reinitialize a model with new
  data while optionally keeping previously learned hyperparameters fixed.

A botorch model's `forward` method should take in a Tensor `X` of design points,
and return a `Posterior` object (which would typically describe the (joint)
probability distribution of the model output(s) over the design points in `X`).

### Terminology

Models may have multiple outputs, multiple inputs,
and may exploit correlation between between different inputs. botorch uses the following terminology to distinguish these model types:

* *Multi-Output Model*: a `Model` (as in the botorch object) with multiple outputs.
* *Multi-Task Model*: A `Model` making use of a logical grouping of inputs/observations (as in the underlying process). For example, there could be multiple tasks where each task has a different fidelity.

Note the following:
* A multi-task model may or may not be a multi-output model.
* Conversely, a multi-output model may or may not be a multi-task model.
* If a model is both, we refer to it as a multi-task-multi-output model.

## botorch Models for Standard Use Cases

botorch provides several GPyTorch models to cover most standard Bayesian
optimization use cases. 

All of these models use Matérn 5/2 ARD kernels and support one or more outputs:
### Single-Task GPs
These models use the same training data for all outputs. If different training data is required for each output, use a `ModelListMultiOutputGP` to handle multiple outputs.
* [`SingleTaskGP`](../api/models.html#singletaskgp): a single-task
  exact GP that infers a homoskedastic noise level (no noise observations)
* [`FixedNoiseGP`](../api/models.html#fixednoisegp): a single-task exact GP that uses fixed observation noise levels (requires noise observations)
* [`HeteroskedasticSingleTaskGP`](../api/models.html#heteropskedasticsingletaskgp):
  a single task exact GP that models heteroskedastic noise via
  an additional internal GP model (requires noise observations)

### ModelList GPs
* [`ModelListGP`](../api/models.html#modellistgp): A multi-output model in
  which outcomes are modeled independently, given a list of any type of
  single-task GP. This model should be used when the same training data is not used for all outputs. 

### Multi-Task GPs
* [`MultiTaskGP`](../api/models.html#multitaskgp): A Hadamard multi-task,
  multi-output GP using an ICM kernel, inferring the noise level (no noise
  observations).
* [`FixedNoiseMultiTaskGP`](../api/models.html#fixednoisemultitaskgp): A Hadamard
  multi-task, multi-output GP using an ICM kernel, with fixed observation noise
  levels (requires noise observations).


## Implementing Custom Models

See the [Using a custom botorch model in Ax](../tutorials/custom_botorch_model_in_ax)
tutorial on how to define your own custom models.
