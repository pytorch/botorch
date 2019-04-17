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


## Standard botorch Models

botorch provides several GPyTorch models to cover most standard Bayesian
optimization use cases:

* [`SingleTaskGP`](../api/models.html#singletaskgp): A single-task exact GP that
  infers a homoskedastic noise level (does not require noise observations).
* [`FixedNoiseGP`](../api/models.html#fixednoisegp): A single-task exact GP that
  uses fixed observation noise levels (requires noise observations).
* [`HeteroskedasticSingleTaskGP`](../api/models.html#heteropskedasticsingletaskgp):
  a single-task exact GP that models heteroskedastic noise via
  an additional internal GP model (requires noise observations)
* [`MultiTaskGP`](../api/models.html#multitaskgp): A Hadamard multi-task,
  multi-output GP using an ICM kernel, inferring the noise level (does not
  require noise observations).
* [`FixedNoiseMultiTaskGP`](../api/models.html#fixednoisemultitaskgp): A Hadamard
  multi-task, multi-output GP using an ICM kernel, with fixed observation noise
  levels (requires noise observations).
* [`MultiOutputGP`](../api/models.html#multioutputgp): A multi-output model in
  which outcomes are modeled independently, given a list of any type of
  single-output GP.

All of the above models use Matérn 5/2 kernels with Automatic Relevance
Discovery (ARD), and have reasonable priors on hyperparameters that make them
work well in settings where the input features are normalized to the unit cube
and the observations are standardized (zero mean, unit variance).

Further, `SingleTaskGP`, `FixedNoiseGP`, and `HeteroskedasticSingleTaskGP` also
work as multi-output models (assuming conditional independence of the outputs
given the input).


## Implementing Custom Models

See the [Using a custom botorch model in Ax](../tutorials/custom_botorch_model_in_ax)
tutorial on how to define your own custom models.
