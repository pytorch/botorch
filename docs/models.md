---
id: models
title: Models
---

## Models in Bayesian Optimization
Models play an essential role in Bayesian Optimization. A model is used as a surrogate
function for the actual underlying black box function that one is trying to
optimize. In botorch, models map a set of design points to a posterior
probability distribution of the model output(s) over the design points.

In Bayesian Optimization, the model used is traditionally a Gaussian Process,
in which case the posterior distribution, by definition, is a multivariate
normal.

## Models in botorch
With the exception of some of the analytic Acquisition functions in the
`botorch.acquisition.analytic` module, **botorch makes no assumption on the model
being a GP, or on the posterior being a multivariate normal.**
The only requirement for using botorch's (quasi-) Monte-Carlo based acquisition
functions is that the model returns a `Posterior` object that implements an
`rsample` method, which can be used to draw samples from the posterior.
In order to use gradient-based optimization algorithms, a model
should allow back-propagating gradients through the samples to the model input.

## botorch `Model` Inferface
botorch models are PyTorch modules that implement the lightweight `Model` inferface. A botorch `Model` requires two only methods:
* A `posterior` method that returns a `botorch.posteriors.Posterior` over the provided points.
* A `reinitialize` method that provides a way to reinitialize a model. 

A botorch model's `forward` method should take in a Tensor `X` of design points, and return a
`Posterior` object (which would typically describe the (joint) probability distribution of the
model output(s) over the design points in `X`).

## botorch Models for Standard Use Cases
botorch provides several GPyTorch models to cover the standard Bayesian optimization use cases. All of these models use Matern 5/2 ARD kernels:
* `SingleTaskGP`: a single-task, single-output exact GP that infers the the homoskedastic noise level (no noise observations)
* `ConstantNoiseGP`: a single task, single-outputexact GP that uses a constant observed noise level (requires noise observation required)
* `HeteroskedasticSingleTaskGP`: a single task and single-output exact GP that models heteroskedastic noise (requires noise observations)

* `MultiOutputGP`: A single-task, multi-output model where outcomes are modeled independently (the outcomes can be modeled using any of the single output GPs)
* `MultiTaskGP`: A Hadamard multi-task, multi-output model using an ICM kernel.

## Implementing Custom Models
Check out the tutorial: [Using a custom botorch model in Ax](../tutorials/custom_botorch_model_in_ax)
