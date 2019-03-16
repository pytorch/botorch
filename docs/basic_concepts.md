---
id: basic_concepts
title: Basic Concepts
---

For a high-level view of what botorch tries to be, see our
[design philosophy](design_philosophy).

botorch works with the following primitives;
- [Models](#models)
- [Posteriors](#posteriors)
- [Objectives](#objectives)
- [Acquisition Functions](#acquisition-functions)
- [Samplers (optional)](#samplers)


**TODO: Illustration of how things work together**


Models play an essential role in Bayesian Optimization. They provide a surrogate
function for the actual underlying black box function that one is trying to
optimize. In botorch, models map a set of design points to a posterior
probability distribution of the model output(s) over the design points.

In Bayesian Optimization, the model used is traditionally a Gaussian Process,
in which case the posterior distribution, by definition, is a multivariate
normal.

However, except for some of the analytic Acquisition functions in the
`botorch.acquisition.analytic` module, botorch makes no assumption on the model
being a GP, or on the posterior being a multivariate normal.
The only requirement for using botorch's (quasi-) Monte-Carlo based acquisition
functions is that the model returns a `Posterior` object that implements an
`rsample` method, which can be used to draw samples from the posterior.
In order to use gradient-based optimization algorithms, this implementation
should allow back-propagating gradients through the samples to the model input.




## Models

botorch models are PyTorch modules required to adhere to a lightweight API.
Their `forward` methods take in a Tensor `X` of design points, and return a
`Posterior` object that describes the (joint) probability distribution of the
model output(s) over the design points in `X`.

**TODO**


## Posteriors

botorch `Posterior` objects are a layer of abstraction that allow to separate
the specific model used from the evaluation (and subsequent optimization) of
acquisition functions. In the simplest case, posteriors are a lightweight
wrapper around explicit distribution objects from `torch.distributions` (or
`gpytorch.distributions`). However, they may be much more complex than that.
For instance, a posterior could be represented implicitly by some base
distribution mapped through some Neural Network. As long as one can sample
from this distribution, the qMC-based acquisition functions can work with it.


**TODO**


## Objectives

Objectives are modules that can be applied to model outputs, allowing for an
easy way to transform model outputs. When using traditional analytic
formulations of acquisition functions, one has to be quite careful to make
sure that the transformation results in a posterior distribution of the
transformed outputs that still satisfies the assumptions of the analytic
formulation. For instance, to use standard Expected Improvement on a transformed
output of a model, the transformation needs to be affine (this is because
Gaussians are closed under affine distributions).

When using (q)MC-based acquisition functions, however, this kind of care is not
required and once can apply general transformations (as long as one makes sure
that gradients can be back-propagated through this transformations).

Instead of passing the model output through an objective, one could of course
also model the transformed objective directly. But this would potentially
involve refitting the model numerous times to try different objectives on the
outputs. Typical use cases for this are scalarization of the outputs of a
multi-output model (e.g. to explore the Pareto Frontier of multiple outcomes in
a ParEGO-like fashion), or optimization subject to outcome constraints,
where one may want to try out different bounds on some of the outputs.


## Acquisition Functions


### Analytic

**TODO**


### Monte-Carlo

**TODO**

## Samplers
