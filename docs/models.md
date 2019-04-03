---
id: models
title: Models
---

**TODO:** Describe the role of models
* botorch is model-agnostic (made easy b/c of pytorch backpropping)
* botorch comes with gpytorch models covering standard use cases
* explain the basics of the model API
* illustrate how to implement custom models (link to tutorial)


# Draft Content

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

botorch models are PyTorch modules required to adhere to a lightweight API.
Their `forward` methods take in a Tensor `X` of design points, and return a
`Posterior` object that describes the (joint) probability distribution of the
model output(s) over the design points in `X`.
