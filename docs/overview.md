---
id: overview
title: Overview
---

For a high-level view of what botorch tries to be, see our
[design philosophy](design_philosophy).


## Black-Box Optimization

![Black Box Optimization](assets/overview_blackbox.png)

On a high level, the problem that Bayesian Optimization is trying to solve is
to optimize some expensive-to-evaluate black box function $f$. The only thing we
can do is evaluate this function on a sequence of test points (and possibly also
on multiple test points in parallel).


## Bayesian Optimization

![Bayesian Optimization](assets/overview_bayesopt.png)

#### Models

In order to be able to perform this optimization, we need a way of extrapolating
out belief about what $f$ looks like at points we have not yet evaluated. In
Bayesian Optimization this is referred to as the *Surrogate Model*. Importantly,
the surrogate model should be able to quantifying the uncertainty of its
predictions in form of a **Posterior** distribution over function values $f(x)$
at points $x$.

In Bayesian Optimization, the model typically used is a Gaussian Process (GP),
in which case the posterior is a multi-variate Gaussian.
**TODO: Say a few things about GPs**
However, while GPs have been a very successful modeling approach, it's also
possible to use other model types (**TODO: Expand on this**).

botorch makes no general assumptions on what kind of model is being used,
so long as is able to produce a posterior over outputs given an input $x$.
See [Models](models.md#models) for more details on models in botorch,


#### Posteriors

Posteriors represent the "belief" a model has about the function values based on
the data it has been trained with. That is, the posterior distribution over the
outputs conditional on the data seen so far (hence the name).

When using a GP model, the posterior is given explicitly as a multivariate
Gaussian (fully parameterized by its mean and covariance matrix). In other cases,
the posterior may be implicit in the model and not easily described by a
small set of parameters.

botorch abstracts away from the particular form of the model posterior by
providing a simple [Posteriors](posteriors.md#posteriors) API.


#### Acquisition Functions

Acquisition functions are heuristics employed to evaluate the usefulness of one
of more design points for achieving the objective of maximizing the underlying
black box function.

Some of these acquisition functions have closed-form solutions under Gaussian
posteriors, but many of them (especially when assessing the joint value of
multiple points in parallel) do not. In the latter case, one can resort to using
Monte-Carlo (MC) sampling in order to approximate the acquisition function.

botorch supports both analytic as well as (quasi-) Monte-Carlo based acquisition
functions. It provides an API for [Acquisition Functions](acquisition.md) that
abstracts away from the particular type, so that optimization can be performed
on the same objects.


## Evaluating Monte-Carlo Acquisition Functions

![Monte-Carlo Acquisition Functions](assets/overview_mcacquisition.png)

**TODO**

#### Objectives

[Objectives](objectives.md#objectives)

**TODO**


## The Reparameterization Trick

![Reparameterization Trick](assets/overview_reparameterization.png)

**TODO**

More about [Samplers](samplers.md).
