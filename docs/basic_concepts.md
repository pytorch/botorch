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

Objectives are modules that allow for transforming model outputs. Typical use
cases for this are scalarization of the outputs of a multi-output model
(e.g. to explore the Pareto Frontier of multiple outcomes in a ParEGO-like
fashion [^ParEGO]), or optimization subject to outcome constraints (in a noisy
setting this is typically achieved by weighting the objective by the probability
of feasibility [^NoisyEI]).

When using traditional analytic formulations of acquisition functions, one has
to be quite careful to make sure that the transformation results in a posterior
distribution of the transformed outputs that still satisfies the assumptions of
the analytic formulation. For instance, to use standard Expected Improvement on
a transformed output of a model, the transformation needs to be affine (this is
because Gaussians are closed under affine transformations).

When using (q)MC-based acquisition functions, however, much fewer assumptions
are required, and one can apply general transformations to the model outputs
with relative impunity (as long as one makes sure that gradients can be
back-propagated through the transformation).

Instead of passing the model output through an objective, one could of course
also model the transformed objective directly. But this would potentially
involve refitting the model numerous times to try different objectives on the
outputs.

[^ParEGO]: J. Knowles. *ParEGO: A hybrid algorithm with on-line landscape
approximation for expensive multiobjective optimization problems.*
IEEE Transactions on Evolutionary Computation, 2006.

[^NoisyEI]: B. Letham, B. Karrer, G. Ottoni and Bakshy, E. *Constrained Bayesian
Optimization with Noisy Experiments.* Bayesian Analysis, 2018.


## Acquisition Functions

In Bayesian Optimization, acquisition functions are heuristics employed to
evaluate the usefulness of one of more design points for achieving the objective
of maximizing the underlying black box function.

Some of these acquisition functions have closed-form solutions under typical
Gaussian posteriors, but many of them (especially when assessing the joint
value of multiple points) do not. botorch supports both analytic as well as
(quasi-) Monte-Carlo based acquisition functions. It comes bundled with the
most common ones out of the box, and makes it very easy to implement and test
new ones.

### Analytic

Analytic acquisition functions are simple enough to allow an explicit expression
in terms of the summary statistics of the posterior distribution at the
evaluated point(s).

A classic such acuqisition funciton is Expected Improvement, given by

$$ EI(x) = \mathbb{E}[\max(f(x) - f_{max}, 0) | ] $$

where $f(x) \sim \mathcal{N}(\mu(x), \sigma^2(x))$ with $\mu(x)$ and $\sigma(x)$
the posterior mean and variance of $f$ at the point $x$, and where $f_{max}$ is
the best function value observed so far (assuming noiseless observations).

**TODO**


### Monte-Carlo

**TODO**

## Samplers
