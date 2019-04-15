---
id: posteriors
title: Posteriors
---


## Posteriors in botorch

A botorch `Posterior` object is a layer of abstraction that separates the specific model used from the evaluation (and subsequent optimization) of acquisition functions. In the simplest case, a posterior is a lightweight wrapper around an explicit distribution object from `torch.distributions` (or `gpytorch.distributions`). However, a botorch `Posterior` can be any distribution (even an implicit one), so long as one can sample from that distribution. For example, a posterior could be represented implicitly by some base distribution mapped through some Neural Network.

While the analytic acquisition functions assume that the posterior is a multivariate Gaussian, the MC-based acquisition functions do not make any assumptions about the underlying distribution. Rather, the MC-based acquisition functions only require that the posterior can be sampled from. As long as posterior implements the `Posterior` interface, the MC-based acquisition functions can work with it (when using gradient-based acquisition function optimization then it must be possible to back-propagate gradients through the samples).

## botorch `Posterior` for Standard Use Cases

Traditionally in Bayesian optimization, the posterior distribution is a multivariate Gaussian. botorch includes a multivariate Gaussian `GPyTorchPosterior` class that wraps a `gpytorch.distributions.MultivariateNormal`, so for standard uses cases, it is not necessary to implement a new `Posterior`.

## Implementing a Custom Posterior
**TODO**: show how an implicit posterior could be used (e.g. by simply wrapping the model)
