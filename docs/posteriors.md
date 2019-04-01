---
id: posteriors
title: Posteriors
---

**TODO:** Describe the role of posteriors
* botorch is semi-agnostic to form of posterior
  - for analytic acq. functions we assume multivariate Gaussian (could possibly
    use others though)
  - for MC-based acquisition functions, the only thing we need to be able to do
  is sample from the posterior
* botorch comes with GPyTorchPosterior implementations - in most cases you
  won't have to worry about implementing a new Posterior
* show how an implicit posterior could be used (e.g. by simply wrapping the
  model)


## Draft Content

botorch `Posterior` objects are a layer of abstraction that allow to separate
the specific model used from the evaluation (and subsequent optimization) of
acquisition functions. In the simplest case, posteriors are a lightweight
wrapper around explicit distribution objects from `torch.distributions` (or
`gpytorch.distributions`). However, they may be much more complex than that.
For instance, a posterior could be represented implicitly by some base
distribution mapped through some Neural Network. As long as one can sample
from this distribution, the qMC-based acquisition functions can work with it.
