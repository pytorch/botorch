---
id: posteriors
title: Posteriors
---

**TODO: Describe the basic Posterior API**

botorch `Posterior` objects are a layer of abstraction that allow to separate
the specific model used from the evaluation (and subsequent optimization) of
acquisition functions. In the simplest case, posteriors are a lightweight
wrapper around explicit distribution objects from `torch.distributions` (or
`gpytorch.distributions`). However, they may be much more complex than that.
For instance, a posterior could be represented implicitly by some base
distribution mapped through some Neural Network. As long as one can sample
from this distribution, the qMC-based acquisition functions can work with it.
