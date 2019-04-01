---
id: samplers
title: MC Samplers
---

**TODO:** Describe the role of MC samplers
* The reparameterization trick relies on transforming samples $\epsilon$ from
  some base distribution.
* A Sampler is an object that provides these base samples in a convenient way.
* When using GPs, the classic parameterization is $\mu(x) + L(x) \epsilon$,
  where $\epsilon$ are i.i.d normal ($\mu$ is the mean of the posterior, and
  $L(x)$ is a root decomposition of the covariance matrix such that
  $L(x)L(x)^T = \Sigma(x)$).
* botorch comes with both MC (`IIDNormalSampler`) and qMC
  (`SobolQMCNormalSampler`) samplers for `N(0, I)` samples.
* if you need other base samples, it's easy to implement a custom sampler.
