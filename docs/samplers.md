---
id: samplers
title: Monte Carlo Samplers
---

Monte Carlo-based acquisition functions rely on the reparameterization trick, which transforms some set of $\epsilon$ from some base distribution into a target distribution.  For example, when drawing posterior samples from a Gaussian process, the classic parameterization is $\mu(x) + L(x) \epsilon$, where $\epsilon$ are i.i.d normal, $\mu$ is the mean of the posterior, and $L(x)$ is a root decomposition of the covariance matrix such that $L(x)L(x)^T = \Sigma(x)$.

Exactly how base samples are generated when using the reparameterization trick can have substantial effects on the convergence of gradients estimated from these samples. Because of this, botorch implements a generic module capable of flexible sampling from any type of probabilistic model.

A `MCSampler` is a `Module` that provides base samples from a `Posterior` object.  These
samplers may then in turn be used in conjunction with MC-based acquisition functions. botorch includes two types of MC samplers for sampling isotropic normal deviates: a vanilla, normal sampler (`IIDNormalSampler`) and randomized quasi-monte carlo sample (`SobolQMCNormalSampler`).

For most use cases, we recommend using `SobolQMCNormalSampler`, as it tends to produce more accurate gradient estimates with much fewer samples relative to the `IIDNormalSampler`.  If you'd like to experiment with alternative sampling procedures, please see the source code for `SobolQMCNormalSampler` as an example.