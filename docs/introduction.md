---
id: introduction
title: Introduction
---

BoTorch (pronounced like "blow-torch") is a library for Bayesian Optimization
research built on top of PyTorch, and is part of the PyTorch ecosystem.

Bayesian Optimization (BO) is an established technique for sequential optimization
of costly-to-evaluate black-box functions. It can be applied to a wide variety
of problems, including Machine Learning (tuning algorithms' hyper-parameters),
A/B testing, as well as many scientific and engineering problems.

BoTorch is best used in tandem with [Ax](https://github.com/facebook/Ax),
Facebook's open-source adaptive experimentation platform, which provides an
easy-to-use interface for defining, managing and running sequential experiments,
while handling (meta-)data management, transformations, and systems integration.


## Why BoTorch

#### Improved Developer Efficiency

BoTorch significantly improves developer efficiency by utilizing quasi-Monte-Carlo
based acquisition functions (by ways of the "re-parameterization trick") in
conjunction with PyTorch's auto-differentiation. This makes it very easy to try
out new algorithms without having to go through pen & paper math to derive
analytic expressions for acquisition functions and their gradients.
More importantly, it opens the door for novel approaches that do not admit
analytic solutions, including batch acquisition functions and proper handling of
rich multi-output models with multiple correlated outcomes.

BoTorch follows the same modular "un-framework" design philosophy as PyTorch,
which makes it very easy for users to swap out or rearrange individual components
in order to customize all aspects of their algorithm, thereby empowering
researchers to do state-of-the art research on modern Bayesian Optimization
methods.


#### State-of-the-art Modeling

Bayesian Optimization traditionally relies heavily on Gaussian Process (GP)
models, which provide well-calibrated uncertainty estimates.
BoTorch provides first-class support for [GPyTorch](https://gpytorch.ai/), a
library for efficient and scalable GPs implemented in PyTorch (and contributed
to significantly by the BoTorch authors).
This enables using GP models for problems that have traditionally not been
amenable to Bayesian Optimization techniques.

In addition, BoTorch's lightweight APIs are model-agnostic (they can for example
work with [Pyro](http://pyro.ai/) models), and support optimization of acquisition
functions over any kind of posterior distribution, as long as it can be sampled from.


#### Harnessing the Power of PyTorch

Built on PyTorch, BoTorch harnesses the power of auto-differentiation, native
support for highly parallelized modern hardware (such as GPUs) using device-agnostic
code, and a dynamic computation graph that facilitates interactive development.

BoTorch's modular design allows for a great deal of modeling flexibility for
including deep architectures through seamless integration with generic PyTorch
modules. Importantly, being able to back-propagate gradients through the full
model allows for joint training of GP and Neural Network modules, and end-to-end
gradient-based optimization of acquisition functions operating on differentiable
models.


#### Bridging the Gap Between Research and Production

BoTorch implements modular building blocks for modern Bayesian optimization.
It bridge the gap between research and production by being a very flexible
research framework, but at the same time, a reliable, production-grade
implementation that integrates well with other higher-level platforms,
specifically [Ax](https://github.com/facebook/Ax).


## Target Audience

The primary audience for hands-on use of BoTorch are researchers and
sophisticated practitioners in Bayesian Optimization and AI.

We recommend using BoTorch as a low-level API for implementing new algorithms
for Ax. Ax has been designed to be an easy-to-use platform for end-users, which
at the same time is flexible enough for Bayesian Optimization researchers to
plug into for handling of feature transformations, (meta-)data management,
storage, etc. See [Using BoTorch with Ax](../botorch_and_ax) for more details.

We recommend that end-users who are not actively doing research on Bayesian
Optimization simply use Ax.
