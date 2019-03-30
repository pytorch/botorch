---
id: introduction
title: Introduction
---


botorch (“Bayesian Optimization in PyTorch”) is a library for Bayesian
Optimization research built on top of PyTorch, and is part of the PyTorch
ecosystem.

Bayesian Optimization is an established technique for sequential optimization
of costly-to-evaluate black-box functions. It can be applied to a wide variety
of problems, including Machine Learning (tuning algorithms' hyper-parameters),
robotics, and A/B testing.

## Why botorch?

#### Improved developer efficiency
Being built on top of PyTorch, `botorch` harnesses the power of native GPU
acceleration and auto-differentiation, while allowing for a great deal of
flexibility through seamless integration with generic PyTorch modules.
By doing so, it significantly improves developer efficiency, as it is very easy
to implement and try out new algorithms. As a result, substantially less
analytical derivations (in particular gradients of acquisition functions) are
necessary to implement new ideas.

#### State-of-the-art modeling
Bayesian Optimization traditionally relies heavily on Gaussian Process (GP)
models, which are often difficult to work with, don't scale to large datasets,
and don't integrate well with modern deep learning innovations.
While `botorch`'s basic API is model-agnostic (it can for example, work with
[Pyro](http://pyro.ai/) in theory), the current implementation focuses on using
[GPyTorch](https://gpytorch.ai/), a package for GPs implemented in PyTorch.
This opens the door for tackling problems that have traditionally not been
amenable to Bayesian Optimization.

#### Bridging the gap between research and production
`botorch` is a modular, extensible library that provides the building blocks for
Bayesian Optimization. It aims to bridge the gap between being a very flexible
research platform and at the same time providing a reliable, production-grade
implementation. `botorch` strives for excellent code quality with enforcing
strict style rules, full type annotations, and comprehensive unit test coverage.


## Target Audience
The primary audience for hand-on use of `botorch` are researchers and
sophisticated practitioners in Bayesian Optimization and Machine Learning /
Artificial Intelligence. For broader use, tried and tested algorithms are
packaged into our companion open-source release, Ax, a library for sequential
experimentation that provides a higher-level interface and does not require
the user to understand ins and outs of Bayesian Optimization.
