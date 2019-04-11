---
id: objectives
title: Objectives
---


An `Objective` is a `Module` that allows for convenient transformation model outputs.
Typical use cases for this are scalarization of the outputs of a multi-output model (see
e.g. [^RandScal]), or optimization subject to outcome constraints, which can be
achieved by weighting the objective by the probability of feasibility [^NoisyEI].

When using classic analytic formulations of acquisition functions, one has
to be quite careful to make sure that the transformation results in a posterior
distribution of the transformed outputs that still satisfies the assumptions of
the analytic formulation. For instance, to use standard Expected Improvement on
a transformed output of a model, the transformation needs to be affine
 (because Gaussians are closed under affine transformations).

When using MC-based acquisition functions, however, much fewer assumptions
are required, and one can apply general transformations to the model outputs
with relative impunity so long as one makes sure that gradients can be
back-propagated through the transformation. botorch implements several MC-based objectives, including `LinearMCObjective` for linear combinations of objectives, and `ConstrainedMCObjective` for constrained objectives (via softmax).

[^RandScal]: B. Paria, K. Kandasamy, and B. Póczos. A Flexible Multi-Objective
Bayesian Optimization Approach using Random Scalarizations. ArXiv, 2018.

[^NoisyEI]: B. Letham, B. Karrer, G. Ottoni and Bakshy, E. Constrained Bayesian
Optimization with Noisy Experiments. Bayesian Analysis, 2018.
