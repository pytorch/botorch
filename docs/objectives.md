---
id: objectives
title: Objectives
---

Objectives are modules that allow for transforming model outputs. Typical use
cases for this are scalarization of the outputs of a multi-output model (see
e.g.[^RandScal]), or optimization subject to outcome constraints (in a noisy
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

[^RandScal]: B. Paria, K. Kandasamy, and B. Póczos. *Flexible Multi-Objective
Bayesian Optimization Approach using Random Scalarizations.* ArXiv e-prints, 2018.

[^NoisyEI]: B. Letham, B. Karrer, G. Ottoni and Bakshy, E. *Constrained Bayesian
Optimization with Noisy Experiments.* Bayesian Analysis, 2018.
