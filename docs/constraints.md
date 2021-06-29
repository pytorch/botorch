---
id: constraints
title: Constraints
---

BoTorch supports two distinct types of constraints: Parameter constraints
and outcome constraints.


### Parameter Constraints

Parameter constraints are constraints on the input space that restrict the
values of the generated candidates. That is, rather than just living inside
a bounding box defined by the `bounds` argument to `optimize_acqf` (or its
derivates), candidate points may be further constrained by linear (in)equality
constraints, specified by the `inequality_constraints` and `equality_constraints`
arguments to `optimize_acqf`.

Parameter constraints are used e.g. when certain configurations are infeasible
to implement, or would result in excessive costs. These constraints do not affect
the model directly, only indirectly in the sense that all newly generated and
later observed points will satisfy these constraints. In particular, you may
have a model that is fit on points that do not satisfy a certain set of parameter
constraints, but still generate candidates subject to those constraints.


### Outcome Constraints

In the context of Bayesian Optimization, outcome constraints usually mean
constraints on some (black-box) outcome that needs to be modeled, just like
the objective function is modeled by a surrogate model. Various approaches
for handling these types of constraints have been proposed, a popular one that
is also adopted by BoTorch (and available in the form of `ConstrainedMCObjective`)
is to use variant of expected improvement in which the improvement in the objective
is weighted by the probability of feasibility under the (modeled) outcome
constraint ([^Gardner2014], [^Letham2017]).

See the [Closed-Loop Optimization](../tutorials/closed_loop_botorch_only)
tutorial for an example of using outcome constraints in BoTorch.



[^Gardner2014]: J.R. Gardner, M. J. Kusner, Z. E. Xu, K. Q. Weinberger and
J. P. Cunningham. Bayesian Optimization with Inequality Constraints. ICML 2014.

[^Letham2017]: B. Letham, B. Karrer, G. Ottoni and E. Bakshy. Constrained Bayesian optimization with noisy experiments. Bayesian Analysis 14(2), 2019.
