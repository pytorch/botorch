---
id: optimization
title: Optimization
---

## Model Fitting

BoTorch provides the convenience method
[`fit_gpytorch_mll()`](../api/fit.html#botorch.fit.fit_gpytorch_mll) for
fitting GPyTorch models (optimizing model hyperparameters) using L-BFGS-B via
`scipy.optimize.minimize()`. We recommend using this method for exact GPs, but
other optimizers may be necessary for models with thousands of parameters or
observations.

## Optimizing Acquisition Functions

#### Using scipy Optimizers on Tensors

The default method used by BoTorch to optimize acquisition functions is
[`gen_candidates_scipy()`](../api/generation.html#botorch.generation.gen.gen_candidates_scipy).
Given a set of starting points (for multiple restarts) and an acquisition
function, this optimizer makes use of `scipy.optimize.minimize()` for
optimization, via either the L-BFGS-B or SLSQP routines.
`gen_candidates_scipy()` automatically handles conversion between `torch` and
`numpy` types, and utilizes PyTorch's autograd capabilities to compute the
gradient of the acquisition function.

#### Using torch Optimizers

A `torch` optimizer such as `torch.optim.Adam` or `torch.optim.SGD` can also be
used directly, without the need to perform `numpy` conversion. These first-order
gradient-based optimizers are particularly useful for the case when the
acquisition function is stochastic, where algorithms like L-BFGS or SLSQP that
are designed for deterministic functions should not be applied. The function
[`gen_candidates_torch()`](../api/generation.html#botorch.generation.gen.gen_candidates_torch)
provides an interface for `torch` optimizers and handles bounding.
See the example notebooks
[here](../tutorials/compare_mc_analytic_acquisition) and
[here](../tutorials/optimize_stochastic) for tutorials on how to use different
optimizers.


### Multiple Random Restarts

Acquisition functions are often difficult to optimize as they are generally
non-convex and often flat (e.g., EI), so BoTorch makes use of multiple random
restarts to improve optimization quality. Each restart can be thought of as an
optimization routine within a local region; thus, taking the best result over
many restarts can help provide an approximation to the global optimization
objective. The function
[`gen_batch_initial_conditions()`](../api/optim.html#botorch.optim.optimize.gen_batch_initial_conditions)
implements heuristics for choosing a set of initial restart locations (candidates).

Rather than optimize sequentially from each initial restart
candidate, `gen_candidates_scipy()` takes advantage of batch mode
evaluation (t-batches) of the acquisition function to solve a single
$b \times q \times d$-dimensional optimization problem, where the objective is
defined as the sum of the $b$ individual q-batch acquisition values.
The wrapper function
[`optimize_acqf()`](../api/optim.html#botorch.optim.optimize.optimize_acqf)
uses
[`get_best_candidates()`](../api/generation.html#botorch.generation.gen.get_best_candidates)
to process the output of `gen_candidates_scipy()` and return the best point
found over the random restarts. For reasonable values of $b$ and $q$, jointly
optimizing over random restarts can significantly reduce wall time by exploiting
parallelism, while maintaining high quality solutions.


### Joint vs. Sequential Candidate Generation for Batch Acquisition Functions

In batch Bayesian Optimization $q$ design points are selected for parallel
experimentation. The parallel (qEI, qNEI, qUCB, qPI) variants of acquisition
functions call for *joint* optimization over the $q$ design points (i.e., solve
an optimization problem with a $q \times d$-dimensional decision), but when $q$
is large, one might also consider *sequentially* selecting the $q$ points using
successive conditioning on so-called "fantasies", and solving $q$ optimization
problems, each with a $d$-dimensional decision. The functions
[`optimize_acqf()`](../api/optim.html#botorch.optim.optimize.optimize_acqf)
by default performs joint optimization; when specifying `sequential=True` it
will perform sequential optimization.

Our empirical observations of the *closed-loop Bayesian Optimization performance*
for $q = 5$ show that joint optimization and sequential optimization have similar
optimization performance on some standard benchmarks, but sequential optimization
comes at a steep cost in wall time (2-6x in our tests). Therefore, for moderately
sized $q$, a reasonable default option is to use joint optimization.

However, it is important to note that as $q$ increases, the performance of joint
optimization can be hindered by the harder $q \times d$-dimensional problem, and
sequential optimization might be preferred. See [^Wilson2018] for further
discussion on how sequential greedy maximization is an effective strategy for
common classes of acquisition functions.

[^Wilson2018]: J. Wilson, F. Hutter, M. Deisenroth. Maximizing Acquisition
Functions for Bayesian Optimization. NeurIPS, 2018.
