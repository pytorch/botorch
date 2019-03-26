---
id: optimization
title: Optimization
---

#### Using `scipy` optimization methods used on `torch` tensors

The default method used by botorch to optimize the acquisition functions and generate a q-batch of candidates is `gen_candidates_scipy`. Given a set of starting points (for multiple restarts) and an acquisition function, `gen_candidates_scipy` makes use of `scipy.optimize.minimize` for optimization, via either the `L-BFGS-B` or `SLSQP` routines. A `numpy` to `torch` converter is used in the following way: when the optimizer calls for the acquisition value and its gradient at a candidate q-batch `numpy` array `x`, the point `x` is first converted to a `torch` tensor and passed to the acquisition function for evaluation. PyTorch's autograd capability is then used to obtain the gradient, and both the acquisition value and gradient are converted back to `numpy` arrays and returned to the optimizer.

#### Using `torch` optimizers

A `torch` optimizer such as `torch.optim.Adam` or `torch.optim.SGD` can also be used directly, without the need to perform `numpy` conversion. These first-order gradient-based optimizers are particularly useful for the case when the acquisition function is stochastic, where `L-BFGS` or Sequential Least-Squares Programming should not be applied. The function `gen_candidates_torch` provides an interface for `torch` optimizers and handles bounding. See the example notebook [LINK TO ACQUISITION TUTORIAL] for a tutorial on trying different optimizers.

#### Multiple random restarts

Acquisition functions are often difficult to optimize as they are generally non-convex and often flat (e.g., EI), so `botorch` makes use of multiple random restarts to improve optimization quality. Each restart can be thought of as an optimization routine within a local region; thus, taking the best result over many restarts can help provide an approximation to the global optimization objective. The function `gen_batch_initial_candidates` implements heuristics for choosing a set of initial restart locations (candidates).

Rather than independently (and sequentially) optimize from each initial restart candidate, the function `gen_candidates_scipy` takes advantage of batch mode evaluation (t-batches) of the acquisition function to solve a single `b x q x d`-dimensional optimization problem, where the objective is defined as the sum of the `b` individual q-batch acquisition values. The wrapper function `joint_optimize` uses `get_best_candidates` to process the output of `gen_candidates_scipy` to return the best point found over the random restarts. For reasonable values of `b` and `q`, jointly optimizing over random restarts can significantly reduce wall time, while maintaining high quality solutions.

#### Joint versus sequential candidate generation for qNEI

In batch Bayesian optimization with noisy expected improvement (qNEI), `q` design points need to be selected for experimentation. The qNEI objective calls for *joint* optimization over the `q` design points (i.e., solve an optimization problem with a `q x d`-dimensional decision), but when `q` is large, one might also consider *sequentially* selecting the `q` points using successive conditioning and solving `q` optimization problems, each with a `d`-dimensional decision. The functions `joint_optimize` and `sequential_optimize` provide for this functionality.

Our empirical observations of the *closed-loop BO performance* for `q = 5` show that joint optimization and sequential optimization have similar performance, but sequential optimization comes at a steep cost in wall time (generally 2-6x). Therefore, for moderately sized `q`, a reasonable default option is to use joint optimization. However, it is important to note that as `q` increases, the performance of joint optimization can be hindered by the `q x d`-dimensional problem and sequential optimization might be preferred. See [^Wilson2018] for further discussion relating the effectiveness of sequential optimization of acquisition functions to greedy maximization of submodular functions.

[^Wilson2018]: J. Wilson, F. Hutter, M. Deisenroth. *Maximizing Acquisition Functions for Bayesian Optimization.* NeurIPS, 2018.
