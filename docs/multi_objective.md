---
id: multi_objective
title: Multi-Objective Bayesian Optimization
---


BoTorch provides first-class support for Multi-Objective (MO) Bayesian Optimization (BO) including implementations of the novel [`qExpectedHypervolumeImprovement`](../api/acquisition.html#botorch.acquisition.multi_objective.monte_carlo.qExpectedHypervolumeImprovement) (q-EHVI), q-ParEGO, and analytic [`ExpectedHypervolumeImprovement`](../api/acquisition.html#mcacquisitionfunction)(../api/acquisition.html#botorch.acquisition.multi_objective.analytic.ExpectedHypervolumeImprovement) (EHVI) with gradients via auto-differentiation acquisition functions [^qEHVI].

The goal in MOBO is learn the *Pareto front*: the set of optimal trade-offs, where an improvement in one objective means deteriorating another objective. Botorch provides implementations for a number of acquisition functions specifically for the multi-objective scenario, as well as generic interfaces for implemented new multi-objective acquisition functions.

## Multi-Objective Acquisition Functions
MOBO leverages many advantages of BoTorch to make provide practical algorithms for computationally intensive and analytically intractable problems. For example, analytic EHVI has no known analytical gradient for when there are more than two objectives, but BoTorch computes analytic gradients for free via auto-differentiation, regardless of the number of objectives [^qEHVI].

For analytic and MC-based MOBO acquisition functions like qEHVI and qParEGO, BoTorch leverages GPU acceleration and quasi-second order methods for acquisition optimization for efficient computation and optimization in many practical scenarios [^qEHVI]. The MC-based acquisition functions support using the sample average approximation for rapid convergence [^BoTorch].

All analytic MO acquisition functions derive from [`MultiObjectiveAnalyticAcquisitionFunction`](../api/acquisition.html#botorch.acquisition.multi_objective.analytic.MultiObjectiveAnalyticAcquisitionFunction) and all MC-based acquisition functions derive from [`MultiObjectiveMCAcquisitionFunction`](../api/acquisition.html#botorch.acquisition.multi_objective.monte_carlo.MultiObjectiveMCAcquisitionFunction). These abstract classes easily integrate with BoTorch's standard optimization machinery.

Additionally, q-ParEGO is trivially implemented using an augmented Chebyshev scalarization as the objective with the [`qExpectedImprovement`](../api/acquisition.html#qexpectedimprovement) acquisition function. Botorch provides a [`get_chebyshev_scalarization`](../api/utils.html#botorch.utils.multi_objective.scalarization.get_chebyshev_scalarizationconvenience) convenience function for generating these scalarizations. In the batch setting evaluation, q-ParEGO uses a different scalarization per candidate [^qEHVI], and optimizing a batch of candidates, each with a different scalarization, is supported using the [`optimize_acqf_list`](../api/optim.html#botorch.optim.optimize.optimize_acqf_list) function.

For a more in-depth example using these acquisition functions, check out the [Multi-Objective Bayesian Optimization tutorial notebook](../tutorials/multi_objective_bo).

## Multi-Objective Utilities

BoTorch provides several utility functions for evaluating performance in MOBO including a method for computing the Pareto front [`is_non_dominated`](../api/utils.html#botorch.utils.multi_objective.pareto.is_non_dominated) and a class for efficiently computing of the [`Hypervolume`] (../api/utils.html#botorch.utils.multi_objective.hypervolume.Hypervolume) dominated by a provided set of points using a dimension sweep algorithm [^Fonseca].


[^qEHVI]: S. Daulton, M. Balandat, and E. Bakshy. Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization. ArXiv e-prints,
[arXiv:2006.05078](https://arxiv.org/abs/2006.05078), Jun. 2020.

[^BoTorch]: M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson,
and E. Bakshy. BoTorch: Programmable Bayesian Optimization in PyTorch. arXiv e-prints,
[arXiv:1910.06403](https://arxiv.org/abs/1910.06403), Oct. 2019.

[^Fonseca] C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary Computation, pages 1157-1163, Vancouver, Canada, July 2006.
