---
id: design_philosophy
title: Design Philosophy
---

## Main Tenets

botorch adheres to the following main design tenets:

#### Modularity & Simplicity
- make it easy for researchers to develop and implement new ideas (by following
  an "unframework" design philosophy & making heavy use of auto-differentiation)
- facilitate model-agnostic Bayesian Optimization (by maintaining lightweight
  APIs and first-class support for Monte-Carlo evaluated acquisition functions)

Most botorch components are `torch.nn.Module` instances, and so it is very easy
for users familiar with PyTorch to implement new components and combine existing
components together.

#### Performance & Scalability
- achieve high levels of performance across different platforms while using the
  same device-agnostic code (by using highly parallelized batch operations in
  conjunction with native support for modern hardware such as GPUs)
- extend the applicability of Bayesian Optimization to very large problems (by
  harnessing scalable modeling frameworks such as [gpytorch](https://gpytorch.ai/))


## Batching, Batching, Batching

**batch** - noun [C] /bætʃ/ :
*A quantity (as of persons or things) considered as a group.*

Batching (as in batching data, batching computations) is a central component to
all modern deep learning platforms. Hence it should come as no surprise that it
also plays a critical role in the design of botorch.

In botorch, we deal with various different kinds of batches:

1. A batch of candidate points `X` to be evaluated in parallel on the black-box
   function we are trying optimize. In botorch, we refer to this kind of batch
   as a "`q`-batch".
2. A batch of `q`-batches to be evaluated in parallel on the surrogate model of
   the black-box function. These facilitate fast evaluation on modern hardware
   such as GPUs. botorch refer to these batches as "`t`-batches" (as in
   "torch-batches").
3. A batched surrogate model, each batch of which models a different output.
   This kind of batching also aims to exploit modern hardware architecture.

Note that none of these notions of batch pertains to the batching of *training
data*, which is commonly done in training Neural Network models (sometimes
called "mini-batching"). botorch aims to be agnostic with regards to the
particular model used - so while model fitting may indeed be performed using
batch training, botorch itself abstracts away from this.

For the purposes of botorch, batching is primarily two things:
- a way of logically grouping multiple design configurations that are to be
  evaluated in parallel on the underlying black-box function
- a powerful design principle that allows harnessing highly efficient
  computations in PyTorch while allowing for device-agnostic code

For more detail on the different batch notions in botorch, take a look at the
[More on Batching](#more_on_batching) section.



## Optimizing Acquisition Functions

One place where botorch takes a somewhat different approach than PyTorch is in
optimizing acquisition functions.

In PyTorch, modules typically map (batches of) data to an output, where the
mapping is parameterized by the parameters of the modules (often the weights
of a Neural Network). Fitting the model means optimizing some loss (which is
defined w.r.t. the underlying distribution of the data). As this distribution
is unknown, one cannot directly evaluate this function. Instead, the empirical
loss function, i.e. the loss evaluated on all data available, is minimized
(this provides an unbiased estimate of the true loss). But even evaluating the
empirical loss function exactly is often impossible or at least impractical
because the amount of input data cannot be processes at once. Thus one typically
subsets ("mini-batches") the data, and by doing so is left with a stochastic
approximation of the empirical loss function (with each data-batch a sample).

In botorch, `AcquisitionFunction` modules map an input design `X` to the
acquisition function value. Optimizing the acquisition function means optimizing
the output over the possible values of `X`. If the acquisition function is
deterministic, so is the optimization problem.

For large Neural Network models, the number of optimization variables is very
high, and can be in the hundreds of thousands or even millions of parameters.
Currently, the only scalable optimization algorithms for this purpose are
first-order stochastic gradient descent algorithms. Many of these are
implemented in the `torch.optim` module. The typical way of optimizing a model
with these algorithms is by extracting the module's parameters (e.g. using
`parameters()`), and writing a manual optimization loop that calls `step()` on
a torch optimizer.

The problem of optimizing acquisition functions is different in that quite often
the dimensionality is much smaller. Indeed, optimizing over `q` design points in
a `d`-dimensional feature space results in `qd` scalar parameters to optimize
over. Both `q` and `d` are often quite small, and hence is the dimensionality of
the problem. Moreover, the optimization problem can be cast as a deterministic
one (either because an analytic acquisition function is used, or because the
reparameterization trick is employed to render the MC-based evaluation of the
acquisition function deterministic in terms of the input tensor `X`). As a
result, optimization algorithms that are typically inadmissible for problems
such as training Neural Networks become promising alternatives to standard SGD
methods. In particular, this includes quasi-second order methods (such as
L-BFGS or SLSQP) that approximate local curvature of the acquisition function by
using past gradient information. These methods are currently not well supported
in the `torch.optim` package, which is why botorch provides a custom interface
that wraps scipy's optimizers.
