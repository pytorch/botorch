---
id: models
title: Models
---

Models play an essential role in Bayesian Optimization (BO). A model is used as a
surrogate function for the actual underlying black box function to be optimized.
In BoTorch, a `Model` maps a set of design points to a posterior probability
distribution of its output(s) over the design points.

In BO, the model used is traditionally a Gaussian Process (GP),
in which case the posterior distribution, by definition is a multivariate
normal. However, with the exception of some of the analytic acquisition
functions in the
[`botorch.acquisition.analytic`](../api/acquisition.html#botorch-acquisition-analytic)
module, **BoTorch makes no assumption on the model being a GP**, or on the
posterior being a multivariate normal. The only requirement for using
BoTorch's Monte-Carlo based acquisition functions is that the model returns a
[`Posterior`](../api/posteriors.html#posterior) object that implements an
`rsample()` method for sampling from the posterior of the model. If you wish to
use gradient-based optimization algorithms, the model should allow
back-propagating gradients through the samples to the model input.


## The BoTorch Model Interface

BoTorch models are PyTorch modules that implement the light-weight
[`Model`](../api/models.html#model) interface. A BoTorch `Model` requires only
a single `posterior()` method that takes in a Tensor `X` of design points,
and returns a [`Posterior`](../api/posteriors.html#posterior) object describing
the (joint) probability distribution of the model output(s) over the design
points in `X`.

When working with GPs, [`GPyTorchModel`](../api/models.html#gpytorchmodel)
provides a base class for conveniently wrapping GPyTorch models.


## Terminology

Models may have multiple outputs, multiple inputs, and may exploit correlation
between different inputs. BoTorch uses the following terminology to
distinguish these model types:

* *Multi-Output Model*: a `Model` (as in the BoTorch object) with multiple
  outputs.
* *Multi-Task Model*: a `Model` making use of a logical grouping of
  inputs/observations (as in the underlying process). For example, there could
  be multiple tasks where each task has a different fidelity.

Note the following:
* A multi-task (MT) model may or may not be a multi-output model.
* Conversely, a multi-output (MO) model may or may not be a multi-task model.
* If a model is both, we refer to it as a multi-task-multi-output (MTMO) model.


## Standard BoTorch Models

BoTorch provides several GPyTorch models to cover most standard BO use cases:

### Single-Task GPs
These models use the same training data for all outputs and assume conditional
independence of the outputs given the input. If different training data is
required for each output, use a [`ModelListGP`](../api/models.html#modellistgp)
instead.
* [`SingleTaskGP`](../api/models.html#singletaskgp): a single-task
  exact GP that infers a homoskedastic noise level (no noise observations).
* [`FixedNoiseGP`](../api/models.html#fixednoisegp): a single-task exact GP that
uses fixed observation noise levels (requires noise observations).
* [`HeteroskedasticSingleTaskGP`](../api/models.html#heteropskedasticsingletaskgp):
  a single-task exact GP that models heteroskedastic noise using an additional
  internal GP model (requires noise observations).

### Model List of Single-Task GPs
* [`ModelListGP`](../api/models.html#modellistgp): A multi-output model in
  which outcomes are modeled independently, given a list of any type of
  single-task GP. This model should be used when the same training data is not
  used for all outputs.

### Multi-Task GPs
* [`MultiTaskGP`](../api/models.html#multitaskgp): a Hadamard multi-task,
  multi-output GP using an ICM kernel, inferring the noise level (does not
  require noise observations).
* [`FixedNoiseMultiTaskGP`](../api/models.html#fixednoisemultitaskgp):
  a Hadamard multi-task, multi-output GP using an ICM kernel, with fixed
  observation noise levels (requires noise observations).

All of the above models use Matérn 5/2 kernels with Automatic Relevance
Discovery (ARD), and have reasonable priors on hyperparameters that make them
work well in settings where the **input features are normalized to the unit
cube** and the **observations are standardized** (zero mean, unit variance).


## Implementing Custom Models

The configurability of the above models is limited (for instance, it is not
straightforward to use a different kernel). Doing so is an intentional design
decision -- we believe that having a few simple and easy-to-understand models for
basic use cases is more valuable than having a highly complex and configurable
model class whose implementation is difficult to understand.

Instead, we advocate that users implement their own models to cover
more specialized use cases. The light-weight nature of BoTorch's Model API makes
this easy to do. See the
[Using a custom BoTorch model in Ax](../tutorials/custom_botorch_model_in_ax)
tutorial for an example.

If you happen to implement a model that would be useful for other
researchers as well (and involves more than just swapping out the Matérn kernel
for an RBF kernel), please consider [contributing](getting_started#contributing)
this model to BoTorch.
