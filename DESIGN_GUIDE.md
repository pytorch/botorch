# botorch design guide [WIP]

This document provides guidance on botorch's internal design.

## Design philosophy

botorch's goal is to bridge the gap between being a very flexible research
platform and at the same time providing a reliable, production-grade
implementation.

In the spirit of PyTorch, botorch aspires to follow a modular, low-overhead
“unframework” design philosophy, which provides building blocks and abstractions, but does not force users to do things in any one particular way.
The goal is to empower the user to easily prototype and test new approaches,
while being able to rely on well-tested building blocks.


### Models and Posteriors [WIP]

At the core of botorch is an abstract, model-agnostic API that defines minimal
`Model` and `Posterior` classes.
- A `Model` is only required to provide a `posterior` method that, when
evaluated at some feature tensor `X`, returns a `Posterior` object that describes the posterior distribution of the modeled output at the points in `X`.
- `Posterior` objects must provide an `rsample` method that allows to draw
samples from the distribution they describe. The produced samples are then
used for Monte-Carlo evaluation of botorch's acquisition functions.

This evaluation path looks as follows:
```
X -> Model -> Posterior -> samples -> AcquisitionFunction -> output
```

In order to effectively optimize the acquisition functions, botorch generally
assumes that gradients of `output` w.r.t. `X` can be computed using PyTorch's
auto-differentiation. In particular, `Model` and `Posterior` should be
implemented in such a way that they fully support back-propagating gradients.

*Note: When using derivative-free optimization, the above backprop requirement
is unnecessary. The user is welcome to explore this avenue - for now, however,
botorch focuses on gradient-based optimization techniques.*


### Acquisition functions and batch mode

botorch supports batch acquisition functions (e.g. q-EI, q-UCB, etc.) that
assign a joint utility to a set of q design points in the parameter space.

Unfortunately, this batch nomenclature gets easily conflated with the PyTorch
notion of batch-evaluation. To avoid confusion in that respect, we adopt the
convention of referring to batches in the batch-acquisition sense as "q-batches",
and to batches in the torch batch-evaluation sense as "t-batches".

Internally, q-batch acquisition functions operate on input tensors of shape
`b x q x d`, where `b` is the number of t-batches, `q` is the number of design
points, and `d` is the dimension of the parameter space. Their output is a
one-dimensional tensor with `b` elements, with the `i`-th element corresponding
to the `i`-th t-batch. Always requiring an explicit batch dimension makes it
much easier and less ambiguous to work with samples from the posterior in a
consistent fashion (see below).

*Note:* To simplify the user-facing API for non-batch evaluation, botorch
implements the `@batch_mode_transform` decorator. If applied to an acquisition function, non-batch Tensor arguments (i.e. of shape `q x d`) to that function
will be automatically converted to a t-batch of size 1, and the function's output
is returned as a scalar Tensor.


### Sample shapes

Internally, botorch evaluates acquisition functions using (quasi)-MC sampling.
Because of the liberal use of batching operations, the shape of the involved
tensors can be somewhat confusing. This section is meant to clarify this.
We use the PyTorch notions of `sample_shape` and `event_shape`:

`event_shape` is the shape of a single sample drawn from the underlying
distribution. For instance,
- evaluating a single-output model at a `1 x n x d` tensor,
representing `n` data points in `d` dimensions each,
yields a posterior with `event_shape` of `1 x n x 1`.
Evaluating the same model at a `b x n x d` tensor (representing a
t-batch of size `b`, with `n` `d`-dimensional data points in each
batch) yields a posterior with `event_shape` of `b x n x 1`.
- evaluating a multi-output model with `t` outputs
at a `b x n x d` tensor yields a posterior with `event_shape` `b x n x t`.
- recall from the previous section that internally, all acquisition functions
are evaluated in batch mode.

`sample_shape` is the shape (possibly multi-dimensional) of the samples drawn
*independently* from the distribution with `event_shape`, resulting in a tensor
of samples of shape `sample_shape` + `event_shape`. For instance,
- drawing a sample of shape `s1 x s2` from a posterior with `event_shape` of
`b x n x t` results in a tensor of shape `s1 x s2 x b x n x t`, where each of
the `s1 * s2` tensors of shape `b x n x t` are independent draws.
