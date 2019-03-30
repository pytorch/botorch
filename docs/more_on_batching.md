---
id: more_on_batching
title: More on Batching
---


#### Batch Acquisition Functions

botorch supports batch acquisition functions that assign a joint utility to a
set of $q$ design points in the parameter space. These are, for obvious reasons,
referred to as q-Acquisition Functions. For instance, botorch ships with support
for q-EI, q-UCB, and a few others.

As discussed in the
[design philosophy](design_philosophy#batching-batching-batching),
botorch has adopted the convention of referring to batches in the
batch-acquisition sense as "q-batches", and to batches in the torch
batch-evaluation sense as "t-batches".

Internally, q-batch acquisition functions operate on input tensors of shape
$b \times q \times d$, where $b$ is the number of t-batches, $q$ is the number
of design points to be considered concurrently, and $d$ is the dimension of the parameter space.
Their output is a one-dimensional tensor with $b$ elements, with the $i$-th
element corresponding to the $i$-th t-batch. Always requiring an explicit batch
dimension makes it much easier and less ambiguous to work with samples from the
posterior in a consistent fashion.

*Note:* To simplify the user-facing API for non-batch evaluation, botorch
implements the `@batch_mode_transform` decorator. If applied to an acquisition function, non-batch Tensor arguments (i.e. of shape $q \times d$) to that function
will be automatically converted to a t-batch of size 1, and the function's output
is returned as a scalar Tensor.


#### Batching Sample Shapes

As their name would suggest, botorch evaluates Monte-Carlo acquisition functions
using (quasi-) Monte-Carlo sampling from the posterior at the input features
$X$. Hence, on top of the existing q-batch and t-batch dimensions, we also end
up with another batch dimension corresponding to the MC samples we draw.
We use the PyTorch notions of `sample_shape` and `event_shape`.

`event_shape` is the shape of a single sample drawn from the underlying
distribution. For instance,
- evaluating a single-output model at a $1 \times n \times d$ tensor,
representing $n$ data points in $d$ dimensions each, yields a posterior with
`event_shape` $1 \times n \times 1$. Evaluating the same model at a
$b \times n \times d$ tensor (representing a t-batch of size $b$, with $n$
$d$-dimensional data points in each batch) yields a posterior with `event_shape`
$b \times n \times 1$.
- evaluating a multi-output model with $t$ outputs at a $b \times n \times d$
tensor yields a posterior with `event_shape` $b \times n \times t$.
- recall from the previous section that internally, all acquisition functions
are evaluated in batch mode.

`sample_shape` is the shape (possibly multi-dimensional) of the samples drawn
*independently* from the distribution with `event_shape`, resulting in a tensor
of samples of shape `sample_shape` + `event_shape`. For instance,
- drawing a sample of shape $s1 \times s2$ from a posterior with `event_shape`
$b \times n \times t$ results in a tensor of shape
$s1 \times s2 \times b \times n \times t$, where each of the $s1 s2$ tensors of
shape $b \times n \times t$ are independent draws.



#### Batched Evaluation of Models and Acquisition Functions
**TODO**

#### Batched Optimization of Random Restarts
**TODO**

#### Batched Models
**TODO** Reference CV tutorial
