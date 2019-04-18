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
of design points to be considered concurrently, and $d$ is the dimension of the
parameter space. Their output is a one-dimensional tensor with $b$ elements,
with the $i$-th element corresponding to the $i$-th t-batch. Always requiring a
explicit t-batch dimension makes it much easier and less ambiguous to work with
samples from the posterior in a consistent fashion.

*Note:* To simplify the user-facing API for non-batch evaluation, botorch
implements the `@batch_mode_transform` decorator. If applied to an acquisition
function, Tensor arguments to that function without a t-batch dimension (i.e. Tensors of shape $q \times d$) will be automatically converted to a t-batch of size 1, and the function's output is returned as a scalar Tensor.


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
$\textit{batch_shape} \times n \times d$ tensor (representing a t-batch-shape of $\textit{batch_shape}$, with $n$
$d$-dimensional data points in each batch) yields a posterior with `event_shape`
$\textit{batch_shape} \times n \times 1$.
- evaluating a multi-output model with $t$ outputs at a $\textit{batch_shape} \times n \times d$
tensor yields a posterior with `event_shape` $\textit{batch_shape} \times n \times t$.
- recall from the previous section that internally, all acquisition functions
are evaluated using a single t-batch dimension.

`sample_shape` is the shape (possibly multi-dimensional) of the samples drawn
*independently* from the distribution with `event_shape`, resulting in a tensor
of samples of shape `sample_shape` + `event_shape`. For instance,
- drawing a sample of shape $s1 \times s2$ from a posterior with `event_shape`
$b \times n \times t$ results in a tensor of shape
$s1 \times s2 \times \textit{batch_shape} \times n \times t$, where each of the $s1 s2$ tensors of
shape $\textit{batch_shape} \times n \times t$ are independent draws.



#### Batched Evaluation of Models and Acquisition Functions
The GPyTorch models provided in botorch support t-batched evaluation with arbitrary t-batch shapes.

##### Non-batched Models

In the simplest case, a model is fit to non-batched training points with shape $n \times d$. 
* *Non-batched evaluation* on a set of test points with shape $m \times d$ yields a joint posterior over the $m$ points.
* *Batched evaluation* on a set of test points with shape $\textit{batch_shape} \times m \times d$ yields $\textit{batch_shape}$ joint posteriors over the $m$ points in each respective batch.

##### Batched Models
The GPyTorch models can also be fit on batched training points with shape $\textit{input_batch_shape} \times n \times d$. Here, each batch is modeled independently (each batch has its own hyperparameters). For example the training points have shape $b_1 \times b_2 \times n \times d$ (two batch dimensions), the batched GPyTorch model is effectively $b_1 \times b_2$ independent models. More generally, suppose we fit a model to training points with shape $\textit{input_batch_shape} \times n \times d$. Then, the test points must support broadcasting to the $\textit{input_batch_shape}$. 

* *Non-batched evaluation* on a set of test points with shape $\textit{input_batch_shape}^* \times m \times d$, where each dimension of $\textit{input_batch_shape}^*$ either matches the corresponding dimension of $\textit{input_batch_shape}$ or is 1 to support broadcasting, yields $\textit{input_batch_shape}$ joint posteriors over the $m$ points (respectively if not broadcasting).

* *Batched evaluation* on a set of test points with shape $\textit{new_batch_shape} \times \textit{input_batch_shape}^* \times m \times d$, where $\textit{new_batch_shape}$ is the new batch shape for batched evaluation, yields $\textit{new_batch_shape} \times \textit{input_batch_shape}$ joint posteriors over the $m$ points in each respective batch (broadcasting as necessary over $\textit{input_batch_shape}$)

#### Batched Multi-Output Models
The [`BatchedMultiOutputGPyTorchModel`](../api/models.html#batchedmultioutputgpytorchmodel) class implements a fast multi-output model (assuming conditional independence of the outputs given the input) by batching over the outputs.

##### Training inputs/targets
Given training inputs with shape $\textit{input_batch_shape} \times n \times d$ and training outputs with shape $\textit{input_batch_shape} \times n \times o$, the `BatchedMultiOutputGPyTorchModel` permutes the training outputs to make the output $o$-dimension a batch dimension such that the augmented training inputs have shape $o \times \textit{input_batch_shape} \times n$. The training inputs (which are required to be the same for all outputs) are expanded to be $o \times \textit{input_batch_shape} \times n \times d$. 

##### Evaluation
When evaluating test points with shape $\textit{new_batch_shape} \times \textit{input_batch_shape} \times m \times d$ via the `posterior` method, the test points are broadcasted to the model(s) for each output. This results in the batched posterior where the mean has shape $\textit{new_batch_shape} \times o \times \textit{input_batch_shape} \times m$ which then is permuted back to the original multi-output shape $\textit{new_batch_shape} \times \textit{input_batch_shape} \times m \times o$.

#### Batched Optimization of Random Restarts
botorch uses random restarts to optimize an acquisition function from multiple starting points. To efficiently optimize an acquisition function for a $q$-batch of candidate points using $r$ random restarts, botorch uses used batched evaluation on a $r \times q \times d$ set of candidate points to independently evaluate and optimize each random restart in parallel. In order to optimize the $r$ acquisition functions using gradient information, the acquisition values the $r$ random restarts are summed before backpropping.

#### Batched Cross Validation
See the [Using batch evaluation for fast cross validation](../tutorials/batch_mode_cross_validation) tutorial for details on using batching for fast cross validation.
