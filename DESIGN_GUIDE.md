# botorch design guide [WIP]

This document provides guidance on botorch's internal design.

## design philosophy

botorch follows a modular, low-overhead “unframework” design philosophy, which
provides building blocks and abstractions, but does not force the user to do
things in any one particular way. This empowers the user to easily prototype and
test new approaches.


## acquisition functions

botorch supports batch acquisition functions (e.g. q-EI, q-UCB, etc.) that
assign a joint utility to a set of q design points in the parameter space.

Unfortunately, this batch nomenclature gets easily conflated with the pytorch
notion of batch-evaluation. To avoid confusion in that respect, we adopt the
convention of referring to batches in the batch-acquisition sense as "q-batches",
and to batches in the torch batch-evaluation sense as "t-batches".

Internally, q-batch acquisition functions operate on input tensors of shape
`b x q x d`, where `b` is the number of t-batches, `q` is the number of design
points, and `d` is the dimension of the parameter space. Their output is a
one-dimensional tensor with `b` elements, with the `i`-th element corresponding
to the `i`-th t-batch.

To simplify the user-facing API, if provided with an input tensor of shape `q x d`,
a t-batch size of 1 is inferred, and the result is returned as a torch scalar.
