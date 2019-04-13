---
id: botorch_and_ax
title: Using botorch with Ax
---
[Ax](https://github.com/facebook/Ax) is a platform for optimizing experiments.
It relies on botorch for implementing Bayesian Optimization algorithms, but
provides much higher-level APIs that make it easy and convenient to specify
problems. It also comes with powerful metadata management, storage of results,
different APIs that support a variety of use cases. Ax makes it convenient to
use botorch in many standard Bayesian Optimization settings. Simply put, if
botorch is the "unframework", then Ax is the "framework".

Ax provides a `BotorchModel` that has sensible default for modeling and
optimization, but that can easily be customized by specifying and passing in
custom model constructors, acquisition functions, and optimization strategies.


## When to use botorch though Ax

Short answer: If it's simple to use botorch through Ax for your problem, then
use Ax. This should apply to most standard use cases.

For instance, say you want to tinker around with some parameters of your botorch
model (e.g. the kind of Kernel), but leave the rest of the the Bayesian
Optimization loop untouched. Then it's really easy to plug your custom botorch
model into ax. That way you can take advantage of Ax's various loop control APIs,
as well as it's powerful automated metadata management. See the
[Using a custom botorch model in Ax](../tutorials/custom_botorch_model_in_ax)
tutorial for how to do this.


### When not to use Ax

If you're working in a non-standard setting, where you might deal with very
high-dimensional feature or parameter spaces, or where the model fitting process
requires interactive work, then using Ax may end up becoming cumbersome. In such
a situation, you might be better off writing your own full Bayesian Optimization
loop outside of Ax (hey, it's python after all...). The
[q-Noisy Constrained EI](../tutorials/closed_loop_botorch_only) tutorial shows
how this can be done.
