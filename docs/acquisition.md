---
id: acquisition
title: Acquisition Functions
---

## Monte Carlo (MC) evaluation of q-batch acquisition functions

Many common acquisition functions can be expressed as the expectation of some
real-valued function of the model output(s) at the design point(s):

$$ H(X) = \mathbb{E}\bigl[ h(Y) \mid Y \sim \mathbb{P}_Y(X) \bigr] $$

Here $\mathbb{P}_Y(X)$ is the posterior distribution of $Y$ given $X$.

Evaluating the acquisition function thus requires evaluating an integral over
the posterior distribution. In most cases, this is analytically intractable (in
particular, analytic expressions do generally not exist for `q > 1`).

An alternative is to use Monte-Carlo (MC) sampling to approximate the integrals.
A MC approximation of $H$ at $X$ using $N$ MC samples is

$$ H(X) \approx \frac{1}{N} \sum_{j=1}^N h(y_j) $$

where $y_j \sim \mathbb{P}_Y(X)$.


botorch relies on the re-parameterization trick ([^KingmaWelling2014], [^Rezende2014])
and MC sampling for optimization and estimation of the batch acquisition functions [^Wilson2017].

As discussed in the [overview](./overview), a single set of base samples can be
used for optimization when the re-parameterization trick is employed. What are
the trade-offs between using a fixed set of base samples versus re-sampling on
every MC evaluation of the acquisition function? If the base samples are fixed,
the problem of optimizing the acquisition function becomes deterministic, allowing
for conventional quasi-second order methods to be used (e.g., `L-BFGS` and Sequential
Least-Squares Programming). These have faster convergence rates than first-order
methods and can speed up acquisition function optimization significantly.

Although the approximated acquisition function is biased in this case (conditonal on
the samples), in most cases the location of optimizer itself is somewhat robust to this
bias.
On the other hand, if re-sampling is used, the optimization objective becomes
stochastic (though unbiased) and a stochastic optimizer should be used.

[^KingmaWelling2014]: D. P. Kingma, M. Welling. Auto-Encoding Variational Bayes.
ICLR, 2013.

[^Rezende2014]: D. J. Rezende, S. Mohamed, D. Wierstra. Stochastic
Backpropagation and Approximate Inference in Deep Generative Models. ICML, 2014.

[^Wilson2017]: J. T. Wilson, R. Moriconi, F. Hutter, M. P. Deisenroth. The Reparameterization Trick for Acquisition Functions. NeurIPS Workshop on Bayesian Optimization, 2017.

## Analytic Acquisition Functions

`botorch` also provides implementation of analytic acquisition functions that
do not depend on MC sampling. These acquisition functions are subclasses of
`AnalyticAcquisitionFunction` and only exist for the case of a single candidate point (`q = 1`). These
include classical acquisition functions such as Expected Improvement (EI),
Upper Confidence Bound (UCB), and Probability of Improvement (PI). An example
comparing the analytic version of EI `ExpectedImprovement` to the MC version
`qExpectedImprovement` can be found in
[this tutorial](../tutorials/compare_mc_analytic_acquisition).

Analytic acquisition functions allow for an explicit expression in terms of the
summary statistics of the posterior distribution at the evaluated point(s).
A classic such acquisition function is Expected Improvement of a single point
for a Gaussian posterior, given by

$$ \text{EI}(x) = \mathbb{E}\bigl[
\max(y - f_{max}, 0) \mid y\sim \mathcal{N}(\mu(x), \sigma^2(x))
\bigr] $$

where $\mu(x)$ and $\sigma(x)$ are the posterior mean and variance of $f$ at the
point $x$, and $f_{max}$ is the best function value observed so far (assuming
noiseless observations). It can be shown that

$$ \text{EI}(x) = \sigma(x) \bigl( z \Phi(z) + \varphi(z) \bigr)$$

where $z = \frac{\mu(x) - f_{\max}}{\sigma(x)}$ and $\Phi$ and $\varphi$ are
the cdf and pdf of the standard Normal distribution, respectively.

With some additional work, it is also possible to express the gradient of
the Expected Improvement with respect to the design $x$. Classic Bayesian
Optimization software will implement this gradient function explicitly, so that
it can be used for numerically optimizing the acquisition function.

botorch, in contrast, harnesses PyTorch's automatic differentiation feature
("autograd") in order to obtain gradients of acquisition functions. This makes
implementing new acquisition functions much less cumbersome, as it does not
require to analytically derive gradients. All that is required is that the
operations performed in the acquisition function computation allow for the
back-propagation of gradient information through the posterior and the model.
