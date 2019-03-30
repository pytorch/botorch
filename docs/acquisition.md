---
id: acquisition
title: Acquisition Functions
---


#### Monte Carlo (MC) evaluation of q-batch acquisition functions

Because analytic expressions do not exist for `q > 1`, `botorch` relies on the
re-parameterization trick and (quasi)-MC sampling for optimization and
estimation of the batch acquisition functions.

- The re-parameterization trick (see e.g. [^KingmaWelling2014], [^Rezende2014])
can be used to write the posterior distribution as a deterministic
transformation of an auxiliary random variable $\epsilon$. For example, a
normally distributed random variable $X$ with mean $\mu$ and standard deviation
$\theta$ has the same distribution as $\mu + \sigma \epsilon$ where $\epsilon$
is a standard normal. Therefore, an expectation with respect to $X$ can be
approximated using samples from $\epsilon$. In the case where $\mu$ and $\sigma$
are parameters of an optimization problem, MC approximations of the objective at
different values of $\mu$ and $\sigma$ can be computed using a single set of
"base samples."

- What are the trade-offs between using a fixed set of base samples versus
re-sampling on every MC evaluation of the acquisition function? If the base
samples are fixed, the problem of optimizing the acquisition function becomes
deterministic, allowing for conventional quasi-second order methods to be used
(e.g., `L-BFGS` and Sequential Least-Squares Programming). These have faster
convergence rates than first-order methods and can speed up acquisition function
optimization significantly. Although the approximated acquisition function is
biased in this case, our anecdotal observations suggest that the location of
optimizer itself does not change much. On the other hand, if re-sampling is used,
the optimization objective becomes stochastic (though unbiased) and a stochastic
optimizer should be used.

- Base samples are constructed using an `MCSampler` object, which provides an
interface that allows for different sampling techniques. `IIDNormalSampler`
utilizes independent standard normal draws, while `SobolQMCNormalSampler` uses
quasi-random, low-discrepancy "Sobol" sequences as uniform samples. These
uniform samples are then transformed to construct normal samples. Sobol
sequences are more evenly distributed than i.i.d. uniform samples and tend to
improve the convergence rate of MC estimates of integrals/expectations.
botorch makes it easy to implement and use custom sampling techniques.

[^KingmaWelling2014]: D. P. Kingma, M. Welling.
*Auto-Encoding Variational Bayes.* ICLR, 2013.

[^Rezende2014]: D. J. Rezende, S. Mohamed, D. Wierstra.
*Stochastic Backpropagation and Approximate Inference in Deep Generative Models.*
ICML, 2014.


#### Analytic formulations

`botorch` also provides implementation of analytic acquisition functions that
do not depend on MC sampling. These acquisition functions are subclasses of
`AnalyticAcquisitionFunction` and only exist for the case of `q = 1`. These
include classical acquisition functions such as Expected Improvement (EI),
Upper Confidence Bound (UCB), and Probability of Improvement (PI). An example
comparing the analytic version of EI `ExpectedImprovement` to the MC version
`qExpectedImprovement` can be found in [LINK TO ACQUISITION TUTORIAL].
