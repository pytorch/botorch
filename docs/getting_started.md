---
id: getting_started
title: Getting Started
---

This section shows you how to get your feet wet with BoTorch.

Before jumping the gun, we recommend you start with the high-level
[Overview](overview) to learn about the basic concepts in BoTorch.


## Installing BoTorch

#### Installation Requirements:

BoTorch is easily installed via
[Anaconda](https://www.anaconda.com/distribution/#download-section) (strongly recommended for OSX)
or `pip`:

<!--DOCUSAURUS_CODE_TABS-->
<!--Conda-->
```bash
conda install botorch -c pytorch -c gpytorch -c conda-forge
```
<!--pip-->
```bash
pip install botorch
```
<!--END_DOCUSAURUS_CODE_TABS-->

For more installation options and detailed instructions, please see the
[Project Readme](https://github.com/pytorch/botorch/blob/main/README.md)
on GitHub.

## Basic Components

Here's a quick run down of the main components of a Bayesian Optimization loop.

1. Fit a Gaussian Process model to data
    ```python
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    train_X = torch.rand(10, 2)
    Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)  # explicit output dimension
    Y += 0.1 * torch.rand_like(Y)
    train_Y = (Y - Y.mean()) / Y.std()

    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll);
    ```

2. Construct an acquisition function
    ```python
    from botorch.acquisition import UpperConfidenceBound

    UCB = UpperConfidenceBound(gp, beta=0.1)
    ```

3. Optimize the acquisition function
    ```python
    from botorch.optim import optimize_acqf

    bounds = torch.stack([torch.zeros(2), torch.ones(2)])
    candidate, acq_value = optimize_acqf(
        UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    )
    ```


## Tutorials

Our Jupyter notebook tutorials help you get off the ground with BoTorch.
View and download them [here](../tutorials).


## API Reference

For an in-depth reference of the various BoTorch internals, see our
[API Reference](../api).


## Contributing

You'd like to contribute to BoTorch? Great! Please see
[here](https://github.com/pytorch/botorch/blob/main/CONTRIBUTING.md)
for how to help out.
