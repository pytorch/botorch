<a href="https://botorch.org">
  <img width="350" src="./botorch_logo_lockup.svg" alt="BoTorch Logo" />
</a>

<hr/>

[![Conda](https://img.shields.io/conda/v/pytorch/botorch.svg)](https://anaconda.org/pytorch/botorch)
[![PyPI](https://img.shields.io/pypi/v/botorch.svg)](https://pypi.org/project/botorch)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CircleCI](https://circleci.com/gh/pytorch/botorch.svg?style=shield)](https://circleci.com/gh/pytorch/botorch)
[![Codecov](https://img.shields.io/codecov/c/github/pytorch/botorch.svg)](https://codecov.io/github/pytorch/botorch)

BoTorch is a library for Bayesian Optimization built on PyTorch.

*BoTorch is currently in beta and under active development!*


#### Why BoTorch ?
BoTorch
* Provides a modular and easily extensible interface for composing Bayesian
  optimization primitives, including probabilistic models, acquisition functions,
  and optimizers.
* Harnesses the power of PyTorch, including auto-differentiation, native support
  for highly parallelized modern hardware (e.g. GPUs) using device-agnostic code,
  and a dynamic computation graph.
* Supports Monte Carlo-based acquisition functions via the
  [reparameterization trick](https://arxiv.org/abs/1312.6114), which makes it
  straightforward to implement new ideas without having to impose restrictive
  assumptions about the underlying model.
* Enables seamless integration with deep and/or convolutional architectures in PyTorch.
* Has first-class support for state-of-the art probabilistic models in
  [GPyTorch](http://www.gpytorch.ai/), including support for multi-task Gaussian
  Processes (GPs) deep kernel learning, deep GPs, and approximate inference.


#### Target Audience

The primary audience for hands-on use of BoTorch are researchers and
sophisticated practitioners in Bayesian Optimization and AI.
We recommend using BoTorch as a low-level API for implementing new algorithms
for [Ax](https://ax.dev). Ax has been designed to be an easy-to-use platform
for end-users, which at the same time is flexible enough for Bayesian
Optimization researchers to plug into for handling of feature transformations,
(meta-)data management, storage, etc.
We recommend that end-users who are not actively doing research on Bayesian
Optimization simply use Ax.


## Installation

**Installation Requirements**
- Python >= 3.7
- PyTorch >= 1.4
- gpytorch >= 1.0
- scipy


##### Installing the latest release

The latest release of BoTorch is easily installed either via
[Anaconda](https://www.anaconda.com/distribution/#download-section) (recommended):
```bash
conda install botorch -c pytorch -c gpytorch
```
or via `pip`:
```bash
pip install botorch
```

You can customize your PyTorch installation (i.e. CUDA version, CPU only option)
by following the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

***Important note for MacOS users:***
* Make sure your PyTorch build is linked against MKL (the non-optimized version
  of BoTorch can be up to an order of magnitude slower in some settings).
  Setting this up manually on MacOS can be tricky - to ensure this works properly,
  please follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
* If you need CUDA on MacOS, you will need to build PyTorch from source. Please
  consult the PyTorch installation instructions above.


##### Installing from latest master

If you would like to try our bleeding edge features (and don't mind potentially
running into the occasional bug here or there), you can install the latest
master directly from GitHub (this will also require installing the current GPyTorch master):
```bash
pip install --upgrade git+https://github.com/cornellius-gp/gpytorch.git
pip install --upgrade git+https://github.com/pytorch/botorch.git
```

**Manual / Dev install**

Alternatively, you can do a manual install. For a basic install, run:
```bash
git clone https://github.com/pytorch/botorch.git
cd botorch
pip install -e .
```

To customize the installation, you can also run the following variants of the
above:
* `pip install -e .[dev]`: Also installs all tools necessary for development
  (testing, linting, docs building; see [Contributing](#contributing) below).
* `pip install -e .[tutorials]`: Also installs all packages necessary for running the tutorial notebooks.


## Getting Started

Here's a quick run down of the main components of a Bayesian optimization loop.
For more details see our [Documentation](https://botorch.org/docs/introduction) and the
[Tutorials](https://botorch.org/tutorials).

1. Fit a Gaussian Process model to data
  ```python
  import torch
  from botorch.models import SingleTaskGP
  from botorch.fit import fit_gpytorch_model
  from gpytorch.mlls import ExactMarginalLogLikelihood

  train_X = torch.rand(10, 2)
  Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)  # explicit output dimension
  Y += 0.1 * torch.rand_like(Y)
  train_Y = (Y - Y.mean()) / Y.std()

  gp = SingleTaskGP(train_X, train_Y)
  mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
  fit_gpytorch_model(mll)
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


## Citing BoTorch

If you use BoTorch, please cite the following paper:
> [M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: Programmable Bayesian Optimization in PyTorch. ArXiv e-prints, 2019.](http://arxiv.org/abs/1910.06403)

```
@article{balandat2019botorch,
  Author = {Balandat, Maximilian and Karrer, Brian and Jiang, Daniel R. and Daulton, Samuel and Letham, Benjamin and Wilson, Andrew Gordon and Bakshy, Eytan},
  Journal = {arxiv e-prints},
  Title = {{BoTorch: Programmable Bayesian Optimization in PyTorch}},
  Year = {2019},
  url = {http://arxiv.org/abs/1910.06403}
}
```


## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.


## License
BoTorch is MIT licensed, as found in the [LICENSE](LICENSE) file.
