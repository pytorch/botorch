<a href="https://botorch.org">
  <img width="300" src="./botorch_logo_lockup.svg" alt="BoTorch Logo" />
</a>

[![Build Status](
  https://travis-ci.com/pytorch/botorch.svg?token=esFvpzSw7sLSsfe1PAr1&branch=master
)](https://travis-ci.com/pytorch/botorch)


BoTorch is a library for Bayesian Optimization built on PyTorch.

*BoTorch is currently in alpha and under active development - be warned*!


### Why BoTorch
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


### Target Audience

The primary audience for hands-on use of BoTorch are researchers and
sophisticated practitioners in Bayesian Optimization and AI.

We recommend using BoTorch as a low-level API for implementing new algorithms
for [Ax](https://github.com/facebook/Ax). Ax has been designed to be
an easy-to-use platform for end-users, which at the same time is flexible enough
for Bayesian Optimization researchers to plug into for handling of feature
transformations, (meta-)data management, storage, etc.

We recommend that end-users who are not actively doing research on Bayesian
Optimization simply use Ax.


## Installation

#### Installation Requirements

- Python >= 3.6
- PyTorch nightly (**TODO:** peg to PyTorch 1.1 once released)
- gpytorch >= 0.3.1 (**TODO:** peg to GPyTorch 0.3.2 once released)
- scipy

**Important note for MacOS users:**
* You will want to make sure your PyTorch build is linked against MKL (the
  non-optimized version of BoTorch can be up to an order of magnitude slower in
  some settings). Setting this up manually on MacOS can be tricky - to ensure
  this works properly please follow the
  [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
* If you need CUDA on MacOS, you will need to build PyTorch from source. Please
  consult the PyTorch installation instructions above.


#### Installing BoTorch

The latest release of BoTorch is easily installed using either pip or conda:
```bash
pip install botorch
```

**TODO: Conda install**


If you'd like to try our bleeding edge features (and don't mind running into an
occasional bug here or there), you can install the latest master from GitHub
(this will also require installing the current GPyTorch master)::
```bash
pip install git+https://github.com/cornellius-gp/gpytorch.git
pip install git+https://github.com/pytorch/botorch.git
```


#### Installing BoTorch from the private repo **TODO: REMOVE**

BoTorch is easily installed using pip:
```bash
pip install git+ssh://git@github.com/pytorch/botorch.git
```

*Note:* You must use **ssh** here since the repo is private - for this to work,
make sure your ssh public key is registered with GitHub, and is usable by ssh.

Alternatively, you can do a manual install. To do a basic install, run:
```bash
cd botorch
pip install -e .
```

To customize the installation, you can also run the following instead:
* `pip install -e .[dev]`: Also installs all tools necessary for development
  (testing, linting, docs building).
* `pip install -e .[tutorial]`: Also installs jupyter for running the tutorial
  notebooks.



## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.


## License
BoTorch is MIT licensed, as found in the LICENSE file.
