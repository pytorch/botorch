# botorch [Alpha]
[![Build Status](
  https://travis-ci.com/facebookexternal/botorch.svg?token=esFvpzSw7sLSsfe1PAr1&branch=master
)](https://travis-ci.com/facebookexternal/botorch)

botorch is a library for Bayesian Optimization in PyTorch.

It is currently an alpha version under active development - be warned!


## Installation

##### Setup Requirements (TODO: Remove once we can use torch Sobol)

The following are required to run the setup:

- Python >= 3.6
- numpy
- cython


##### Installation Requirements

- PyTorch >= 1.0.1
- gpytorch `eb96db228b9a1aeb8314ec1d8bc448d11d4cc46c` (**TODO:** pin beta to 0.3.0 release)
- scipy

**Important:**
You will want to have you PyTorch build link against **MKL** (the non-optimized
version of botorch can be up to an order of magnitude slower). Setting this up
manually can be tricky - to make sure this works please use the Anaconda
installation instructions on https://pytorch.org/.


### Install gpytorch using pip

botorch uses the latest gpytorch features. There is no current release that
includes these, so we pin gpytorch to a specific version. This will be unnecessary
for the beta release.

```bash
pip install git+https://github.com/cornellius-gp/gpytorch.git@eb96db228b9a1aeb8314ec1d8bc448d11d4cc46c
```
**Note:** The botorch 0.1a0 alpha release has been tested with the above commit
of gpytorch - if you already have a version of gpytorch installed, make sure
you uninstall that (using `pip uninstall gpytorch`) before running the above
command.


### Install botorch

To run the botorch setup, you'll need cython (**TODO:** Remove)
```bash
pip install cython
```

We recommend installing botorch using pip via ssh:
```bash
pip install git+ssh://git@github.com/facebookexternal/botorch.git
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



## Installation using conda

**TODO: conda install is unsupported until the repo is public**



## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.


## License
botorch is MIT licensed, as found in the LICENSE file.
