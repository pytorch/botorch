# botorch [Alpha]
[![Build Status](
  https://travis-ci.com/facebookexternal/botorch.svg?token=esFvpzSw7sLSsfe1PAr1&branch=master
)](https://travis-ci.com/facebookexternal/botorch)

botorch is a library for Bayesian Optimization in PyTorch.

It is currently an alpha version under active development - be warned!


#### Setup Requirements

The following are required to run the setup:

- Python >= 3.6
- numpy
- cython

**TODO:** Remove Setup Requirements altogether once we can use torch Sobol


#### Dependencies

- PyTorch >= 1.0.1 [^pytorch_build]  (**TODO:**: Update to 1.1 when release)
- gpytorch latest (**TODO:** lock beta to 0.3.0 release)
- scipy

[^pytorch_build]: You will want to have PyTorch link against MKL. This can be
  kind of finicky, to make sure this works use the conda install as described in
  the quick start instructions on https://pytorch.org/). If you set up a clean
  conda environment as described below you won't have to worry about this.
  To use CUDA on Mac OS, you need to build PyTorch from source.


### Installing from private repo

The botorch repo is currently private, so you'll need to do a little more work.
This will simplify once botorch is fully open-sourced.

#### Set up a clean conda environment (optional)
* Download the `botorch_base.yml` file
* Create the base environment using `conda env create -f botorch_base.yml`
* Activate the environment using `conda activate botorch_base`


#### Install the gpytorch latest using pip
```bash
pip install git+https://github.com/cornellius-gp/gpytorch.git
```

#### Install botorch using pip via ssh (recommended):
```bash
pip install git+ssh://git@github.com/facebookexternal/botorch.git
```

*Note:* You **must** use ssh here since the repo is private - for this to work,
make sure your ssh public key is registered with GitHub, and is usable by ssh.

#### Manual install

* Download botorch from the [Git repository](https://github.com/facebookexternal/botorch).
* Install all [build dependencies](#setup-requirements)
* Run the following to get the default installation:
```bash
cd botorch
pip install -e .
```

Alternatively, you can also do the following:
* `pip install -e .[dev]`: Also installs all tools necessary for development
  (testing, linting, docs building).
* `pip install -e .[tutorial]`: Also installs jupyter for running the tutorial
  notebooks


### Install using conda

**TODO: conda install is unsupported until the repo is public**



## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
botorch is MIT licensed, as found in the LICENSE file.
