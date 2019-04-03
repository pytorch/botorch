---
id: installation
title: Installation
---

#### Setup Requirements

The following are required to run the setup:

- Python >= 3.6
- numpy
- cython

**TODO:** Remove Setup Requirements once we use torch Sobol


#### Dependencies

- PyTorch >= 1.0.1 [^pytorch_build]  (**TODO**: Update to 1.1)
- gpytorch >= 0.2.1
- scipy

[^pytorch_build]: You will want to have PyTorch link against MKL. This can be
  kind of finicky, to make sure this works use the conda install as described in
  the quick start instructions on https://pytorch.org/). To use CUDA on Mac OS,
  you need to build PyTorch from source (also see quickstart instructions).


### Installing from private repo

The botorch repo is currently private, so you'll need to do a little more work.
This will simplify once botorch is fully open-sourced.

#### Set up a clean conda environment (optional)
* Download the `botorch_base.yml` file
* Create the base environment using `conda env create -f botorch_base.yml`
* Activate the environment using `conda activate botorch_base`

#### Install using pip via ssh (recommended):
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


### Install using pip

To install the latest official release, run
```bash
pip install botorch
```

To install the current master (unstable), run
```bash
pip install git+https://github.com/facebookexternal/botorch.git
```


### Install using conda (TODO)

To install the latest official release, run
```bash
conda install botorch
```
