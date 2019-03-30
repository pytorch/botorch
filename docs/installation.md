---
id: installation
title: Installation
---

### Installation Requirements

- Python >= 3.6
- PyTorch >= 1.0.1  (**TODO**: Update to 1.1)
- gpytorch >= 0.2.1
- cython  (**TODO**: Remove once using torch Sobol)
- scipy


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
* Install all [dependencies](#installation-requirements)
* Run the following:
```bash
cd botorch
pip install -e .
```


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


### Notes

1. To use **CUDA on MacOS**, pytorch needs to be built from source instead
  (see the quick start instructions on https://pytorch.org/)
