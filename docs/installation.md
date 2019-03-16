---
id: installation
title: Installation
---

### Installation Requirements

- Python >= 3.6
- PyTorch >= 1.0.1
- gpytorch >= 0.2.1
- cython
- scipy


#### Installing into a clean conda environment

1. Create the base environment using `conda env create -f botorch_base.yml`

2. Activate the environment using `conda activate botorch_base`

3. Install via pip: `pip install git+ssh://git@github.com/facebookexternal/botorch.git`

*Note:* In 3. you **must** use ssh since the repo is private - for that to work, make
sure your ssh public key is registered with GitHub, and is usable by ssh.


#### Manual install

After installing the dependencies,
* Download botorch from the [Git repository](https://github.com/facebookexternal/botorch).
* `cd` into the `botorch` project and run:

```bash
pip3 install -e .
```


### Notes

1. To use **CUDA on MacOS**, pytorch needs to be built from source instead
  (see the quick start instructions on https://pytorch.org/)
