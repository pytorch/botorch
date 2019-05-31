#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

PYTORCH_NIGHTLY=false

while getopts 'n' flag; do
  case "${flag}" in
    n) PYTORCH_NIGHTLY=true ;;
    *) echo "usage: $0 [-n]" >&2
       exit 1 ;;
    esac
  done

# update conda
conda update -y -n base -c defaults conda

# required to use conda develop
conda install -y conda-build

if [[ $PYTORCH_NIGHTLY == true ]]; then
  # install CPU version for much smaller download
  conda install -y -c pytorch pytorch-nightly-cpu
else
  # install CPU version for much smaller download
  conda install -y -c pytorch pytorch-cpu
fi

# get gpytorch master
git clone https://github.com/cornellius-gp/gpytorch.git ../gpytorch

# install gpytorch
conda develop ../gpytorch

# install other deps
conda install -y scipy sphinx pytest flake8
conda install -y -c conda-forge black pytest-cov sphinx-autodoc-typehints

# install botorch
conda develop .
