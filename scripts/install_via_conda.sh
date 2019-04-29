#!/bin/bash

LATEST=false

while getopts 'l' flag; do
  case "${flag}" in
    l) LATEST=true ;;
    *) echo "usage: $0 [-l]" >&2
       exit 1 ;;
    esac
  done

# update conda
conda update -y -n base -c defaults conda

# required to use conda develop
conda install -y conda-build

if [[ $LATEST == true ]]; then
  # install CPU version for much smaller download
  conda install -y -c pytorch pytorch-nightly-cpu
  # get gpytorch master
  git clone https://github.com/cornellius-gp/gpytorch.git ../gpytorch
  # install gpytorch
  conda develop ../gpytorch
fi

if [[ $LATEST == false ]]; then
  # install CPU version for much smaller download
  conda install -y -c pytorch pytorch-cpu
  conda install -y -c gpytorch gpytorch
fi

# install other deps
conda install -y scipy sphinx pytest flake8
conda install -y -c conda-forge black pytest-cov sphinx-autodoc-typehints

# install botorch
conda develop .
