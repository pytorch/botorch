#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

PYTORCH_NIGHLTY=false
DEPLOY=false

while getopts 'nd' flag; do
  case "${flag}" in
    n) PYTORCH_NIGHLTY=true ;;
    d) DEPLOY=true ;;
    *) echo "usage: $0 [-n] [-d]" >&2
       exit 1 ;;
    esac
  done

# NOTE: All of the below installs use sudo, b/c otherwise pip will get
# permission errors installing in the docker container. An alternative would be
# to use a virtualenv, but that would lead to bifurcation of the CircleCI config
# since we'd need to source the environemnt in each step.

# upgrade pip
sudo pip install --upgrade pip

# Install CPU version to save download size (don't let gpytorch install the full one)
if [[ $PYTORCH_NIGHLTY == true ]]; then
  sudo pip install --progress-bar off numpy
  sudo pip install --progress-bar off torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html
else
  sudo pip install --progress-bar off torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
fi

# install gpytorch master
sudo pip install --progress-bar off git+https://github.com/cornellius-gp/gpytorch.git

# install botorch + dev deps
sudo pip install -e .[dev]

if [[ $DEPLOY == true ]]; then
  sudo pip install --progress-bar off beautifulsoup4 ipython nbconvert
fi
