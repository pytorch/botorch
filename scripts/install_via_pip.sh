#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

LATEST=false
DEPLOY=false

while getopts 'ld' flag; do
  case "${flag}" in
    l) LATEST=true ;;
    d) DEPLOY=true ;;
    *) echo "usage: $0 [-l] [-d]" >&2
       exit 1 ;;
    esac
  done

# NOTE: All of the below installs use sudo, b/c otherwise pip will get
# permission errors installing in the docker container. An alternative would be
# to use a virtualenv, but that would lead to bifurcation of the CircleCI config
# since we'd need to source the environemnt in each step.

# upgrade pip
sudo pip install --upgrade pip

if [[ $LATEST == true ]]; then
  # install gpytorch master
  sudo pip install git+https://github.com/cornellius-gp/gpytorch.git
  # install botorch + dev deps
  sudo pip install -e .[dev]
  # This needs to happen at the end b/c GPyTorch overwrites the dev version of torch
  sudo pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
else
  # Install CPU version to save download size. TODO: Update when PyTorch 1.1 is released
  sudo pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
  sudo pip install -e .[dev]
  # TODO: Remove when PyTorch 1.1 is released
  sudo pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
fi

if [[ $DEPLOY == true ]]; then
  sudo pip install beautifulsoup4 ipython nbconvert
fi
