#!/bin/bash

LATEST=false

while getopts 'l' flag; do
  case "${flag}" in
    l) LATEST=true
      ;;
    esac
  done

# Note: Everything is done in a virtualenv b/c otherwise pip has
# permission issues installing in the CircleCI docker container
python -m venv ../env
source ../env/bin/activate

# upgrade pip
pip install --upgrade pip

if [[ $LATEST == true ]]; then
  # install gpytorch master
  pip install git+https://github.com/cornellius-gp/gpytorch.git
  # install botorch + dev deps
  pip install -e .[dev]
  # This needs to happen at the end b/c GPyTorch overwrites the dev version of torch
  pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
fi

if [[ $LATEST == false ]]; then
  # Install CPU version to save download size. TODO: Update when PyTorch 1.1 is released
  pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
  pip install -e .[dev]
  # TODO: Remove when PyTorch 1.1 is released
  pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
fi
