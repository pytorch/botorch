#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# update conda
conda update -y -n base -c defaults conda

# install pip into conda
conda install -y pip

# install python packaging deps
pip install --upgrade setuptools build wheel twine

# install conda-build
conda install -y conda-build

# test python packaging
python -m build --sdist --wheel

# test conda packaging
conda config --show
conda config --add channels pytorch
conda config --add channels gpytorch
conda config --add channels conda-forge
cd .conda || exit
./build_conda.sh
