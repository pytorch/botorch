#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# we cannot use relative paths here, since setuptools_scm options in
# pyproject.toml cannot dynamically determine the root dir
cd .. || exit
BOTORCH_VERSION="$(python -m setuptools_scm)"
export BOTORCH_VERSION
cd .conda || exit

conda build .
