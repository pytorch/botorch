#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

BOTORCH_VERSION="$(python ../setup.py --version)"
export BOTORCH_VERSION

conda build .
