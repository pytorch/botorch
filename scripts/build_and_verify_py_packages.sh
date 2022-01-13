#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Get version number (created dynamically via setuptools-scm)
version=$(python setup.py --version)
cur_dir=$(pwd)

# set up temporary working directory
work_dir=$(mktemp -d)
cd "${work_dir}" || exit

# we can re-use this path
path="${cur_dir}/dist/botorch-${version}"

# install wheel and verify import works and version is correct
pip uninstall -y botorch
pip install "${path}-py3-none-any.whl"
wheel_version=$(python -c "import botorch; print(botorch.__version__)" | tail -n 1)
if [[ $wheel_version != "$version" ]]; then
  echo "Incorrect wheel version. Expected: ${version}, Actual: ${wheel_version}"
  exit 1
fi

# do the same for the source dist
pip uninstall -y botorch
pip install "${path}.tar.gz"
src_version=$(python -c "import botorch; print(botorch.__version__)" | tail -n 1)
if [[ $src_version != "$version" ]]; then
  echo "Incorrect source dist version. Expected: ${version}, Actual: ${src_version}"
  exit 1
fi
pip uninstall -y botorch


# clean up
cd "${cur_dir}" || exit
rm -rf "${work_dir}"
