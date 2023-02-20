#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

echo "cloning"
git clone https://github.com/pytorch/botorch.git botorch-main
cd botorch-main
echo "creating file"
touch test_file0.csv

echo "Checking out branch artifacts"
git fetch origin artifacts &&
    git checkout artifacts &&
    git add test_file0.csv &&
    echo "Committing and pushing" && 
    git commit test_file0.csv -m "Adding most recent tutorials output" &&
    echo "Pushing" &&
    git push origin artifacts

echo "Cleaning up"
cd ..
rm -rf botorch-main 