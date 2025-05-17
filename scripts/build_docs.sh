#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# run this script from the project root using `./scripts/build_docs.sh`

usage() {
  echo "Usage: $0 [-b] [-o]"
  echo ""
  echo "Build BoTorch documentation. Must be executed from root of BoTorch repository."
  echo ""
  echo "  -b   Build static version of documentation (otherwise start server)."
  echo "  -o   Only Docusaurus (skip Sphinx, tutorials). Useful when just make change to Docusaurus settings."
  echo ""
  exit 1
}

BUILD_STATIC=false
ONLY_DOCUSAURUS=false

while getopts 'bho' flag; do
  case "${flag}" in
    b)
      BUILD_STATIC=true
      ;;
    h)
      usage
      ;;
    o)
      ONLY_DOCUSAURUS=true
      ;;
    *)
      usage
      ;;
  esac
done


if [[ $ONLY_DOCUSAURUS == false ]]; then
  echo "-----------------------------------"
  echo "Generating tutorials"
  echo "-----------------------------------"
  python3 scripts/convert_ipynb_to_mdx.py --clean
fi

echo "-----------------------------------"
echo "Getting Docusaurus deps"
echo "-----------------------------------"
cd website || exit
yarn

if [[ $BUILD_STATIC == true ]]; then
  echo "-----------------------------------"
  echo "Building static site"
  echo "-----------------------------------"
  yarn build
else
  echo "-----------------------------------"
  echo "Starting local server"
  echo "-----------------------------------"
  yarn start
fi
