#!/bin/bash

usage() {
  echo "Usage: $0 [-b]"
  echo ""
  echo "Build Ax documentation."
  echo ""
  echo "  -b   Build static version of documentation (otherwise start server)"
  echo ""
  exit 1
}

BUILD_STATIC=false

while getopts 'hb' flag; do
  case "${flag}" in
    h)
      usage
      ;;
    b)
      BUILD_STATIC=true
      ;;
    *)
      usage
      ;;
  esac
done

echo "-----------------------------------"
echo "Generating API reference via Sphinx"
echo "-----------------------------------"
cd sphinx || exit
make html
cd ..

# build Docusaurus site
echo "-----------------------------------"
echo "Building Docusaurus site"
echo "-----------------------------------"
cd website || exit
yarn

# run script to parse html generated by sphinx
echo "--------------------------------------------"
echo "Parsing Sphinx docs and moving to Docusaurus"
echo "--------------------------------------------"
cd ..
mkdir -p "website/pages/api/"

cwd=$(pwd)
python scripts/parse_sphinx.py -i "${cwd}/sphinx/build/html/" -o "${cwd}/website/pages/api/"

SPHINX_JS_DIR='sphinx/build/html/_static/'
DOCUSAURUS_JS_DIR='website/static/js/'

mkdir -p $DOCUSAURUS_JS_DIR

# move JS files from /sphinx/build/html/_static/*:
cp "${SPHINX_JS_DIR}documentation_options.js" "${DOCUSAURUS_JS_DIR}documentation_options.js"
cp "${SPHINX_JS_DIR}jquery.js" "${DOCUSAURUS_JS_DIR}jquery.js"
cp "${SPHINX_JS_DIR}underscore.js" "${DOCUSAURUS_JS_DIR}underscore.js"
cp "${SPHINX_JS_DIR}doctools.js" "${DOCUSAURUS_JS_DIR}doctools.js"
cp "${SPHINX_JS_DIR}language_data.js" "${DOCUSAURUS_JS_DIR}language_data.js"
cp "${SPHINX_JS_DIR}searchtools.js" "${DOCUSAURUS_JS_DIR}searchtools.js"

# searchindex.js is not static util
cp "sphinx/build/html/searchindex.js" "${DOCUSAURUS_JS_DIR}searchindex.js"

# copy module sources
cp -r "sphinx/build/html/_sources/" "website/static/_sphinx-sources/"

echo "-----------------------------------"
echo "Generating tutorials"
echo "-----------------------------------"
mkdir -p "website/_tutorials"
mkdir -p "website/static/files"
python scripts/parse_tutorials.py -w "${cwd}"

# Starting local server
cd website || exit

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
