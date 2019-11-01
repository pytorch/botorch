#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

usage() {
  echo "Usage: $0 [-d] [-v VERSION]"
  echo ""
  echo "Build and push updated BoTorch site. Will either update latest or bump stable version."
  echo ""
  echo "  -d           Use Docusaurus bot GitHub credentials. If not specified, will use default GitHub credentials."
  echo "  -v=VERSION   Build site for new library version. If not specified, will update master."
  echo ""
  exit 1
}

DOCUSAURUS_BOT=false
VERSION=false

while getopts 'dhv:' option; do
  case "${option}" in
    d)
      DOCUSAURUS_BOT=true
      ;;
    h)
      usage
      ;;
    v)
      VERSION=${OPTARG}
      ;;
    *)
      usage
      ;;
  esac
done


# Function to get absolute filename
fullpath() {
  echo "$(cd "$(dirname "$1")" || exit; pwd -P)/$(basename "$1")"
}

# Current directory (needed for cleanup later)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make temporary directory
WORK_DIR=$(mktemp -d)
cd "${WORK_DIR}" || exit

if [[ $DOCUSAURUS_BOT == true ]]; then
  # Assumes docusaurus bot credentials have been stored in ~/.netrc, e.g. via
  # git config --global user.email "docusaurus-bot@users.noreply.github.com"
  # git config --global user.name "BoTorch website deployment script"
  # echo "machine github.com login docusaurus-bot password $GITHUB_TOKEN" > ~/.netrc

  # Clone both master & gh-pages branches
  git clone https://docusaurus-bot@github.com/pytorch/botorch.git botorch-master
  git clone --branch gh-pages https://docusaurus-bot@github.com/pytorch/botorch.git botorch-gh-pages
else
  git clone git@github.com:pytorch/botorch.git botorch-master
  git clone --branch gh-pages git@github.com:pytorch/botorch.git botorch-gh-pages
fi

# A few notes about the script below:
# * Docusaurus versioning was designed to *only* version the markdown
#   files in the docs/ subdirectory. We are repurposing parts of Docusaurus
#   versioning, but snapshotting the entire site. Versions of the site are
#   stored in the v/ subdirectory on gh-pages:
#
#   --gh-pages/
#     |-- api/
#     |-- css/
#     |-- docs/
#     |   ...
#     |-- v/
#     |   |-- 1.0.1/
#     |   |-- 1.0.2/
#     |   |   ...
#     |   |-- latest/
#     |   ..
#     |-- versions.html
#
# * The stable version is in the top-level directory. It is also
#   placed into the v/ subdirectory so that it does not need to
#   be built again when the version is augmented.
# * We want to support serving / building the Docusaurus site locally
#   without any versions. This means that we have to keep versions.js
#   outside of the website/ subdirectory.
# * We do not want to have a tracked file that contains all versions of
#   the site or the latest version. Instead, we determine this at runtime.
#   We use what's on gh-pages in the versions subdirectory as the
#   source of truth for available versions and use the latest tag on
#   the master branch as the source of truth for the latest version.

if [[ $VERSION == false ]]; then
  echo "-----------------------------------------"
  echo "Updating latest (master) version of site "
  echo "-----------------------------------------"

  # Populate _versions.json from existing versions; this is used
  # by versions.js & needed to build the site (note that we don't actually
  # use versions.js for latest build, but we do need versions.js
  # in website/pages in order to use docusaurus-versions)
  CMD="import os, json; "
  CMD+="vs = [v for v in os.listdir('botorch-gh-pages/v') if v != 'latest' and not v.startswith('.')]; "
  CMD+="print(json.dumps(vs))"
  python3 -c "$CMD" > botorch-master/website/_versions.json

  # Move versions.js to website subdirectory.
  # This is the page you see when click on version in navbar.
  cp botorch-master/scripts/versions.js botorch-master/website/pages/en/versions.js
  cd botorch-master/website || exit

  # Replace baseUrl (set to /v/latest/) & disable Algolia
  CONFIG_FILE=$(fullpath "siteConfig.js")
  python3 ../scripts/patch_site_config.py -f "${CONFIG_FILE}" -b "/v/latest/" --disable_algolia

  # Tag site with "latest" version
  yarn
  yarn run version latest

  # Build site
  cd .. || exit
  ./scripts/build_docs.sh -b
  rm -rf website/build/botorch/docs/next  # don't need this

  # Move built site to gh-pages (but keep old versions.js)
  cd "${WORK_DIR}" || exit
  cp botorch-gh-pages/v/latest/versions.html versions.html
  rm -rf botorch-gh-pages/v/latest
  mv botorch-master/website/build/botorch botorch-gh-pages/v/latest
  # versions.html goes both in top-level and under en/ (default language)
  cp versions.html botorch-gh-pages/v/latest/versions.html
  cp versions.html botorch-gh-pages/v/latest/en/versions.html
  cp -R botorch-master/.circleci botorch-gh-pages/

  # Push changes to gh-pages
  cd botorch-gh-pages || exit
  git add .
  git commit -m 'Update latest version of site'
  git push

else
  echo "-----------------------------------------"
  echo "Building new version ($VERSION) of site "
  echo "-----------------------------------------"

  # Checkout master branch with specified tag
  cd botorch-master || exit
  git fetch --tags
  git checkout "v${VERSION}"

  # Populate _versions.json from existing versions; this contains a list
  # of versions present in gh-pages (excluding latest). This is then used
  # to populate versions.js (which forms the page that people see when they
  # click on version number in navbar).
  # Note that this script doesn't allow building a version of the site that
  # is already on gh-pages.
  CMD="import os, json; "
  CMD+="vs = [v for v in os.listdir('../botorch-gh-pages/v') if v != 'latest' and not v.startswith('.')]; "
  CMD+="assert '${VERSION}' not in vs, '${VERSION} is already on gh-pages.'; "
  CMD+="vs.append('${VERSION}'); "
  CMD+="print(json.dumps(vs))"
  python3 -c "$CMD" > website/_versions.json

  cp scripts/versions.js website/pages/en/versions.js

  # Tag site as 'stable'
  cd website || exit
  yarn
  yarn run version stable

  # Build new version of site (this will be stable, default version)
  cd .. || exit
  ./scripts/build_docs.sh -b

  # Move built site to new folder (new-site) & carry over old versions
  # from existing gh-pages
  cd "${WORK_DIR}" || exit
  rm -rf botorch-master/website/build/botorch/docs/next  # don't need this
  mv botorch-master/website/build/botorch new-site
  mv botorch-gh-pages/v new-site/v

  # Build new version of site (to be placed in v/$VERSION/)
  # the only thing that changes here is the baseUrl (for nav purposes)
  # we build this now so that in the future, we can just bump version and not move
  # previous stable to versions
  cd botorch-master/website || exit

  # Replace baseUrl (set to /v/$VERSION/) & disable Algolia
  CONFIG_FILE=$(fullpath "siteConfig.js")
  python3 ../scripts/patch_site_config.py -f "${CONFIG_FILE}" -b "/v/${VERSION}/" --disable_algolia

  # Tag exact version & build site
  yarn run version "${VERSION}"
  cd .. || exit
  # Only run Docusaurus (skip tutorial build & Sphinx)
  ./scripts/build_docs.sh -b -o
  rm -rf website/build/botorch/docs/next  # don't need this
  rm -rf website/build/botorch/docs/stable  # or this
  mv website/build/botorch "../new-site/v/${VERSION}"

  # Need to run script to update versions.js for previous versions in
  # new-site/v with the newly built versions.js. Otherwise,
  # the versions.js for older versions in versions subdirectory
  # won't be up-to-date and will not have a way to navigate back to
  # newer versions. This is the only part of the old versions that
  # needs to be updated when a new version is built.
  cd "${WORK_DIR}" || exit
  python3 botorch-master/scripts/update_versions_html.py -p "${WORK_DIR}"
  cp -R botorch-master/.circleci new-site/

  # move contents of newsite to botorch-gh-pages, preserving commit history
  rm -rfv ./botorch-gh-pages/*
  rsync -avh ./new-site/ ./botorch-gh-pages/
  cd botorch-gh-pages || exit
  git add --all
  git commit -m "Publish version ${VERSION} of site"
  git push

fi

# Clean up
cd "${SCRIPT_DIR}" || exit
rm -rf "${WORK_DIR}"
