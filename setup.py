#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import sys

from setuptools import find_packages, setup


REQUIRED_MAJOR = 3
REQUIRED_MINOR = 7

# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need "
        "python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)


TEST_REQUIRES = ["pytest", "pytest-cov"]

DEV_REQUIRES = TEST_REQUIRES + ["black", "flake8", "sphinx", "sphinx-autodoc-typehints"]

TUTORIALS_REQUIRES = ["jupyter", "matplotlib", "cma", "torchvision"]

# get version string from module
with open(os.path.join(os.path.dirname(__file__), "botorch/__init__.py"), "r") as f:
    version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)

# read in README.md as the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="botorch",
    version=version,
    description="Bayesian Optimization in PyTorch",
    author="Facebook, Inc.",
    license="MIT",
    url="https://botorch.org",
    project_urls={
        "Documentation": "https://botorch.org",
        "Source": "https://github.com/pytorch/botorch",
        "conda": "https://anaconda.org/pytorch/botorch",
    },
    keywords=["Bayesian optimization", "PyTorch"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=["torch>=1.4", "gpytorch>=1.1.1", "scipy"],
    packages=find_packages(),
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
        "tutorials": TUTORIALS_REQUIRES,
    },
)
