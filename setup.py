#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from setuptools import find_packages, setup

# Minimum required python version
REQUIRED_MAJOR = 3
REQUIRED_MINOR = 10

# Requirements for testing, formatting, and tutorials
TEST_REQUIRES = ["pytest", "pytest-cov"]
FMT_REQUIRES = ["flake8", "ufmt", "flake8-docstrings"]
TUTORIALS_REQUIRES = [
    "ax-platform",
    "cma",
    "jupyter",
    "kaleido",
    "matplotlib",
    "memory_profiler",
    "papermill",
    "pykeops",
    "torchvision",
    "mdformat",
    "pandas",
    "lxml",
    "mdformat-myst",
    "tabulate",
]

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

# Assign root dir location for later use
root_dir = os.path.dirname(__file__)


def read_deps_from_file(filname):
    """Read in requirements file and return items as list of strings"""
    with open(os.path.join(root_dir, filname), "r") as fh:
        return [line.strip() for line in fh.readlines() if not line.startswith("#")]


# Read in the requirements from the requirements.txt file
install_requires = read_deps_from_file("requirements.txt")

# Allow non-pinned (usually dev) versions of gpytorch and linear_operator
if os.environ.get("ALLOW_LATEST_GPYTORCH_LINOP"):
    # Allows more recent previously installed versions. If there is no
    # previously installed version, installs the latest release.
    install_requires = [
        (
            dep.replace("==", ">=")
            if "gpytorch" in dep or "linear_operator" in dep
            else dep
        )
        for dep in install_requires
    ]

# Read in pinned versions of the formatting tools
FMT_REQUIRES += read_deps_from_file("requirements-fmt.txt")
# Dev is test + formatting + docs generation
DEV_REQUIRES = TEST_REQUIRES + FMT_REQUIRES + ["sphinx", "sphinx-rtd-theme"]

# read in README.md as the long description
with open(os.path.join(root_dir, "README.md"), "r") as fh:
    long_description = fh.read()

setup(
    name="botorch",
    description="Bayesian Optimization in PyTorch",
    author="Meta Platforms, Inc.",
    license="MIT",
    url="https://botorch.org",
    project_urls={
        "Documentation": "https://botorch.org",
        "Source": "https://github.com/pytorch/botorch",
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
    python_requires=f">={REQUIRED_MAJOR}.{REQUIRED_MINOR}",
    packages=find_packages(exclude=["test", "test.*"]),
    install_requires=install_requires,
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
        "tutorials": TUTORIALS_REQUIRES,
    },
)
