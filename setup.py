#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
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

DEV_REQUIRES = TEST_REQUIRES + [
    "flake8",
    "sphinx",
    "black==21.4b2",
    "libcst==0.3.19",
    "usort==0.6.4",
    "ufmt",
]

TUTORIALS_REQUIRES = [
    "ax-platform",
    "cma",
    "jupyter",
    "kaleido",
    "matplotlib",
    "memory_profiler",
    "pykeops",
    "torchvision",
]

root_dir = os.path.dirname(__file__)

# read in README.md as the long description
with open(os.path.join(root_dir, "README.md"), "r") as fh:
    long_description = fh.read()

setup(
    name="botorch",
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
    setup_requires=["setuptools-scm"],
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "write_to": os.path.join(root_dir, "botorch", "version.py"),
        "local_scheme": (
            "no-local-version"
            if os.environ.get("SCM_NO_LOCAL_VERSION", False)
            else "node-and-date"
        ),
    },
    packages=find_packages(exclude=["test", "test.*"]),
    install_requires=[
        "torch>=1.9",
        "gpytorch>=1.6",
        "scipy",
        "multipledispatch",
        "pyro-ppl==1.8.0",
    ],
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
        "tutorials": TUTORIALS_REQUIRES,
    },
)
