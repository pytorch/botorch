#!/usr/bin/env python3

import os
import re
import sys

from setuptools import find_packages, setup


REQUIRED_MAJOR = 3
REQUIRED_MINOR = 6

fatals = []


# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    fatals.append(
        (
            "Your version of python ({major}.{minor}) is too old. You need "
            "python >= {required_major}.{required_minor}."
        ).format(
            major=sys.version_info.major,
            minor=sys.version_info.minor,
            required_minor=REQUIRED_MINOR,
            required_major=REQUIRED_MAJOR,
        )
    )


def missing(package_name):
    """Formatting helper for errors."""
    fatals.append(
        "The '{}' package is missing. Please install it before "
        "running the setup script.".format(package_name)
    )


# error out if setup dependencies not met
if fatals:
    sys.exit(
        "You need to fix the following issues before you can install botorch:\n - "
        + "\n - ".join(fatals)
    )


TEST_REQUIRES = ["pytest", "pytest-cov"]

DEV_REQUIRES = TEST_REQUIRES + ["black", "flake8", "sphinx", "sphinx-autodoc-typehints"]

TUTORIAL_REQUIRES = ["jupyter", "matplotlib", "torchvision"]

# get version string from setup.py
with open(os.path.join(os.path.dirname(__file__), "botorch/__init__.py"), "r") as f:
    version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)

setup(
    name="botorch",
    version=version,
    description="Bayesian Optimization in PyTorch",
    author="Facebook, Inc.",
    license="MIT",
    url="https://github.com/pytorch/botorch",
    keywords=["Bayesian Optimization", "pytorch", "Gaussian Process"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    install_requires=["torch>=1.0.1", "gpytorch>=0.3.2", "scipy"],
    packages=find_packages(),
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
        "tutorial": TUTORIAL_REQUIRES,
    },
)
