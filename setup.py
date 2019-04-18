#!/usr/bin/env python3

import sys

from setuptools import find_packages, setup
from setuptools.extension import Extension


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


# check for numpy (required for building Sobol cython)
try:
    import numpy
except ImportError:
    missing("numpy")


# check for Cython itself
try:
    from Cython.Build import cythonize
except ImportError:
    missing("cython")


# error out if setup dependencies not met
if fatals:
    sys.exit(
        "You need to fix the following issues before you can install botorch:\n - "
        + "\n - ".join(fatals)
    )


# TODO: Use torch Sobol once torch 1.1 is released
EXTENSIONS = [Extension("botorch.qmc.sobol", ["botorch/qmc/sobol.pyx"])]

DEV_REQUIRES = [
    "black",
    "flake8",
    "pytest>=3.6",
    "pytest-cov",
    "sphinx",
    "sphinx-autodoc-typehints",
]

TUTORIAL_REQUIRES = ["jupyter", "matplotlib"]


setup(
    name="botorch",
    version="0.1a2",
    description="Bayesian Optimization in PyTorch",
    author="Facebook, Inc.",
    license="MIT",
    url="https://github.com/facebookexternal/botorch",
    keywords=["Bayesian Optimization", "pytorch", "Gaussian Process"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    setup_requires=["cython", "numpy"],
    install_requires=["torch>=1.0.1", "gpytorch>=0.3.1", "scipy"],
    include_dirs=[numpy.get_include()],
    packages=find_packages(),
    ext_modules=cythonize(EXTENSIONS),
    extras_require={"dev": DEV_REQUIRES, "tutorial": TUTORIAL_REQUIRES},
)
