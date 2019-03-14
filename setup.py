#!/usr/bin/env python3

import sys

from setuptools import find_packages, setup
from setuptools.extension import Extension


REQUIRED_MAJOR = 3
REQUIRED_MINOR = 6

fatals = []


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
    fatals.append(
        "The '{}' package is missing. Please install it before "
        "running the setup script.".format(package_name)
    )


try:
    import numpy
except ImportError:
    missing("numpy")

try:
    from Cython.Build import cythonize
except ImportError:
    missing("cython")


if fatals:
    sys.exit(
        "You need to fix the following issues before you can install botorch:\n - "
        + "\n - ".join(fatals)
    )


# TODO: Use torch Sobol when available: https://github.com/pytorch/pytorch/pull/10505
extensions = [Extension("botorch.qmc.sobol", ["botorch/qmc/sobol.pyx"])]

setup(
    name="botorch",
    version="alpha",
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
    setup_requires=["cython", "numpy"],  # TODO: Ship generated .c sources
    install_requires=["gpytorch>=0.2.1", "scipy"],
    include_dirs=[numpy.get_include()],
    packages=find_packages(),
    ext_modules=cythonize(extensions),
)
