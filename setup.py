#!/usr/bin/env python3

import numpy
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension


extensions = [Extension("botorch.qmc.sobol", ["botorch/qmc/sobol.pyx"])]

setup(
    name="botorch",
    version="pre-alpha",
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
