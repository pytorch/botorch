#!/usr/bin/env python3

from .gp_regression import FixedNoiseGP, HeteroskedasticSingleTaskGP, SingleTaskGP
from .model import Model
from .multi_output_gp_regression import MultiOutputGP
from .multitask import MultiTaskGP


__all__ = [
    "FixedNoiseGP",
    "HeteroskedasticSingleTaskGP",
    "Model",
    "MultiOutputGP",
    "MultiTaskGP",
    "SingleTaskGP",
]
