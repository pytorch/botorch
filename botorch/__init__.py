#!/usr/bin/env python3

import os
import re

from . import acquisition, exceptions, models, optim, posteriors, test_functions
from .cross_validation import batch_cross_validation
from .fit import fit_gpytorch_model
from .gen import gen_candidates_scipy, gen_candidates_torch, get_best_candidates
from .utils import manual_seed


# get version string from setup.py
with open(os.path.join(os.path.dirname(__file__), os.pardir, "setup.py"), "r") as f:
    __version__ = re.search(r"version=['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)


__all__ = [
    "acquisition",
    "batch_cross_validation",
    "exceptions",
    "fit_gpytorch_model",
    "gen_candidates_scipy",
    "gen_candidates_torch",
    "get_best_candidates",
    "manual_seed",
    "models",
    "optim",
    "posteriors",
    "test_functions",
]
