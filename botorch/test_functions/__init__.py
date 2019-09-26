#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .multi_fidelity import AugmentedBranin, AugmentedHartmann, AugmentedRosenbrock
from .synthetic import (
    Ackley,
    Beale,
    Branin,
    Bukin,
    Cosine8,
    DixonPrice,
    DropWave,
    EggHolder,
    Griewank,
    Hartmann,
    HolderTable,
    Levy,
    Michalewicz,
    Powell,
    Rastrigin,
    Rosenbrock,
    Shekel,
    SixHumpCamel,
    StyblinskiTang,
    SyntheticTestFunction,
    ThreeHumpCamel,
)


__all__ = [
    "Ackley",
    "AugmentedBranin",
    "AugmentedHartmann",
    "AugmentedRosenbrock",
    "Beale",
    "Branin",
    "Bukin",
    "Cosine8",
    "DixonPrice",
    "DropWave",
    "EggHolder",
    "Griewank",
    "Hartmann",
    "HolderTable",
    "Levy",
    "Michalewicz",
    "Powell",
    "Rastrigin",
    "Rosenbrock",
    "Shekel",
    "SixHumpCamel",
    "StyblinskiTang",
    "SyntheticTestFunction",
    "ThreeHumpCamel",
]
