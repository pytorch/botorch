#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .aug_branin import neg_aug_branin
from .aug_hartmann6 import neg_aug_hartmann6
from .aug_rosenbrock import neg_aug_rosenbrock
from .branin import neg_branin
from .eggholder import neg_eggholder
from .hartmann6 import neg_hartmann6
from .holder_table import neg_holder_table
from .michalewicz import neg_michalewicz
from .styblinski_tang import neg_styblinski_tang


__all__ = [
    "neg_branin",
    "neg_eggholder",
    "neg_hartmann6",
    "neg_holder_table",
    "neg_michalewicz",
    "neg_styblinski_tang",
    "neg_aug_branin",
    "neg_aug_hartmann6",
    "neg_aug_rosenbrock",
]
