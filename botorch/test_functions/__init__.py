#!/usr/bin/env python3

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
]
