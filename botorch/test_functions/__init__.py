#!/usr/bin/env python3

from .branin import neg_branin
from .hartmann6 import neg_hartmann6
from .holder_table import neg_holder_table
from .styblinski_tang import neg_styblinski_tang


__all__ = ["neg_branin", "neg_hartmann6", "neg_holder_table", "neg_styblinski_tang"]
