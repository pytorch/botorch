#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging


LOG_LEVEL_DEFAULT = logging.CRITICAL


def _get_logger(
    name: str = "botorch", level: int = LOG_LEVEL_DEFAULT
) -> logging.Logger:
    """Gets a default botorch logger

    Logging level can be tuned via botorch.setting.log_level

    Args:
        name: Name for logger instance
        level: Logging threshhold for the given logger. Logs of greater or
            equal severity will be printed to STDERR
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add timestamps to log messages.
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(levelname)s %(asctime)s] %(name)s: %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    return logger


logger = _get_logger()
