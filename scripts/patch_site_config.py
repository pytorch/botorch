#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import re


def patch_config(
    config_file: str, base_url: str = None, disable_algolia: bool = True
) -> None:
    config = open(config_file, "r").read()

    if base_url is not None:
        config = re.sub("baseUrl = '/';", "baseUrl = '{}';".format(base_url), config)
    if disable_algolia is True:
        config = re.sub(
            "const includeAlgolia = true;", "const includeAlgolia = false;", config
        )

    with open(config_file, "w") as outfile:
        outfile.write(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Path Docusaurus siteConfig.js file when building site."
    )
    parser.add_argument(
        "-f",
        "--config_file",
        metavar="path",
        required=True,
        help="Path to configuration file.",
    )
    parser.add_argument(
        "-b",
        "--base_url",
        type=str,
        required=False,
        help="Value for baseUrl.",
        default=None,
    )
    parser.add_argument(
        "--disable_algolia",
        required=False,
        action="store_true",
        help="Disable algolia.",
    )
    args = parser.parse_args()
    patch_config(args.config_file, args.base_url, args.disable_algolia)
