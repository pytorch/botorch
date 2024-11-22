#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import yaml


def parse_requirements(filepath):
    """Parse requirements file and return a dict of package versions."""
    versions = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Handle different requirement formats
                if "==" in line:
                    pkg, version = line.split("==")
                    versions[pkg.strip().lower()] = version.strip()
    return versions


def parse_precommit_config(filepath):
    """Parse pre-commit config and extract ufmt repo rev and hook dependencies."""
    with open(filepath) as f:
        config = yaml.safe_load(f)

    versions = {}
    for repo in config["repos"]:
        if "https://github.com/omnilib/ufmt" in repo.get("repo", ""):
            # Get ufmt version from rev - assumes fixed format: vX.Y.Z
            versions["ufmt"] = repo.get("rev", "").replace("v", "")

            # Get dependency versions
            for hook in repo["hooks"]:
                if hook["id"] == "ufmt":
                    for dep in hook.get("additional_dependencies", []):
                        if "==" in dep:
                            pkg, version = dep.split("==")
                            versions[pkg.strip().lower()] = version.strip()
            break
    return versions


def main():
    # Find the pre-commit config and requirements files
    config_file = Path(".pre-commit-config.yaml")
    requirements_file = Path("requirements-fmt.txt")

    if not config_file.exists():
        print(f"Error: Could not find {config_file}")
        sys.exit(1)

    if not requirements_file.exists():
        print(f"Error: Could not find {requirements_file}")
        sys.exit(1)

    # Parse both files
    req_versions = parse_requirements(requirements_file)
    config_versions = parse_precommit_config(config_file)

    # Check versions
    mismatches = []
    for pkg, req_ver in req_versions.items():
        req_ver = req_versions.get(pkg, None)
        config_ver = config_versions.get(pkg, None)

        if req_ver != config_ver:
            found_version_str = f"{pkg}: {requirements_file} has {req_ver},"
            if pkg == "ufmt":
                mismatches.append(
                    f"{found_version_str} pre-commit config rev has v{config_ver}"
                )
            else:
                mismatches.append(
                    f"{found_version_str} pre-commit config has {config_ver}"
                )

    # Report results
    if mismatches:
        msg_str = "".join("\n\t" + msg for msg in mismatches)
        print(
            f"Version mismatches found:{msg_str}"
            "\nPlease update the versions in `.pre-commit-config.yaml` to be "
            "consistent with those in `requirements-fmt.txt` (source of truth)."
            "\nNote: all versions must be pinned exactly ('==X.Y.Z') in both files."
        )
        sys.exit(1)
    else:
        print("All versions match!")
        sys.exit(0)


if __name__ == "__main__":
    main()
