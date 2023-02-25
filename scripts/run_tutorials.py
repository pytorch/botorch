#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import tempfile
import time
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, Dict, Optional, Tuple

import nbformat
import pandas as pd
from memory_profiler import memory_usage
from nbconvert import PythonExporter


IGNORE = {  # ignored in smoke tests and full runs
    "vae_mnist.ipynb",  # requires setting paths to local data
    "bope.ipynb",  # flaky, keeps failing the workflows
    "preference_bo.ipynb",  # failing. Fix planned
    # Causing the tutorials to crash when run without smoke test. Likely OOM.
    # Fix planned.
    "constraint_active_search.ipynb",
    # Timing out
    "saasbo.ipynb",
    # Timing out
    "scalable_constrained_bo.ipynb",
}
IGNORE_SMOKE_TEST_ONLY = {  # only used in smoke tests
    "thompson_sampling.ipynb",  # very slow without KeOps + GPU
    "composite_mtbo.ipynb",  # TODO: very slow, figure out if we can make it faster
    "Multi_objective_multi_fidelity_BO.ipynb",  # TODO: very slow, speed up
}


def _read_command_line_output(command: str) -> str:
    output = subprocess.run(command.split(" "), stdout=subprocess.PIPE).stdout.decode(
        "utf-8"
    )
    return output


def get_mode_as_str(smoke_test: bool) -> str:
    return "smoke-test" if smoke_test else "standard"


def get_output_file_path(smoke_test: bool) -> str:
    """
    On push and in the nightly cron, a csv will be uploaded to
    https://github.com/pytorch/botorch/tree/artifacts/tutorial_performance_data .
    So file name contains time (for uniqueness) and commit hash (for debugging)
    """
    commit_hash = _read_command_line_output("git rev-parse --short HEAD").strip("\n")
    time = str(datetime.datetime.now())
    mode = get_mode_as_str(smoke_test=smoke_test)
    fname = f"{mode}_{commit_hash}_{time}.csv"
    return fname


def parse_ipynb(file: Path) -> str:
    with open(file, "r") as nb_file:
        nb_str = nb_file.read()
    nb = nbformat.reads(nb_str, nbformat.NO_CONVERT)
    exporter = PythonExporter()
    script, _ = exporter.from_notebook_node(nb)
    return script


def run_script(script: str, env: Optional[Dict[str, str]] = None) -> None:
    # need to keep the file around & close it so subprocess does not run into I/O issues
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf_name = tf.name
        with open(tf_name, "w") as tmp_script:
            tmp_script.write(script)
    if env is not None:
        env = {**os.environ, **env}
    run_out = subprocess.run(
        ["ipython", tf_name],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,  # Count runtime >30 minutes as a failure
    )
    os.remove(tf_name)
    return run_out


def run_tutorial(
    tutorial: Path, smoke_test: bool = False
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Runs the tutorial in a subprocess, catches any raised errors and returns
    them as a string, and returns runtime and memory information as a dict.
    """
    script = parse_ipynb(tutorial)
    tic = time.monotonic()
    print(f"Running tutorial {tutorial.name}.")
    env = {"SMOKE_TEST": "True"} if smoke_test else None
    try:
        mem_usage, run_out = memory_usage(
            (run_script, (script,), {"env": env}), retval=True, include_children=True
        )
    except subprocess.TimeoutExpired:
        error = f"Tutorial {tutorial.name} exceeded the maximum runtime of 30 minutes."
        return error, {}

    try:
        run_out.check_returncode()
    except CalledProcessError:
        error = "\n".join(
            [
                f"Encountered error running tutorial {tutorial.name}:",
                "stdout:",
                run_out.stdout,
                "stderr:",
                run_out.stderr,
            ]
        )
        return error, {}
    runtime = time.monotonic() - tic
    performance_info = {
        "runtime": runtime,
        "start_mem": mem_usage[0],
        "max_mem": max(mem_usage),
    }

    return None, performance_info


def run_tutorials(
    repo_dir: str,
    include_ignored: bool = False,
    smoke_test: bool = False,
    name: Optional[str] = None,
) -> None:
    """
    Run each tutorial, print statements on how it ran, and write a data set as a csv
    to a directory.
    """
    mode = "smoke test" if smoke_test else "standard"
    results_already_stored = (
        elt
        for elt in os.listdir()
        if elt[-4:] == ".csv" and elt.split("_")[0] in ("smoke-test", "standard")
    )
    for fname in results_already_stored:
        raise RuntimeError(
            f"There are already tutorial results files stored, such as {fname}. "
            "This is not allowed because GitHub Actions will look for all "
            "tutorial results files and write them to the 'artifacts' branch. "
            "Please remove all files matching pattern "
            "'standard_*.csv' or 'smoke-test_*.csv' in the current directory."
        )
    print(f"Running tutorial(s) in {mode} mode.")
    if not smoke_test:
        print("This may take a long time...")
    tutorial_dir = Path(repo_dir).joinpath("tutorials")
    num_runs = 0
    num_errors = 0
    ignored_tutorials = IGNORE if smoke_test else IGNORE | IGNORE_SMOKE_TEST_ONLY

    tutorials = sorted(
        t for t in tutorial_dir.iterdir() if t.is_file and t.suffix == ".ipynb"
    )
    if name is not None:
        tutorials = [t for t in tutorials if t.name == name]
        if len(tutorials) == 0:
            raise RuntimeError(f"Specified tutorial {name} not found in directory.")

    df = pd.DataFrame(
        {
            "name": [t.name for t in tutorials],
            "ran_successfully": False,
            "runtime": float("nan"),
            "start_mem": float("nan"),
            "max_mem": float("nan"),
        }
    ).set_index("name")

    for tutorial in tutorials:
        if not include_ignored and tutorial.name in ignored_tutorials:
            print(f"Ignoring tutorial {tutorial.name}.")
            continue
        num_runs += 1
        error, performance_info = run_tutorial(tutorial, smoke_test=smoke_test)
        if error:
            num_errors += 1
            print(error)
        else:
            print(
                f"Running tutorial {tutorial.name} took "
                f"{performance_info['runtime']:.2f} seconds. Memory usage "
                f"started at {performance_info['start_mem']} MB and the maximum"
                f" was {performance_info['max_mem']} MB."
            )
            df.loc[tutorial.name, "ran_successfully"] = True
            for k in ["runtime", "start_mem", "max_mem"]:
                df.loc[tutorial.name, k] = performance_info[k]
        break

    if num_errors > 0:
        raise RuntimeError(
            f"Running {num_runs} tutorials resulted in {num_errors} errors."
        )

    fname = get_output_file_path(smoke_test=smoke_test)
    print(f"Writing report to {fname}.")
    df.to_csv(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the tutorials.")
    parser.add_argument(
        "-p", "--path", metavar="path", required=True, help="botorch repo directory."
    )
    parser.add_argument(
        "-s", "--smoke", action="store_true", help="Run in smoke test mode."
    )
    parser.add_argument(
        "--include-ignored",
        action="store_true",
        help="Run all tutorials (incl. ignored).",
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Run a specific tutorial by name. The name should include the "
        ".ipynb extension. If the tutorial is on the ignore list, you still need "
        "to specify --include-ignored.",
    )
    args = parser.parse_args()
    run_tutorials(
        repo_dir=args.path,
        include_ignored=args.include_ignored,
        smoke_test=args.smoke,
        name=args.name,
    )
