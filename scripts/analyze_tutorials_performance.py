#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
The "artifacts" branch stores CSVs of tutorial runtime and memory:
https://github.com/pytorch/botorch/tree/artifacts/tutorial_performance_data

This file should be run from the artifacts branch, where the data lives,
and from the repo root.
So that it is properly version-controlled, it is checked into
main, and mirrored to the artifacts branch via GH Actions. It

1) Merges those CSVs into one
2) Runs 'notebooks/tutorials_performance_tracking.ipynb' to produce
visualizations
"""
import os
import pandas as pd
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_tutorial(dirname: str, fname: str) -> None:
    with open(os.path.join(dirname, fname), "r") as infile:
        nb_str = infile.read()
    nb = nbformat.reads(nb_str, nbformat.NO_CONVERT)
    timeout = int(60 * 60 * 2.5)
    ep = ExecutePreprocessor(timeout=timeout)

    # execute notebook, using `tutorial_dir` as working directory
    ep.preprocess(nb, {"metadata": {"path": dirname}})


def read_data_one_file(data_dir: str, fname: str) -> pd.DataFrame:
    """
    The file name format is {mode}_{commit_hash}_{date_time}.csv
    """
    mode, commit_hash, date_time = fname[:-4].split("_")
    df = pd.read_csv(os.path.join(data_dir, fname)).assign(
        mode=mode,
        commit_hash=commit_hash,
        datetime=pd.to_datetime(date_time),
        fname=fname,
    )
    # clean out '.ipynb' if it is present
    df["name"] = df["name"].apply(lambda x: x[: -len(".ipynb")])
    return df


def concatenate_data(data_dir: str) -> None:
    """Read in all data and write it to one file."""
    df = pd.concat(
        (
            read_data_one_file(data_dir=data_dir, fname=fname)
            for fname in os.listdir(data_dir)
            if fname != "all_data.csv"
        ),
        ignore_index=True,
    ).sort_values(["mode", "datetime"], ignore_index=True)
    df.to_csv(os.path.join(data_dir, "all_data.csv"), index=False)


if __name__ == "__main__":
    repo_root = os.getcwd()
    data_dir = os.path.join(repo_root, "tutorial_performance_data")
    concatenate_data(data_dir)
    run_tutorial(
        os.path.join(repo_root, "notebooks"), "tutorials_performance_tracking.ipynb"
    )
