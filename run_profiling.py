# TODO: this isn't working yet! The inner function must also be profiled
# (at least for memory).
# Otherwise, we won't catch allocations that happen inside it if they are
# deallocated at the end .
"""
Run this file from the command line as
`python run_profiling [fn_name] [optional extra_args]`.

It will run functions from `botorch/profiling.py`, profile them according to either
time or memory (pre-specified), and print the output.

Functions:
    - run_test_memory_fn (memory)
    - run_qnei (memory)
    - run_qnehvi (memory)
    - run_fit_fully_bayesian_model_nuts (time)
    - run_large_t_batch_posterior_sampling (memory)

Example:
    `python run_profiling run_qnei 100`

Output:
    Filename: [...]/botorch/botorch/profiling.py

    Line #    Mem usage    Increment  Occurrences   Line Contents
    =============================================================
        46    175.8 MiB    175.8 MiB           1   def _memory_intensive_fn(n: int):
        47    938.7 MiB    763.0 MiB           1       arr = np.random.random(n)
        48    938.7 MiB      0.0 MiB           1       return arr[0]


    Filename: [...]/botorch/botorch/profiling.py

    Line #    Mem usage    Increment  Occurrences   Line Contents
    =============================================================
        51    175.8 MiB    175.8 MiB           1   @memory_profile
        52                                         def run_test_memory_fn(n: int = 10000000):  # noqa: E501
        53    175.8 MiB      0.0 MiB           1       memory_profile(_memory_intensive_fn)(n)
"""

import sys

from botorch.profiling import (
    run_fit_fully_bayesian_model_nuts,
    run_large_t_batch_posterior_sampling,
    run_qnehvi,
    run_qnei,
    run_test_memory_fn,
    time_profiler,
)


fn_names = {
    "run_test_memory_fn": (run_test_memory_fn, "memory"),
    "run_qnei": (run_qnei, "memory"),
    "run_qnehvi": (run_qnehvi, "memory"),
    "run_fit_fully_bayesian_model_nuts": (run_fit_fully_bayesian_model_nuts, "time"),
    "run_large_t_batch_posterior_sampling": (
        run_large_t_batch_posterior_sampling,
        "memory",
    ),
}


def run_and_profile_fn(problem_name: str, *additional_args) -> None:
    """
    Example: "python profiling.py qnei 2"
    """

    fn, memory_or_time = fn_names[problem_name]
    if memory_or_time == "time":
        time_profiler.enable()
    fn(*additional_args)
    if memory_or_time == "time":
        time_profiler.disable()
        time_profiler.print_stats()


if __name__ == "__main__":
    probem_name = sys.argv[1]
    additional_args = (int(elt) for elt in sys.argv[2:])
    run_and_profile_fn(probem_name, *additional_args)
