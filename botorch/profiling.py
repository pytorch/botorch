from typing import Callable, Optional, Tuple
from unittest import mock

import numpy as np

import torch
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.sampling.samplers import IIDNormalSampler
from line_profiler import LineProfiler
from memory_profiler import profile as memory_profile


time_profiler = LineProfiler()


def time_profile(fn: Callable):
    def wrapper(*args, **kwargs):
        time_profiler.add_function(fn)
        return fn(*args, **kwargs)

    return wrapper


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_train_x_and_y(
    n: int, dim: int, batch_dim: Optional[int] = None
) -> Tuple[torch.tensor, torch.tensor]:
    tkwargs = {"device": _get_device(), "dtype": torch.double}
    if batch_dim is None:
        x = torch.rand(n, dim, **tkwargs)
    else:
        x = torch.rand(batch_dim, n, dim, **tkwargs)
    y = torch.sin(x).sum(dim=-1, keepdim=True)
    return x, y


def _memory_intensive_fn(n: int):
    arr = np.random.random(n)
    return arr[0]


@memory_profile
def run_test_memory_fn(n: int = 100000000):
    """
    Run a simple memory-intensive function.

    Note that memory usage spikes within `_memory_intensive_fn` but not within
    `run_test_memory_fn` due to (good) garbage collection. This may only happen
    for larger values of `n`.
    """
    memory_profile(_memory_intensive_fn)(n)


@memory_profile
def run_qnei(q: int = 100, dim: int = 10, n: int = 50, batch_dim: int = 5) -> None:
    """
    Run qNoisyExpectedImprovement.

    Becomes memory-intensive as 'q' increases, risking OOMs. My Mac starts freezing
    up around q = 10000. Interestingly, there doesn't seem to be any _deallocation_
    within qNEI.__call__. Memory allocation increases a lot the first time it's called
    and then slightly on subsequent calls. This is either due to caching or poor garbage
    collection.

    TODO:
        1) make sure we're profiling the right method
        2) See if there's an issue with garbage collection because oddly
           memory is not being deallocated after call

    Example output with q=1000, stripping comments and docstring:

        Line #    Mem usage    Increment  Occurrences   Line Contents
    =============================================================
        63    176.9 MiB    176.9 MiB           1   @memory_profile
        64                                         def run_qnei(q: int = 100, dim: int = 10, n: int = 50, batch_dim: int = 5) -> None:  # noqa: E501
        83    177.1 MiB      0.3 MiB           1       train_X, train_Y = _get_train_x_and_y(n, dim, batch_dim)
        84
        85    178.3 MiB      1.2 MiB           1       model = SingleTaskGP(train_X, train_Y)
        86    185.1 MiB      6.8 MiB           1       qNEI = qNoisyExpectedImprovement(model, X_baseline=train_X)
        87
        88    185.1 MiB      0.0 MiB           1       test_X = torch.rand(batch_dim, q, dim)
        89
        92    185.1 MiB      0.0 MiB           1       wrapped_qnei_call = memory_profile(qNEI.__call__)
        93    677.2 MiB    492.1 MiB           1       qNEI(test_X)
        94                                             # wrapped_qnei_call(test_X)
        95    699.0 MiB     21.9 MiB           1       qNEI(test_X)
        96    721.3 MiB     22.2 MiB           1       qNEI(test_X)

    Args:
        q: Starts freezing up my Mac at 10000; q=1000 runs.
        dim: dimension of feature space. Realistically might be 100.
        n: number of points considered jointly. Realistically might be500
        batch_dim: Default 5

    """
    train_X, train_Y = _get_train_x_and_y(n, dim, batch_dim)

    model = SingleTaskGP(train_X, train_Y)
    qNEI = qNoisyExpectedImprovement(model, X_baseline=train_X)

    test_X = torch.rand(batch_dim, q, dim)

    # we can't do `memory_profile(qNEI)` since `memory_profile` doesn't work on an
    # instance, but profiling `qNEI.__call__` accomplishes the same thing.
    # wrapped_qnei_call = memory_profile(qNEI.__call__)
    qNEI(test_X)
    qNEI(test_X)
    qNEI(test_X)


@memory_profile
def run_qnehvi(q: int = 100, dim: int = 2, n: int = 50) -> None:
    """
    Noisy Expected Hypervolume Improvement

    should have similar memory issues to run_qnei
    Based on https://botorch.org/tutorials/multi_objective_bo
    # TODO figure out what reasonable input dimensions are
    """
    tkwargs = {"device": _get_device(), "dtype": torch.double}
    train_X = torch.rand(10, 4, **tkwargs)
    train_Y = torch.sin(train_X[:, :1])
    model = SaasFullyBayesianSingleTaskGP(
        train_X=train_X, train_Y=train_Y, train_Yvar=None
    )
    # This is not the function we're interested in, but it needs to be run so
    # the model can be fit.
    fit_fully_bayesian_model_nuts(
        model, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
    )
    sampler = IIDNormalSampler(num_samples=2)
    qNEHVI = qNoisyExpectedHypervolumeImprovement(  # noqa: F841
        model=ModelListGP(model, model),
        X_baseline=train_X,
        ref_point=torch.zeros(2, **tkwargs),
        sampler=sampler,
    )
    # TODO: figure out what to call qNEHVI on
    raise NotImplementedError


def run_fit_fully_bayesian_model_nuts(dim: int = 100, n: int = 200) -> None:
    """
    Profile `fit_fully_bayesian_model_nuts` for time.

    * Almost all of the time in `fit_fully_bayesian_model_nuts` is spent in
        `mcmc.run`.
    * Almost all of the time in `mcmc.run` is spent in `MCMC.sampler.run`.
    * Almost all of the time in `MCMC.sampler.run` is in `_gen_samples`
    * The time in `_gen_samples` is mainly spent in `kernel.sample`, with some
        spent in `kernel.setup`.

    We can't profile `mcmc.run` normally because it's decorated by `poutine.block`,
    which is generated and thus doesn't have normal lines of code to attach
    line-profiler output to. (It has no `__code__` attribute). So I'm just replacing
    that decorator with the line profiler.

    https://stackoverflow.com/questions/7667567/can-i-patch-a-python-decorator-before-it-wraps-a-function
    """
    profile_MCMC_run = False
    profile_fit_fully_bayesian_model_nuts = False

    train_X, train_Y = _get_train_x_and_y(n, dim)

    gp = SaasFullyBayesianSingleTaskGP(train_X=train_X, train_Y=train_Y)
    # replace existing decorator with time_profile
    import pyro

    if profile_MCMC_run:
        with mock.patch("pyro.poutine.block", time_profile):
            import importlib

            importlib.reload(pyro.infer.mcmc.api)

    from pyro.infer.mcmc.api import MCMC

    with (
        mock.patch.object(pyro.infer.mcmc.MCMC, "run", MCMC.run),
        mock.patch(
            "pyro.infer.mcmc.api._UnarySampler.run",
            time_profile(pyro.infer.mcmc.api._UnarySampler.run),
        ),
        mock.patch(
            "pyro.infer.mcmc.api._MultiSampler.run",
            time_profile(pyro.infer.mcmc.api._MultiSampler.run),
        ),
        mock.patch(
            "pyro.infer.mcmc.api._gen_samples",
            time_profile(pyro.infer.mcmc.api._gen_samples),
        ),
    ):
        fit_fn = (
            time_profile(fit_fully_bayesian_model_nuts)
            if profile_fit_fully_bayesian_model_nuts
            else fit_fully_bayesian_model_nuts
        )
        fit_fn(gp, warmup_steps=32, num_samples=16, thinning=16, disable_progbar=True)


@memory_profile
def run_large_t_batch_posterior_sampling(
    b: int = 1000000,
    dim: int = 1,
    n_train: int = 2,
    train_batch_dim: int = 1,
    n_points: int = 1,
) -> None:
    """
    Similar to `run_qnei`, it's interesting that memory usage _stays_ elevated after
    calling it. Due to either caching or an issue with garbage collection.

        Line #    Mem usage    Increment  Occurrences   Line Contents
    =============================================================
    209    175.8 MiB    175.8 MiB           1   @memory_profile
    210                                         def run_large_t_batch_posterior_sampling(  # noqa: E501
    211                                             b: int = 1000000,
    212                                             dim: int = 1,
    213                                             n_train: int = 2,
    214                                             train_batch_dim: int = 1,
    215                                             n_points: int = 1,
    216                                         ) -> None:
    222    176.0 MiB      0.2 MiB           1       train_X, train_Y = _get_train_x_and_y(n_train, dim, train_batch_dim)
    223
    224    177.1 MiB      1.1 MiB           1       model = SingleTaskGP(train_X, train_Y)
    225    178.9 MiB      1.8 MiB           1       posterior = model.posterior(train_X[0, :, :])
    226                                             # memory_profile(posterior.sample)(sample_shape=torch.Size([b, n_points]))
    227    224.9 MiB     46.0 MiB           1       posterior.sample(sample_shape=torch.Size([b, n_points]))
    228    224.9 MiB      0.0 MiB           1       posterior.sample(sample_shape=torch.Size([b, n_points]))
    229    224.9 MiB      0.0 MiB           1       posterior.sample(sample_shape=torch.Size([b, n_points]))
    """
    train_X, train_Y = _get_train_x_and_y(n_train, dim, train_batch_dim)

    model = SingleTaskGP(train_X, train_Y)
    posterior = model.posterior(train_X[0, :, :])
    memory_profile(posterior.sample)(sample_shape=torch.Size([b, n_points]))
    posterior.sample(sample_shape=torch.Size([b, n_points]))
    posterior.sample(sample_shape=torch.Size([b, n_points]))
    posterior.sample(sample_shape=torch.Size([b, n_points]))
