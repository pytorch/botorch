#! /usr/bin/env python3

from abc import ABC
from contextlib import ExitStack
from typing import List, Optional

import gpytorch
from torch import Tensor

from ..posteriors.gpytorch import GPyTorchPosterior
from ..posteriors.posterior import Posterior
from .model import Model


class GPyTorchModel(Model, ABC):
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        detach_test_caches: bool = True,
    ) -> GPyTorchPosterior:
        self.eval()  # pyre-ignore
        with ExitStack() as es:
            es.enter_context(gpytorch.settings.debug(False))
            es.enter_context(gpytorch.settings.fast_pred_var())
            es.enter_context(gpytorch.settings.detach_test_caches(detach_test_caches))
            posterior = GPyTorchPosterior(self(X))
        if observation_noise:
            posterior = self.add_observation_noise(
                posterior=posterior,
                X=X,
                output_indices=output_indices,
                detach_test_caches=detach_test_caches,
            )
        elif output_indices is not None:
            # we just need to properly subset the covariance matrix here
            raise NotImplementedError
        return posterior

    def add_observation_noise(
        self,
        posterior: Posterior,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        detach_test_caches: bool = True,
    ) -> Posterior:
        with ExitStack() as es:
            es.enter_context(gpytorch.settings.debug(False))
            es.enter_context(gpytorch.settings.fast_pred_var())
            es.enter_context(gpytorch.settings.detach_test_caches(detach_test_caches))
            mvn = self.likelihood(posterior.mvn, X)  # pyre-ignore
            posterior = GPyTorchPosterior(mvn)
        if output_indices is not None:
            raise NotImplementedError
        return posterior
