#! /usr/bin/env python3

from abc import ABC
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
    ) -> GPyTorchPosterior:
        self.eval()  # pyre-ignore
        with gpytorch.fast_pred_var():
            posterior = GPyTorchPosterior(self(X))
        if observation_noise:
            posterior = self.add_observation_noise(
                posterior=posterior, X=X, output_indices=output_indices
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
    ) -> Posterior:
        with gpytorch.fast_pred_var():
            mvn = self.likelihood(posterior.mvn, X)  # pyre-ignore
            posterior = GPyTorchPosterior(mvn)
        if output_indices is not None:
            raise NotImplementedError
        return posterior
