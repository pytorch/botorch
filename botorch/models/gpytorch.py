#! /usr/bin/env python3

from abc import ABC
from contextlib import ExitStack
from typing import List, Optional

import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal
from torch import Tensor

from ..posteriors.gpytorch import GPyTorchPosterior
from .model import Model


class GPyTorchModel(Model, ABC):
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        detach_test_caches: bool = True,
    ) -> GPyTorchPosterior:
        if output_indices is not None and output_indices != [0]:
            raise RuntimeError(
                "Cannot pass more than one output index to single-output model"
            )
        self.eval()  # pyre-ignore
        with ExitStack() as es:
            es.enter_context(gpytorch.settings.debug(False))
            es.enter_context(gpytorch.settings.fast_pred_var())
            es.enter_context(gpytorch.settings.detach_test_caches(detach_test_caches))
            mvn = self(X)
            if observation_noise:
                mvn = self.likelihood(mvn, X)
        return GPyTorchPosterior(mvn=mvn)


class MultiOutputGPyTorchModel(GPyTorchModel, ABC):
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
            if output_indices is not None:
                mvns = [self.forward_i(i, X) for i in output_indices]
                if observation_noise:
                    mvns = [
                        self.likelihood_i(i, mvn, X)
                        for i, mvn in zip(output_indices, mvns)
                    ]
            else:
                mvns = self(*[X for _ in range(self.num_outputs)])
                if observation_noise:
                    mvns = self.likelihood(*[(mvn, X) for mvn in mvns])
        mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
        return GPyTorchPosterior(mvn=mvn)


class MultiTaskGPyTorchModel(MultiOutputGPyTorchModel, ABC):
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
            if output_indices is not None:
                # we just need to properly subset the covariance matrix here
                raise NotImplementedError
            mvn = self(X)
            if observation_noise:
                mvn = self.likelihood(mvn, X)
        return GPyTorchPosterior(mvn=mvn)
