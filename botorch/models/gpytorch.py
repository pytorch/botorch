#! /usr/bin/env python3

"""
Abstract class for all GPyTorch-based botorch models.

To implement your own, simply inherit from both the provided classes and a
GPyTorch Model class such as an ExactGP.
"""


from abc import ABC
from contextlib import ExitStack
from typing import List, Optional

import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal
from torch import Tensor

from ..posteriors.gpytorch import GPyTorchPosterior
from .model import Model


class GPyTorchModel(Model, ABC):
    """Abstract base class for models based on GPyTorch models.

    The easiest way to use this is to subclass a model from a GPyTorch model
    class (e.g. an `ExactGP`) and this `GPyTorchModel`. See e.g. `SingleTaskGP`.
    """

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        detach_test_caches: bool = True,
    ) -> GPyTorchPosterior:
        """Computes the posterior over model outputs at the provided points.

        Args:
            X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
                feature space, `q` is the number of points considered jointly,
                and `b` is the batch dimension.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add observation noise to the posterior.
            detach_test_caches: If True, detach GPyTorch test caches during
                computation of the posterior. Required for being able to compute
                derivatives with respect to training inputs at test time (used
                e.g. by qNoisyExpectedImprovement).

        Returns:
            A `Posterior` object, representing a batch of `b` joint distributions
                over `q` points and the outputs selected by `output_indices` each.
                Includes measurement noise if `observation_noise=True`.
        """
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
    """Abstract base class for models based on multi-output GPyTorch models."""

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        detach_test_caches: bool = True,
    ) -> GPyTorchPosterior:
        """Computes the posterior over model outputs at the provided points.

        Args:
            X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
                feature space, `q` is the number of points considered jointly,
                and `b` is the batch dimension.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add observation noise to the posterior.
            detach_test_caches: If True, detach GPyTorch test caches during
                computation of the posterior. Required for being able to compute
                derivatives with respect to training inputs at test time (used
                e.g. by qNoisyExpectedImprovement).

        Returns:
            A `Posterior` object, representing a batch of `b` joint distributions
                over `q` points and the outputs selected by `output_indices` each.
                Includes measurement noise if `observation_noise=True`.
        """
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
        if len(mvns) == 1:
            mvn = mvns[0]
        else:
            mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
        return GPyTorchPosterior(mvn=mvn)


class MultiTaskGPyTorchModel(MultiOutputGPyTorchModel, ABC):
    """Abstract base class for models based on multi-task GPyTorch models."""

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        detach_test_caches: bool = True,
    ) -> GPyTorchPosterior:
        """Computes the posterior over model outputs at the provided points.

        Args:
            X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
                feature space, `q` is the number of points considered jointly,
                and `b` is the batch dimension.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add observation noise to the posterior.
            detach_test_caches: If True, detach GPyTorch test caches during
                computation of the posterior. Required for being able to compute
                derivatives with respect to training inputs at test time (used
                e.g. by qNoisyExpectedImprovement).

        Returns:
            A `Posterior` object, representing a batch of `b` joint distributions
                over `q` points and the outputs selected by `output_indices` each.
                Includes measurement noise if `observation_noise=True`.
        """
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
