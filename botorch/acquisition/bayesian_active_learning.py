# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition functions for Bayesian active learning. This includes:
BALD [Houlsby2011bald]_ and its batch version [kirsch2019batchbald]_.

References

.. [kirsch2019batchbald]
    Andreas Kirsch, Joost van Amersfoort, Yarin Gal.
    BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian
    Active Learning.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2019.

"""

from __future__ import annotations

from typing import Optional

from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.fully_bayesian import MCMC_DIM, SaasFullyBayesianSingleTaskGP
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor


class FullyBayesianAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model: Model):
        """Base class for acquisition functions which require a Fully Bayesian
        model treatment.

        Args:
            model: A fully bayesian single-outcome model.
        """
        if model._is_fully_bayesian:
            super().__init__(model)

        else:
            raise ValueError(
                "Fully Bayesian acquisition functions require "
                "a SaasFullyBayesianSingleTaskGP to run."
            )


class qBayesianActiveLearningByDisagreement(
    FullyBayesianAcquisitionFunction, MCSamplerMixin
):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        sampler: Optional[MCSampler] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        """
        Batch implementation [kirsch2019batchbald]_ of BALD [Houlsby2011bald]_,
        which maximizes the mutual information between the next observation and the
        hyperparameters of the model. Computed by Monte Carlo integration.

        Args:
            model: A fully bayesian model (SaasFullyBayesianSingleTaskGP).
            sampler: The sampler used for drawing samples to approximate the entropy
                of the Gaussian Mixture posterior.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points

        """
        super().__init__(model=model)
        MCSamplerMixin.__init__(self, sampler=sampler)
        self.set_X_pending(X_pending)
        self.posterior_transform = posterior_transform

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qBayesianActiveLearningByDisagreement on the candidate set `X`.
        A monte carlo-estimated information gain is computed over a Gaussian Mixture
        marginal posterior, and the Gaussian conditional posterior to obtain the
        qBayesianActiveLearningByDisagreement on the candidate set `X`.

        Args:
            X: `batch_shape x q x D`-dim Tensor of input points.

        Returns:
            A `batch_shape x num_models`-dim Tensor of BALD values.
        """
        posterior = self.model.posterior(
            X, observation_noise=True, posterior_transform=self.posterior_transform
        )
        # draw samples from the mixture posterior.
        # samples: num_samples x batch_shape x num_models x q x num_outputs
        samples = self.get_posterior_samples(posterior=posterior)

        # Estimate the entropy of 'num_samples' samples from 'num_models' models by
        # evaluating the log_prob on each sample on the mixture posterior
        # (which constitutes of M models). thus, order N*M^2 computations

        # Make room and move the model dim to the front, squeeze the num_outputs dim.
        # prev_samples: num_models x num_samples x batch_shape x 1 x q
        prev_samples = samples.unsqueeze(0).transpose(0, MCMC_DIM).squeeze(-1)

        # avg the probs over models in the mixture - dim (-2) will be broadcasted
        # with the num_models of the posterior --> querying all samples on all models
        # posterior.mvn takes q-dimensional input by default, which removes the q-dim
        # component_sample_probs: num_models x num_samples x batch_shape x num_models
        component_sample_probs = posterior.mvn.log_prob(prev_samples).exp()

        # average over mixture components
        mixture_sample_probs = component_sample_probs.mean(dim=-1)

        # this is the average over the model and sample dim
        prev_entropy = -mixture_sample_probs.log().mean(dim=[0, 1])

        # the posterior entropy is an average entropy over gaussians, so no mixture
        post_entropy = -posterior.mvn.log_prob(samples.squeeze(-1)).mean(0)
        bald = prev_entropy.unsqueeze(-1) - post_entropy
        return bald
