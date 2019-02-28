#! /usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Optional

from torch import Tensor

from ..posteriors import Posterior


class Model(ABC):
    """Abstract base class for botorch models."""

    @abstractmethod
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        **kwargs
    ) -> Posterior:
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

        Returns:
            A `Posterior` object, representing a batch of `b` joint distributions
                over `q` points and `t` outputs each.
        """
        pass

    def reinitialize(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Y_se: Optional[Tensor] = None,
        keep_params: bool = True,
    ) -> None:
        """Re-initializes a model given new data

        Args:
            train_X: A `n x d`-dim Tensor containing the new training inputs.
            train_Y: A `n x t`-dim Tensor containing the new training outputs.
            train_Y_se: An `n x t`-dim Tensor containing the observed measurement
                noise at the training outputs.
            keep_params: If True, do not reset the model hyperparameters.
        """
        raise NotImplementedError
