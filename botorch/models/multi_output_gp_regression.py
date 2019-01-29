#! /usr/bin/env python3

from typing import List, Optional

from gpytorch.models import IndependentModelList
from torch import Tensor

from .gpytorch import MultiOutputGPyTorchModel
from .model import Model


class MultiOutputGP(IndependentModelList, MultiOutputGPyTorchModel):
    def __init__(self, gp_models: List[Model]) -> None:
        super().__init__(*gp_models)

    def reinitialize(
        self,
        train_Xs: List[Tensor],
        train_Ys: List[Tensor],
        train_Y_ses: Optional[List[Tensor]] = None,
        keep_params: bool = True,
    ) -> None:
        """Reinitialize model and likelihood.

        Args:
            train_Xs: A list of tensors of new training data
            train_Ys: A list of tensors of new training observations
            train_Y_ses: A list of tensors of new training noise observations
            keep_params: If True, keep the parameter values (speeds up refitting
                on similar data)

        Note: this does not refit the model.
        """
        if train_Y_ses is None:
            train_Y_ses = [None for _ in range(len(train_Xs))]
        for model, train_X, train_Y, train_Y_se in zip(
            self.models, train_Xs, train_Ys, train_Y_ses
        ):
            model.reinitialize(
                train_X=train_X,
                train_Y=train_Y,
                train_Y_se=train_Y_se,
                keep_params=keep_params,
            )
