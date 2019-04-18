#! /usr/bin/env python3

r"""
Model List GP Regression models.
"""

from typing import Any, List, Optional

from gpytorch.models import IndependentModelList
from torch import Tensor

from .gpytorch import ModelListGPyTorchModel
from .model import Model


class ModelListGP(IndependentModelList, ModelListGPyTorchModel):
    r"""A multi-output GP model with independent GPs for the outputs.

    This model supports different-shaped training inputs for each of its
    sub-models. It can be used with any botorch models.

    Internally, this model is just a list of individual models, but it implements
    the same input/output interface as all other botorch models. This makes it
    very flexible and convenient to work with. The sequential evaluation comes
    at a performance cost though - if you are using a block design (i.e. the
    same number of training example for each output, and a similar model
    structure, you should consider using a batched GP model instead).
    """

    def __init__(self, gp_models: List[Model]) -> None:
        r"""A multi-output GP model with independent GPs for the outputs.

        Args:
            gp_models: A list of single-output botorch models.

        Example:
            >>> model1 = SingleTaskGP(train_X1, train_Y1)
            >>> model2 = SingleTaskGP(train_X2, train_Y2)
            >>> model = ModelListGP([model1, model2])
        """
        super().__init__(*gp_models)

    def reinitialize(
        self,
        train_Xs: List[Tensor],
        train_Ys: List[Tensor],
        train_Yvars: Optional[List[Tensor]] = None,
        keep_params: bool = True,
        **kwargs: Any,
    ) -> None:
        r"""Reinitialize model and likelihood given new data.

        Args:
            train_Xs: A list of tensors of new training data.
            train_Ys: A list of tensors of new training observation.
            train_Yvars: A list of tensors of new training noise observations.
            keep_params: If True, keep the model's hyperparameter values (speeds
                up refitting on similar data).

        This does not refit the individual underlying models. If device/dtype
        of the new training data for a sub-model are different from that of the
        sub-model, then that model is moved to the new device/dtype.

        Example:
            >>> new_trainXs = [torch.rand(10, 2), torch.rand(20, 2)]
            >>> new_trainYs = [f1(new_train_Xs[0]), f2(new_train_Xs[1])]
            >>> model.reinitialize(new_trainXs, new_trainYs)
        """
        if train_Yvars is None:
            train_Yvars = [None for _ in range(len(train_Xs))]
        for model, train_X, train_Y, train_Yvar in zip(
            self.models, train_Xs, train_Ys, train_Yvars
        ):
            model.reinitialize(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                keep_params=keep_params,
            )
