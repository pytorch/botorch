#! /usr/bin/env python3

r"""
Model List GP Regression models.
"""

from typing import List

from gpytorch.models import IndependentModelList

from .gpytorch import GPyTorchModel, ModelListGPyTorchModel


class ModelListGP(IndependentModelList, ModelListGPyTorchModel):
    r"""A multi-output GP model with independent GPs for the outputs.

    This model supports different-shaped training inputs for each of its
    sub-models. It can be used with any BoTorch models.

    Internally, this model is just a list of individual models, but it implements
    the same input/output interface as all other BoTorch models. This makes it
    very flexible and convenient to work with. The sequential evaluation comes
    at a performance cost though - if you are using a block design (i.e. the
    same number of training example for each output, and a similar model
    structure, you should consider using a batched GP model instead).
    """

    def __init__(self, gp_models: List[GPyTorchModel]) -> None:
        r"""A multi-output GP model with independent GPs for the outputs.

        Args:
            gp_models: A list of single-output BoTorch models.

        Example:
            >>> model1 = SingleTaskGP(train_X1, train_Y1)
            >>> model2 = SingleTaskGP(train_X2, train_Y2)
            >>> model = ModelListGP([model1, model2])
        """
        super().__init__(*gp_models)
