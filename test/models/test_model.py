#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import InputDataError
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model, ModelDict, ModelList
from botorch.models.utils import parse_training_data
from botorch.posteriors.deterministic import DeterministicPosterior
from botorch.posteriors.posterior_list import PosteriorList
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class NotSoAbstractBaseModel(Model):
    def posterior(self, X, output_indices, observation_noise, **kwargs):
        pass


class GenericDeterministicModelWithBatchShape(GenericDeterministicModel):
    # mocking torch.nn.Module components is kind of funky, so let's do this instead
    @property
    def batch_shape(self):
        return self._batch_shape


class DummyPosteriorTransform(PosteriorTransform):
    def evaluate(self, Y):
        return 2 * Y + 1

    def forward(self, posterior):
        return PosteriorList(
            *[DeterministicPosterior(2 * p.mean + 1) for p in posterior.posteriors]
        )


class TestBaseModel(BotorchTestCase):
    def test_abstract_base_model(self):
        with self.assertRaises(TypeError):
            Model()

    def test_not_so_abstract_base_model(self):
        model = NotSoAbstractBaseModel()
        with self.assertRaises(NotImplementedError):
            model.condition_on_observations(None, None)
        with self.assertRaises(NotImplementedError):
            model.num_outputs
        with self.assertRaises(NotImplementedError):
            model.batch_shape
        with self.assertRaises(NotImplementedError):
            model.subset_output([0])

    def test_construct_inputs(self):
        with patch.object(
            parse_training_data, "parse_training_data", return_value={"a": 1}
        ):
            model = NotSoAbstractBaseModel()
            self.assertEqual(model.construct_inputs(None), {"a": 1})

    def test_model_list(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        m1 = GenericDeterministicModel(lambda X: X[-1:], num_outputs=1)
        m2 = GenericDeterministicModel(lambda X: X[-2:], num_outputs=2)
        model = ModelList(m1, m2)
        self.assertEqual(model.num_outputs, 3)
        # test _get_group_subset_indices
        gsi = model._get_group_subset_indices(idcs=None)
        self.assertEqual(len(gsi), 2)
        self.assertIsNone(gsi[0])
        self.assertIsNone(gsi[1])
        gsi = model._get_group_subset_indices(idcs=[0, 2])
        self.assertEqual(len(gsi), 2)
        self.assertEqual(gsi[0], [0])
        self.assertEqual(gsi[1], [1])
        # test subset_model
        m_sub = model.subset_output(idcs=[0, 1])
        self.assertIsInstance(m_sub, ModelList)
        self.assertEqual(m_sub.num_outputs, 2)
        m_sub = model.subset_output(idcs=[1, 2])
        self.assertIsInstance(m_sub, GenericDeterministicModel)
        self.assertEqual(m_sub.num_outputs, 2)
        # test posterior
        X = torch.rand(2, 2, **tkwargs)
        p = model.posterior(X=X)
        self.assertIsInstance(p, PosteriorList)
        # test batch shape
        m1 = GenericDeterministicModelWithBatchShape(lambda X: X[-1:], num_outputs=1)
        m2 = GenericDeterministicModelWithBatchShape(lambda X: X[-2:], num_outputs=2)
        model = ModelList(m1, m2)
        m1._batch_shape = torch.Size([2])
        m2._batch_shape = torch.Size([2])
        self.assertEqual(model.batch_shape, torch.Size([2]))
        m2._batch_shape = torch.Size([3])
        with self.assertRaisesRegex(
            NotImplementedError,
            "is only supported if all constituent models have the same `batch_shape`",
        ):
            model.batch_shape

    def test_posterior_transform(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        m1 = GenericDeterministicModel(
            lambda X: X.sum(dim=-1, keepdims=True), num_outputs=1
        )
        m2 = GenericDeterministicModel(
            lambda X: X.prod(dim=-1, keepdims=True), num_outputs=1
        )
        model = ModelList(m1, m2)
        X = torch.rand(5, 3, **tkwargs)
        posterior_tf = model.posterior(X, posterior_transform=DummyPosteriorTransform())
        self.assertTrue(
            torch.allclose(
                posterior_tf.mean, torch.cat((2 * m1(X) + 1, 2 * m2(X) + 1), dim=-1)
            )
        )


class TestModelDict(BotorchTestCase):
    def test_model_dict(self):
        models = {"m1": MockModel(MockPosterior()), "m2": MockModel(MockPosterior())}
        model_dict = ModelDict(**models)
        self.assertIs(model_dict["m1"], models["m1"])
        self.assertIs(model_dict["m2"], models["m2"])
        with self.assertRaisesRegex(
            InputDataError, "Expected all models to be a BoTorch `Model`."
        ):
            ModelDict(m=MockPosterior())
