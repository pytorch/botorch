#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
from botorch.utils.testing import BotorchTestCase
from botorch_community.models.vblls import VBLLModel
from botorch_community.posteriors.bll_posterior import BLLPosterior


def _reg_data_singletask(d, n=10):
    X = torch.randn(10, d, dtype=torch.float64)
    y = torch.randn(10, 1, dtype=torch.float64)
    return X, y


def _get_fast_training_settings():
    return {
        "num_epochs": 3,
        "lr": 0.01,
    }


class TestVBLLModel(BotorchTestCase):
    def test_initialization(self) -> None:
        d, num_hidden, num_outputs, num_layers = 2, 3, 1, 4
        model = VBLLModel(
            in_features=d,
            hidden_features=num_hidden,
            num_layers=num_layers,
            out_features=num_outputs,
        )
        self.assertEqual(model.num_inputs, d)
        self.assertEqual(model.num_outputs, num_outputs)

        hidden_layer_count = sum(
            isinstance(layer, torch.nn.Linear)
            for submodule in model.backbone[1:]  # note that the first layer is excluded
            for layer in (
                submodule if isinstance(submodule, torch.nn.Sequential) else [submodule]
            )
        )
        self.assertEqual(
            hidden_layer_count,
            num_layers,
            f"Expected {num_layers} hidden layers, but got {hidden_layer_count}.",
        )

    def test_backbone_initialization(self) -> None:
        d, num_hidden = 4, 3
        test_backbone = torch.nn.Sequential(
            torch.nn.Linear(d, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden),
        )
        model = VBLLModel(backbone=test_backbone, hidden_features=num_hidden)

        for key in test_backbone.state_dict():
            self.assertTrue(
                torch.allclose(
                    test_backbone.state_dict()[key],
                    model.backbone.state_dict()[key],
                    atol=1e-6,
                ),
                f"Mismatch of backbone state_dict for key: {key}",
            )

    def test_freezing_backbone(self) -> None:
        d, num_hidden = 4, 3
        for freeze_backbone in (True, False):
            test_backbone = torch.nn.Sequential(
                torch.nn.Linear(d, num_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(num_hidden, num_hidden),
                torch.nn.ELU(),
            ).to(dtype=torch.float64)

            model = VBLLModel(
                backbone=copy.deepcopy(test_backbone),
                hidden_features=num_hidden,  # match the output of the backbone
            )

            X, y = _reg_data_singletask(d)
            optim_settings = {
                "num_epochs": 1,
                "lr": 10.0,  # large lr to make sure that the weights change
                "freeze_backbone": freeze_backbone,
            }
            model.fit(X, y, optimization_settings=optim_settings)

            if freeze_backbone:
                # Ensure all parameters remain unchanged
                model_bb_sdict = model.backbone.state_dict()
                for key, val in test_backbone.state_dict().items():
                    self.assertTrue(
                        torch.allclose(val, model_bb_sdict[key], atol=1e-6),
                        f"Parameter {key} changed with freeze_backbone=True",
                    )
            else:
                # Ensure at least one parameter has changed
                model_bb_sdict = model.backbone.state_dict()
                changed = False
                for key, val in test_backbone.state_dict().items():
                    if not torch.allclose(val, model_bb_sdict[key], atol=1e-6):
                        changed = True
                        break
                self.assertTrue(
                    changed,
                    "Expected at least one parameter to change, but all remained the same with freeze_backbone=False",
                )

    def test_update_of_reg_weight(self) -> None:
        kl_scale = 2.0
        d = 2
        model = VBLLModel(
            in_features=d,
            hidden_features=3,
            out_features=1,
            num_layers=1,
            kl_scale=kl_scale,
        )
        self.assertEqual(
            model.model.head.regularization_weight,
            1.0,
            "Regularization weight should be 1.0 after init.",
        )

        X, y = _reg_data_singletask(d)

        optim_settings = _get_fast_training_settings()
        model.fit(X, y, optimization_settings=optim_settings)

        self.assertEqual(
            model.model.head.regularization_weight,
            kl_scale / len(y),
            f"Regularization weight should be {kl_scale}/{len(y)}, but got {model.model.head.regularization_weight}.",
        )

    def test_shape_of_sampling(self) -> None:
        d = 4

        for out_features in (1, 2):
            model = VBLLModel(
                in_features=d,
                hidden_features=4,
                out_features=out_features,
                num_layers=1,
            )
            n_test_points = 3
            X = torch.rand(n_test_points, d, dtype=torch.float64)
            for sample_shape in (torch.Size(), torch.Size([2])):
                sample_path = model.sample(sample_shape=sample_shape)
                y_hat = sample_path(X)
                expected_shape = sample_shape + torch.Size(
                    [n_test_points, out_features]
                )
                self.assertEqual(
                    y_hat.shape,
                    expected_shape,
                    f"Expected samples to have shape {expected_shape}, but got {y_hat.shape}.",
                )

    def test_shape_of_forward(self) -> None:
        d = 4
        for out_features in (1, 2):
            model = VBLLModel(
                in_features=d,
                hidden_features=4,
                out_features=out_features,
                num_layers=1,
            )
            n_test_points = 3
            X = torch.rand(n_test_points, d, dtype=torch.float64)
            y_pred = model(X)

            y_hat = y_pred.mean
            y_var = y_pred.variance

            expected_shape = torch.Size([n_test_points, out_features])
            self.assertEqual(
                y_hat.shape,
                expected_shape,
                f"Expected forward pass to have shape {expected_shape}, but got {y_hat.shape}.",
            )
            self.assertEqual(
                y_var.shape,
                expected_shape,
                f"Expected forward pass to have shape {expected_shape}, but got {y_var.shape}.",
            )

    def test_shape_of_predictions(self) -> None:
        d = 4
        model = VBLLModel(
            in_features=d, hidden_features=4, out_features=1, num_layers=1
        )
        X, y = _reg_data_singletask(d)
        optim_settings = _get_fast_training_settings()

        model.fit(X, y, optimization_settings=optim_settings)

        for batch_shape in (torch.Size([2]), torch.Size()):
            X = torch.rand(batch_shape + torch.Size([3, d]), dtype=torch.float64)
            expected_shape = batch_shape + torch.Size([3, 1])

            post = model.posterior(X)

            # check that the posterior is an instance of BLLPosterior
            self.assertIsInstance(
                post,
                BLLPosterior,
                "Expected posterior to be an instance of BLLPosterior.",
            )

            # mean prediction
            self.assertEqual(
                post.mean.shape,
                expected_shape,
                f"Expected mean predictions to have shape {expected_shape}, but got {post.mean.shape}.",
            )

            # variance prediction
            self.assertEqual(
                post.variance.shape,
                expected_shape,
                f"Expected variance predictions to have shape {expected_shape}, but got {post.mean.shape}.",
            )
