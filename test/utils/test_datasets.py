#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from botorch.exceptions.errors import InputDataError, UnsupportedError
from botorch.utils.containers import DenseContainer, SliceContainer
from botorch.utils.datasets import (
    ContextualDataset,
    FixedNoiseDataset,
    MultiTaskDataset,
    RankingDataset,
    SupervisedDataset,
)
from botorch.utils.testing import BotorchTestCase
from torch import rand, randperm, Size, stack, Tensor, tensor


def make_dataset(
    num_samples: int = 3,
    d: int = 2,
    m: int = 1,
    has_yvar: bool = False,
    feature_names: Optional[List[str]] = None,
    outcome_names: Optional[List[str]] = None,
    batch_shape: Optional[torch.Size] = None,
) -> SupervisedDataset:
    feature_names = feature_names or [f"x{i}" for i in range(d)]
    outcome_names = outcome_names or [f"y{i}" for i in range(m)]
    batch_shape = batch_shape or torch.Size()
    return SupervisedDataset(
        X=rand(*batch_shape, num_samples, d),
        Y=rand(*batch_shape, num_samples, m),
        Yvar=rand(*batch_shape, num_samples, m) if has_yvar else None,
        feature_names=feature_names,
        outcome_names=outcome_names,
    )


class TestDatasets(BotorchTestCase):
    def test_supervised(self):
        # Generate some data
        X = rand(3, 2)
        Y = rand(3, 1)
        feature_names = ["x1", "x2"]
        outcome_names = ["y"]

        # Test `__init__`
        dataset = SupervisedDataset(
            X=X, Y=Y, feature_names=feature_names, outcome_names=outcome_names
        )
        self.assertIsInstance(dataset.X, Tensor)
        self.assertIsInstance(dataset._X, Tensor)
        self.assertIsInstance(dataset.Y, Tensor)
        self.assertIsInstance(dataset._Y, Tensor)
        self.assertEqual(dataset.feature_names, feature_names)
        self.assertEqual(dataset.outcome_names, outcome_names)

        dataset2 = SupervisedDataset(
            X=DenseContainer(X, X.shape[-1:]),
            Y=DenseContainer(Y, Y.shape[-1:]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )
        self.assertIsInstance(dataset2.X, Tensor)
        self.assertIsInstance(dataset2._X, DenseContainer)
        self.assertIsInstance(dataset2.Y, Tensor)
        self.assertIsInstance(dataset2._Y, DenseContainer)
        self.assertEqual(dataset, dataset2)

        # Test `_validate`
        with self.assertRaisesRegex(ValueError, "Batch dimensions .* incompatible."):
            SupervisedDataset(
                X=rand(1, 2),
                Y=rand(2, 1),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        with self.assertRaisesRegex(ValueError, "`Y` and `Yvar`"):
            SupervisedDataset(
                X=rand(2, 2),
                Y=rand(2, 1),
                Yvar=rand(2),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        with self.assertRaisesRegex(ValueError, "feature_names"):
            SupervisedDataset(
                X=rand(2, 2),
                Y=rand(2, 1),
                feature_names=[],
                outcome_names=outcome_names,
            )
        with self.assertRaisesRegex(ValueError, "outcome_names"):
            SupervisedDataset(
                X=rand(2, 2),
                Y=rand(2, 1),
                feature_names=feature_names,
                outcome_names=[],
            )

        # Test with Yvar.
        dataset = SupervisedDataset(
            X=X,
            Y=Y,
            Yvar=DenseContainer(Y, Y.shape[-1:]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )
        self.assertIsInstance(dataset.X, Tensor)
        self.assertIsInstance(dataset._X, Tensor)
        self.assertIsInstance(dataset.Y, Tensor)
        self.assertIsInstance(dataset._Y, Tensor)
        self.assertIsInstance(dataset.Yvar, Tensor)
        self.assertIsInstance(dataset._Yvar, DenseContainer)

        # More equality checks with & without Yvar.
        self.assertEqual(dataset, dataset)
        self.assertNotEqual(dataset, dataset2)
        self.assertNotEqual(dataset2, dataset)

    def test_fixedNoise(self):
        # Generate some data
        X = rand(3, 2)
        Y = rand(3, 1)
        Yvar = rand(3, 1)
        feature_names = ["x1", "x2"]
        outcome_names = ["y"]
        dataset = FixedNoiseDataset(
            X=X,
            Y=Y,
            Yvar=Yvar,
            feature_names=feature_names,
            outcome_names=outcome_names,
        )
        self.assertTrue(torch.equal(dataset.X, X))
        self.assertTrue(torch.equal(dataset.Y, Y))
        self.assertTrue(torch.equal(dataset.Yvar, Yvar))
        self.assertEqual(dataset.feature_names, feature_names)
        self.assertEqual(dataset.outcome_names, outcome_names)

        with self.assertRaisesRegex(
            ValueError, "`Y` and `Yvar`"
        ), self.assertWarnsRegex(DeprecationWarning, "SupervisedDataset"):
            FixedNoiseDataset(
                X=X,
                Y=Y,
                Yvar=Yvar.squeeze(),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )

    def test_ranking(self):
        # Test `_validate`
        X_val = rand(16, 2)
        X_idx = stack([randperm(len(X_val))[:3] for _ in range(1)])
        X = SliceContainer(X_val, X_idx, event_shape=Size([3 * X_val.shape[-1]]))
        feature_names = ["x1", "x2"]
        outcome_names = ["ranking indices"]

        with self.assertRaisesRegex(ValueError, "The `values` field of `X`"):
            RankingDataset(
                X=X,
                Y=tensor([[-1, 0, 1]]),
                feature_names=feature_names[:1],
                outcome_names=outcome_names,
            )

        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            RankingDataset(
                X=X,
                Y=tensor([[-1, 0, 1]]),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        RankingDataset(
            X=X,
            Y=tensor([[2, 0, 1]]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )

        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            RankingDataset(
                X=X,
                Y=tensor([[0, 1, 3]]),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        RankingDataset(
            X=X,
            Y=tensor([[0, 1, 2]]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )

        with self.assertRaisesRegex(ValueError, "missing zero-th rank."):
            RankingDataset(
                X=X,
                Y=tensor([[1, 2, 2]]),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        RankingDataset(
            X=X,
            Y=tensor([[0, 1, 1]]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )

        with self.assertRaisesRegex(ValueError, "ranks not skipped after ties."):
            RankingDataset(
                X=X,
                Y=tensor([[0, 0, 1]]),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        RankingDataset(
            X=X,
            Y=tensor([[0, 0, 2]]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )

    def test_multi_task(self):
        dataset_1 = make_dataset(outcome_names=["y"])
        dataset_2 = make_dataset(outcome_names=["z"])
        dataset_3 = make_dataset(has_yvar=True, outcome_names=["z"])
        dataset_4 = make_dataset(has_yvar=True, outcome_names=["y"])
        # Test validation.
        with self.assertRaisesRegex(
            UnsupportedError, "containing more than one outcome"
        ):
            MultiTaskDataset(datasets=[make_dataset(m=2)], target_outcome_name="y0")
        with self.assertRaisesRegex(
            UnsupportedError, "multiple datasets for the same outcome"
        ):
            MultiTaskDataset(datasets=[dataset_1, dataset_1], target_outcome_name="y")
        with self.assertRaisesRegex(InputDataError, "Target outcome is not present"):
            MultiTaskDataset(datasets=[dataset_1], target_outcome_name="z")
        with self.assertRaisesRegex(UnsupportedError, "modeling batched inputs"):
            MultiTaskDataset(
                datasets=[make_dataset(batch_shape=torch.Size([2]))],
                target_outcome_name="y0",
            )
        with self.assertRaisesRegex(InputDataError, "names of the task features"):
            MultiTaskDataset(
                datasets=[
                    dataset_1,
                    make_dataset(feature_names=["x1", "x3"], outcome_names=["z"]),
                ],
                target_outcome_name="z",
                task_feature_index=1,
            )
        with self.assertRaisesRegex(
            UnsupportedError, "all or none of the datasets to have a Yvar."
        ):
            MultiTaskDataset(datasets=[dataset_1, dataset_3], target_outcome_name="z")

        # Test correct construction.
        mt_dataset = MultiTaskDataset(
            datasets=[dataset_1, dataset_2],
            target_outcome_name="z",
        )
        self.assertEqual(len(mt_dataset.datasets), 2)
        self.assertIsNone(mt_dataset.task_feature_index)
        self.assertIs(mt_dataset.datasets["y"], dataset_1)
        self.assertIs(mt_dataset.datasets["z"], dataset_2)
        self.assertIsNone(mt_dataset.Yvar)
        self.assertFalse(mt_dataset.has_heterogeneous_features)
        expected_X = torch.cat(
            [
                torch.cat([dataset_1.X, torch.ones(3, 1)], dim=-1),
                torch.cat([dataset_2.X, torch.zeros(3, 1)], dim=-1),
            ],
            dim=0,
        )
        expected_Y = torch.cat([ds.Y for ds in [dataset_1, dataset_2]], dim=0)
        self.assertTrue(torch.equal(expected_X, mt_dataset.X))
        self.assertTrue(torch.equal(expected_Y, mt_dataset.Y))
        self.assertIs(
            mt_dataset.get_dataset_without_task_feature(outcome_name="y"), dataset_1
        )

        # Test with Yvar and target_feature_index.
        mt_dataset = MultiTaskDataset(
            datasets=[dataset_3, dataset_4],
            target_outcome_name="z",
            task_feature_index=1,
        )
        self.assertEqual(mt_dataset.task_feature_index, 1)
        expected_X_2 = torch.cat([dataset_3.X, dataset_4.X], dim=0)
        expected_Yvar_2 = torch.cat([dataset_3.Yvar, dataset_4.Yvar], dim=0)
        self.assertTrue(torch.equal(expected_X_2, mt_dataset.X))
        self.assertTrue(torch.equal(expected_Yvar_2, mt_dataset.Yvar))
        # Check that the task feature is removed correctly.
        ds_3_no_task = mt_dataset.get_dataset_without_task_feature(outcome_name="z")
        self.assertTrue(torch.equal(ds_3_no_task.X, dataset_3.X[:, :1]))
        self.assertTrue(torch.equal(ds_3_no_task.Y, dataset_3.Y))
        self.assertTrue(torch.equal(ds_3_no_task.Yvar, dataset_3.Yvar))
        self.assertEqual(ds_3_no_task.feature_names, dataset_3.feature_names[:1])
        self.assertEqual(ds_3_no_task.outcome_names, dataset_3.outcome_names)

        # Test from_joint_dataset.
        sort_idcs = [3, 4, 5, 0, 1, 2]  # X & Y will get sorted based on task feature.
        for outcome_names_per_task in [None, {0: "x", 1: "y"}]:
            joint_dataset = SupervisedDataset(
                X=expected_X,
                Y=expected_Y,
                feature_names=["x0", "x1", "task"],
                outcome_names=["z"],
            )
            mt_dataset = MultiTaskDataset.from_joint_dataset(
                dataset=joint_dataset,
                task_feature_index=-1,
                target_task_value=0,
                outcome_names_per_task=outcome_names_per_task,
            )
            self.assertEqual(len(mt_dataset.datasets), 2)
            if outcome_names_per_task is None:
                self.assertEqual(list(mt_dataset.datasets.keys()), ["z", "task_1"])
                self.assertEqual(mt_dataset.target_outcome_name, "z")
            else:
                self.assertEqual(list(mt_dataset.datasets.keys()), ["x", "y"])
                self.assertEqual(mt_dataset.target_outcome_name, "x")

            self.assertTrue(torch.equal(mt_dataset.X, expected_X[sort_idcs]))
            self.assertTrue(torch.equal(mt_dataset.Y, expected_Y[sort_idcs]))
            self.assertTrue(
                torch.equal(
                    mt_dataset.datasets[mt_dataset.target_outcome_name].Y, dataset_2.Y
                )
            )
            self.assertIsNone(mt_dataset.Yvar)
        with self.assertRaisesRegex(UnsupportedError, "more than one outcome"):
            MultiTaskDataset.from_joint_dataset(
                dataset=make_dataset(m=2),
                task_feature_index=-1,
                target_task_value=0,
            )

        # With heterogeneous feature sets.
        dataset_5 = make_dataset(d=1, outcome_names=["z"])
        mt_dataset = MultiTaskDataset(
            datasets=[dataset_1, dataset_5],
            target_outcome_name="y",
        )
        self.assertTrue(mt_dataset.has_heterogeneous_features)
        with self.assertRaisesRegex(
            UnsupportedError, "datasets with heterogeneous feature sets"
        ):
            mt_dataset.X

    def test_contextual_datasets(self):
        num_contexts = 3
        feature_names = [f"x_c{i}" for i in range(num_contexts)]
        parameter_decomposition = {
            f"context_{i}": [f"x_c{i}"] for i in range(num_contexts)
        }
        context_buckets = list(parameter_decomposition.keys())
        context_outcome_list = [f"y:context_{i}" for i in range(num_contexts)]
        metric_decomposition = {f"{c}": [f"y:{c}"] for c in context_buckets}

        # test construction of agg outcome
        dataset_list1 = [
            make_dataset(
                d=1 * num_contexts,
                has_yvar=True,
                feature_names=feature_names,
                outcome_names=["y"],
            )
        ]
        context_dt = ContextualDataset(
            datasets=dataset_list1,
            parameter_decomposition=parameter_decomposition,
            context_buckets=context_buckets,
        )
        self.assertEqual(len(context_dt.datasets), len(dataset_list1))
        self.assertListEqual(context_dt.context_buckets, context_buckets)
        self.assertListEqual(context_dt.outcome_names, ["y"])
        self.assertListEqual(context_dt.feature_names, feature_names)
        self.assertIs(context_dt.datasets["y"], dataset_list1[0])
        self.assertIs(context_dt.X, dataset_list1[0].X)
        self.assertIs(context_dt.Y, dataset_list1[0].Y)
        self.assertIs(context_dt.Yvar, dataset_list1[0].Yvar)

        # test construction of context outcome
        dataset_list2 = [
            make_dataset(
                d=1 * num_contexts,
                has_yvar=True,
                feature_names=feature_names,
                outcome_names=[context_outcome_list[0]],
            )
        ]
        for m in context_outcome_list[1:]:
            dataset_list2.append(
                SupervisedDataset(
                    X=dataset_list2[0].X,
                    Y=rand(dataset_list2[0].Y.size()),
                    Yvar=rand(dataset_list2[0].Yvar.size()),
                    feature_names=feature_names,
                    outcome_names=[m],
                )
            )
        context_dt = ContextualDataset(
            datasets=dataset_list2,
            parameter_decomposition=parameter_decomposition,
            context_buckets=context_buckets,
            metric_decomposition=metric_decomposition,
        )
        self.assertEqual(len(context_dt.datasets), len(dataset_list2))
        self.assertListEqual(context_dt.context_buckets, context_buckets)
        self.assertListEqual(context_dt.outcome_names, context_outcome_list)
        self.assertListEqual(context_dt.feature_names, feature_names)
        self.assertTrue(torch.equal(context_dt.X, dataset_list2[-1].X))
        self.assertEqual(context_dt.Y.shape[-1], len(context_outcome_list))
        self.assertEqual(context_dt.Yvar.shape[-1], len(context_outcome_list))
        for dt in dataset_list2:
            self.assertIs(context_dt.datasets[dt.outcome_names[0]], dt)

        # test the ordering via context buckets
        context_dt_reverse = ContextualDataset(
            datasets=dataset_list2,
            parameter_decomposition=parameter_decomposition,
            context_buckets=context_buckets[::-1],  # reverse order
            metric_decomposition=metric_decomposition,
        )
        self.assertListEqual(
            context_dt_reverse.outcome_names, context_outcome_list[::-1]
        )
        self.assertTrue(
            torch.equal(context_dt.Y, torch.flip(context_dt_reverse.Y, (1,)))
        )
        self.assertTrue(
            torch.equal(context_dt.Yvar, torch.flip(context_dt_reverse.Yvar, (1,)))
        )

        # Test handling None Yvar
        dataset_list3 = [
            make_dataset(
                d=1 * num_contexts,
                has_yvar=False,
                feature_names=feature_names,
                outcome_names=[context_outcome_list[0]],
            )
        ]
        for m in context_outcome_list[1:]:
            dataset_list3.append(
                SupervisedDataset(
                    X=dataset_list3[0].X,
                    Y=rand(dataset_list3[0].Y.size()),
                    Yvar=None,
                    feature_names=feature_names,
                    outcome_names=[m],
                )
            )
        context_dt3 = ContextualDataset(
            datasets=dataset_list3,
            parameter_decomposition=parameter_decomposition,
            context_buckets=context_buckets,
            metric_decomposition=metric_decomposition,
        )
        self.assertIsNone(context_dt3.Yvar)

        # test dataset validation
        wrong_metric_decomposition = {
            f"{c}": [f"y:{c}"] for c in context_buckets if c != "context_0"
        }
        wrong_metric_decomposition["context_0"] = ["y:context_0", "y:context_1"]
        with self.assertRaisesRegex(
            ValueError, "context_0 bucket contains multiple outcomes"
        ):
            ContextualDataset(
                datasets=dataset_list2,
                parameter_decomposition=parameter_decomposition,
                context_buckets=context_buckets,
                metric_decomposition=wrong_metric_decomposition,
            )

        with self.assertRaisesRegex(
            InputDataError, "Require same X for context buckets"
        ):
            ContextualDataset(
                datasets=[
                    make_dataset(d=num_contexts, outcome_names=[m])
                    for m in context_outcome_list
                ],
                parameter_decomposition=parameter_decomposition,
                context_buckets=context_buckets,
            )

        with self.assertRaisesRegex(
            InputDataError,
            "metric_decomposition must be provided when there are multiple datasets.",
        ):
            ContextualDataset(
                datasets=dataset_list2,
                parameter_decomposition=parameter_decomposition,
                context_buckets=context_buckets,
            )

        with self.assertRaisesRegex(
            InputDataError,
            "metric_decomposition is redundant when there is "
            + "one dataset for overall outcome.",
        ):
            ContextualDataset(
                datasets=dataset_list1,
                parameter_decomposition=parameter_decomposition,
                context_buckets=context_buckets,
                metric_decomposition=metric_decomposition,
            )

        with self.assertRaisesRegex(
            InputDataError,
            "Keys of parameter decomposition and context buckets do not match.",
        ):
            ContextualDataset(
                datasets=dataset_list1,
                parameter_decomposition=parameter_decomposition,
                context_buckets=["context_0", "context_1"],
            )

        with self.assertRaisesRegex(
            InputDataError,
            "Keys of metric decomposition and context buckets do not match.",
        ):
            ContextualDataset(
                datasets=dataset_list2,
                parameter_decomposition=parameter_decomposition,
                context_buckets=context_buckets,
                metric_decomposition={
                    f"{c}": [f"y:{c}"] for c in context_buckets if c != "context_0"
                },
            )

        wrong_metric_decomposition = {
            f"{c}": [f"y:{c}"] for c in context_buckets if c != "context_0"
        }
        wrong_metric_decomposition["context_0"] = ["wrong_metric"]
        missing_outcome = "y:context_0"
        with self.assertRaisesRegex(
            InputDataError, f"{missing_outcome} is missing in metric_decomposition."
        ):
            ContextualDataset(
                datasets=dataset_list2,
                parameter_decomposition=parameter_decomposition,
                context_buckets=context_buckets,
                metric_decomposition=wrong_metric_decomposition,
            )

        # Test error for mixed Yvar
        dataset_list3[0]._Yvar = dataset_list2[0]._Yvar
        with self.assertRaisesRegex(
            InputDataError, "Require Yvar to be specified for all buckets, or for none"
        ):
            ContextualDataset(
                datasets=dataset_list3,
                parameter_decomposition=parameter_decomposition,
                context_buckets=context_buckets,
                metric_decomposition=wrong_metric_decomposition,
            )
