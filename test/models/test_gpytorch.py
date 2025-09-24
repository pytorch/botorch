#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions import (
    BotorchTensorDimensionError,
    BotorchTensorDimensionWarning,
)
from botorch.exceptions.errors import DeprecationError, InputDataError
from botorch.exceptions.warnings import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models.gpytorch import (
    BatchedMultiOutputGPyTorchModel,
    GPyTorchModel,
    ModelListGPyTorchModel,
)
from botorch.models.model import FantasizeMixin
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import ChainedInputTransform, InputTransform
from botorch.models.utils import fantasize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.test_helpers import SimpleGPyTorchModel
from botorch.utils.testing import BotorchTestCase, get_random_data
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP, IndependentModelList
from gpytorch.settings import trace_mode
from torch import Tensor


class SimpleInputTransform(InputTransform, torch.nn.Module):
    def __init__(self, transform_on_train: bool) -> None:
        r"""
        Args:
            transform_on_train: A boolean indicating whether to apply the
                transform in train() mode.
        """
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = True
        self.transform_on_fantasize = True
        # to test the `input_transform.to()` call
        self.register_buffer("add_value", torch.ones(1))

    def transform(self, X: Tensor) -> Tensor:
        return X + self.add_value


class SimpleBatchedMultiOutputGPyTorchModel(
    BatchedMultiOutputGPyTorchModel, ExactGP, FantasizeMixin
):
    _batch_shape: torch.Size | None = None

    def __init__(self, train_X, train_Y, outcome_transform=None, input_transform=None):
        r"""
        Args:
            train_X: A tensor of inputs, passed to self.transform_inputs.
            train_Y: Passed to outcome_transform.
            outcome_transform: Transform applied to train_Y.
            input_transform: A Module that performs the input transformation, passed to
                self.transform_inputs.
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            outcome_transform.train()
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(transformed_X, train_Y)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        likelihood = GaussianLikelihood(batch_shape=self._aug_batch_shape)
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=self._aug_batch_shape),
            batch_shape=self._aug_batch_shape,
        )
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

    def forward(self, x):
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @property
    def batch_shape(self) -> torch.Size:
        if self._batch_shape is not None:
            return self._batch_shape
        return super().batch_shape


class SimpleModelListGPyTorchModel(IndependentModelList, ModelListGPyTorchModel):
    def __init__(self, *gp_models: GPyTorchModel):
        r"""
        Args:
            gp_models: Arbitrary number of GPyTorchModels.
        """
        super().__init__(*gp_models)


class TestGPyTorchModel(BotorchTestCase):
    def test_gpytorch_model(self):
        for dtype, use_octf in itertools.product(
            (torch.float, torch.double), (False, True)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=1) if use_octf else None
            train_X = torch.rand(5, 1, **tkwargs)
            train_Y = torch.sin(train_X)
            # basic test
            model = SimpleGPyTorchModel(train_X, train_Y, octf)
            self.assertEqual(model.num_outputs, 1)
            self.assertEqual(model.batch_shape, torch.Size())
            test_X = torch.rand(2, 1, **tkwargs)
            posterior = model.posterior(test_X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 1]))
            if use_octf:
                # ensure un-transformation is applied
                tmp_tf = model.outcome_transform
                del model.outcome_transform
                p_tf = model.posterior(test_X)
                model.outcome_transform = tmp_tf
                expected_var = tmp_tf.untransform_posterior(p_tf).variance
                self.assertAllClose(posterior.variance, expected_var)
            # test observation noise
            posterior = model.posterior(test_X, observation_noise=True)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 1]))
            posterior = model.posterior(
                test_X, observation_noise=torch.rand(2, 1, **tkwargs)
            )
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 1]))
            # test noise shape validation
            with self.assertRaises(BotorchTensorDimensionError):
                model.posterior(test_X, observation_noise=torch.rand(2, **tkwargs))
            # test subset_output
            with self.assertRaises(NotImplementedError):
                model.subset_output([0])

            # test fantasize
            n_samps = 2
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_samps]))
            for n in [0, 2]:
                x = torch.rand(n, 1, **tkwargs)
                cm = model.fantasize(X=x, sampler=sampler)
                self.assertIsInstance(cm, SimpleGPyTorchModel)
                self.assertEqual(cm.train_targets.shape, torch.Size([n_samps, 5 + n]))
                cm = model.fantasize(
                    X=x,
                    sampler=sampler,
                    observation_noise=torch.rand(n, 1, **tkwargs),
                )
                self.assertIsInstance(cm, SimpleGPyTorchModel)
                self.assertEqual(cm.train_targets.shape, torch.Size([n_samps, 5 + n]))

            # test that boolean observation noise is deprecated
            msg = "`fantasize` no longer accepts a boolean for `observation_noise`."
            with self.assertRaisesRegex(DeprecationError, msg):
                model.fantasize(
                    torch.rand(2, 1, **tkwargs),
                    sampler=sampler,
                    observation_noise=True,
                )

    def test_validate_tensor_args(self) -> None:
        n, d = 3, 2
        for batch_shape, output_dim_shape, dtype in itertools.product(
            (torch.Size(), torch.Size([2])),
            (torch.Size(), torch.Size([1]), torch.Size([2])),
            (torch.float, torch.double),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            X = torch.empty(batch_shape + torch.Size([n, d]), **tkwargs)
            # test using the same batch_shape as X
            Y = torch.empty(batch_shape + torch.Size([n]) + output_dim_shape, **tkwargs)
            if len(output_dim_shape) > 0:
                # check that no exception is raised
                for strict in [False, True]:
                    GPyTorchModel._validate_tensor_args(X, Y, strict=strict)
            else:
                expected_message = (
                    "An explicit output dimension is required for targets."
                )
                with self.assertRaisesRegex(
                    BotorchTensorDimensionError, expected_message
                ):
                    GPyTorchModel._validate_tensor_args(X, Y)
                with self.assertWarnsRegex(
                    BotorchTensorDimensionWarning,
                    (
                        "Non-strict enforcement of botorch tensor conventions. "
                        "The following error would have been raised with strict "
                        "enforcement: "
                    )
                    + expected_message,
                ):
                    GPyTorchModel._validate_tensor_args(X, Y, strict=False)
            # test using different batch_shape
            if len(batch_shape) > 0:
                expected_message = (
                    "Expected X and Y to have the same number of dimensions"
                )
                with self.assertRaisesRegex(
                    BotorchTensorDimensionError, expected_message
                ):
                    GPyTorchModel._validate_tensor_args(X, Y[0])
                with self.assertWarnsRegex(
                    BotorchTensorDimensionWarning,
                    (
                        "Non-strict enforcement of botorch tensor conventions. "
                        "The following error would have been raised with strict "
                        "enforcement: "
                    )
                    + expected_message,
                ):
                    GPyTorchModel._validate_tensor_args(X, Y[0], strict=False)
            # with Yvar
            if len(output_dim_shape) > 0:
                Yvar = torch.empty(torch.Size([n]) + output_dim_shape, **tkwargs)
                GPyTorchModel._validate_tensor_args(X, Y, Yvar)
                Yvar = torch.empty(n, 5, **tkwargs)
                for strict in [False, True]:
                    with self.assertRaisesRegex(
                        BotorchTensorDimensionError,
                        "An explicit output dimension is required for "
                        "observation noise.",
                    ):
                        GPyTorchModel._validate_tensor_args(X, Y, Yvar, strict=strict)

    def test_fantasize_flag(self):
        train_X = torch.rand(5, 1)
        train_Y = torch.sin(train_X)
        model = SimpleGPyTorchModel(train_X, train_Y)
        model.eval()
        test_X = torch.ones(1, 1)
        model(test_X)
        self.assertFalse(model.last_fantasize_flag)
        model.posterior(test_X)
        self.assertFalse(model.last_fantasize_flag)
        model.fantasize(test_X, SobolQMCNormalSampler(sample_shape=torch.Size([2])))
        self.assertTrue(model.last_fantasize_flag)
        model.last_fantasize_flag = False
        with fantasize():
            model.posterior(test_X)
            self.assertTrue(model.last_fantasize_flag)

    def test_input_transform(self):
        # simple test making sure that the input transforms are applied to both
        # train and test inputs
        for dtype, transform_on_train in itertools.product(
            (torch.float, torch.double), (False, True)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X = torch.rand(5, 1, **tkwargs)
            train_Y = torch.sin(train_X)
            intf = SimpleInputTransform(transform_on_train)
            model = SimpleGPyTorchModel(train_X, train_Y, input_transform=intf)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 2}})

            test_X = torch.rand(2, 1, **tkwargs)
            model.posterior(test_X)
            # posterior calls model.forward twice, one with training inputs only,
            # other with both train and test inputs
            expected_train = intf(train_X) if transform_on_train else train_X
            expected_test = intf(test_X)
            self.assertTrue(
                torch.equal(model.transformed_call_args[-2], expected_train)
            )
            self.assertTrue(
                torch.equal(
                    model.transformed_call_args[-1],
                    torch.cat([expected_train, expected_test], dim=0),
                )
            )

    def test_posterior_transform(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X = torch.rand(5, 1, **tkwargs)
        train_Y = torch.sin(train_X)
        model = SimpleGPyTorchModel(train_X, train_Y)
        post_tf = ScalarizedPosteriorTransform(weights=torch.zeros(1, **tkwargs))
        post = model.posterior(torch.rand(3, 1, **tkwargs), posterior_transform=post_tf)
        self.assertTrue(torch.equal(post.mean, torch.zeros(3, 1, **tkwargs)))

    def test_float_warning_and_dtype_error(self):
        with self.assertWarnsRegex(InputDataWarning, "double precision"):
            SimpleGPyTorchModel(torch.rand(5, 1), torch.randn(5, 1))
        with self.assertRaisesRegex(InputDataError, "same dtype"):
            SimpleGPyTorchModel(torch.rand(5, 1), torch.randn(5, 1, dtype=torch.double))


class TestBatchedMultiOutputGPyTorchModel(BotorchTestCase):
    def test_batched_multi_output_gpytorch_model(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X = torch.rand(5, 1, **tkwargs)
            train_Y = torch.cat([torch.sin(train_X), torch.cos(train_X)], dim=-1)
            # basic test
            model = SimpleBatchedMultiOutputGPyTorchModel(train_X, train_Y)
            self.assertEqual(model.num_outputs, 2)
            self.assertEqual(model.batch_shape, torch.Size())
            test_X = torch.rand(2, 1, **tkwargs)
            posterior = model.posterior(test_X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
            # test observation noise
            posterior = model.posterior(test_X, observation_noise=True)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
            posterior = model.posterior(
                test_X, observation_noise=torch.rand(2, 2, **tkwargs)
            )
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
            # test subset_output
            with self.assertRaises(NotImplementedError):
                model.subset_output([0])
            # test fantasize
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]))
            cm = model.fantasize(torch.rand(2, 1, **tkwargs), sampler=sampler)
            self.assertIsInstance(cm, SimpleBatchedMultiOutputGPyTorchModel)
            self.assertEqual(cm.train_targets.shape, torch.Size([2, 2, 7]))
            cm = model.fantasize(
                torch.rand(2, 1, **tkwargs),
                sampler=sampler,
                observation_noise=torch.rand(2, 2, **tkwargs),
            )
            self.assertIsInstance(cm, SimpleBatchedMultiOutputGPyTorchModel)
            self.assertEqual(cm.train_targets.shape, torch.Size([2, 2, 7]))

            # test get_batch_dimensions
            get_batch_dims = SimpleBatchedMultiOutputGPyTorchModel.get_batch_dimensions
            for input_batch_dim in (0, 3):
                for num_outputs in (1, 2):
                    input_batch_shape, aug_batch_shape = get_batch_dims(
                        train_X=(
                            train_X.unsqueeze(0).expand(3, 5, 1)
                            if input_batch_dim == 3
                            else train_X
                        ),
                        train_Y=train_Y[:, 0:1] if num_outputs == 1 else train_Y,
                    )
                    expected_input_batch_shape = (
                        torch.Size([3]) if input_batch_dim == 3 else torch.Size([])
                    )
                    self.assertEqual(input_batch_shape, expected_input_batch_shape)
                    self.assertEqual(
                        aug_batch_shape,
                        (
                            expected_input_batch_shape + torch.Size([])
                            if num_outputs == 1
                            else expected_input_batch_shape + torch.Size([2])
                        ),
                    )

    def test_condition_on_observations(self):
        for dtype, use_octf in itertools.product(
            (torch.float, torch.double), (False, True)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=1) if use_octf else None
            train_X = torch.rand(5, 1, **tkwargs)
            train_Y = torch.sin(train_X)
            model = SimpleGPyTorchModel(train_X, train_Y, octf)

            # must predict before conitioning on observations
            model.posterior(torch.rand(2, 1, **tkwargs))

            # test conditioning on observations
            cm = model.condition_on_observations(
                torch.rand(2, 1, **tkwargs), torch.rand(2, 1, **tkwargs)
            )
            self.assertIsInstance(cm, SimpleGPyTorchModel)
            self.assertEqual(cm.train_targets.shape, torch.Size([7]))
            model = SimpleGPyTorchModel(torch.rand(5, 1), torch.randn(5, 1))

        model.posterior(torch.rand(2, 1))  # evaluate the model to form caches.
        # Outside of fantasize, the inputs are validated.
        with self.assertWarnsRegex(
            BotorchTensorDimensionWarning, "Non-strict enforcement of"
        ):
            model.condition_on_observations(torch.randn(2, 1), torch.randn(5, 2, 1))
        # Inside of fantasize, the inputs are not validated.
        with fantasize(), warnings.catch_warnings(record=True) as ws:
            warnings.filterwarnings("always", category=BotorchTensorDimensionWarning)
            model.condition_on_observations(torch.randn(2, 1), torch.randn(5, 2, 1))
        self.assertFalse(any(w.category is BotorchTensorDimensionWarning for w in ws))

    def test_condition_on_observations_batched(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X = torch.rand(5, 1, **tkwargs)
            train_Y = torch.cat([torch.sin(train_X), torch.cos(train_X)], dim=-1)
            model = SimpleBatchedMultiOutputGPyTorchModel(train_X, train_Y)

            # must predict before conitioning on observations
            model.posterior(torch.rand(2, 1, **tkwargs))

            # test conditioning on observations
            cm = model.condition_on_observations(
                torch.rand(2, 1, **tkwargs), torch.rand(2, 2, **tkwargs)
            )
            self.assertIsInstance(cm, SimpleBatchedMultiOutputGPyTorchModel)
            self.assertEqual(cm.train_targets.shape, torch.Size([2, 7]))

    def test_posterior_transform(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X = torch.rand(5, 2, **tkwargs)
        train_Y = torch.sin(train_X)
        model = SimpleBatchedMultiOutputGPyTorchModel(train_X, train_Y)
        post_tf = ScalarizedPosteriorTransform(weights=torch.zeros(2, **tkwargs))
        post = model.posterior(torch.rand(3, 2, **tkwargs), posterior_transform=post_tf)
        self.assertTrue(torch.equal(post.mean, torch.zeros(3, 1, **tkwargs)))

    def test_posterior_in_trace_mode(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X = torch.rand(5, 1, **tkwargs)
        train_Y = torch.cat([torch.sin(train_X), torch.cos(train_X)], dim=-1)
        model = SimpleBatchedMultiOutputGPyTorchModel(train_X, train_Y)

        class MeanVarModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                # get the model posterior
                posterior = self.model.posterior(x, observation_noise=True)
                mean = posterior.mean.detach()
                std = posterior.variance.sqrt().detach()
                return mean, std

        wrapped_model = MeanVarModelWrapper(model)
        with torch.no_grad(), trace_mode():
            X_test = torch.rand(3, 1, **tkwargs)
            wrapped_model(X_test)  # Compute caches
            traced_model = torch.jit.trace(wrapped_model, X_test)
            mean, std = traced_model(X_test)
            self.assertEqual(mean.shape, torch.Size([3, 2]))


class TestModelListGPyTorchModel(BotorchTestCase):
    def test_model_list_gpytorch_model(self):
        torch.manual_seed(12345)
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X1, train_X2 = (
                torch.rand(5, 1, **tkwargs),
                torch.rand(5, 1, **tkwargs),
            )
            train_Y1 = torch.sin(train_X1)
            train_Y2 = torch.cos(train_X2)
            # test SAAS type batch shape
            m1 = SimpleBatchedMultiOutputGPyTorchModel(train_X1, train_Y1)
            m2 = SimpleBatchedMultiOutputGPyTorchModel(train_X2, train_Y2)
            m1._batch_shape = torch.Size([2])
            m2._batch_shape = torch.Size([2])
            model = SimpleModelListGPyTorchModel(m1, m2)
            self.assertEqual(model.batch_shape, torch.Size([2]))
            # test different batch shapes (broadcastable)
            m1 = SimpleGPyTorchModel(
                train_X1.expand(2, *train_X1.shape), train_Y1.expand(2, *train_Y1.shape)
            )
            m2 = SimpleGPyTorchModel(train_X2, train_Y2)
            model = SimpleModelListGPyTorchModel(m1, m2)
            self.assertEqual(model.num_outputs, 2)
            msg = (
                "Component models of SimpleModelListGPyTorchModel have "
                "different batch shapes"
            )
            with self.assertWarnsRegex(Warning, msg):
                self.assertEqual(model.batch_shape, torch.Size([2]))
            # test different batch shapes (not broadcastable)
            m2 = SimpleGPyTorchModel(
                train_X2.expand(3, *train_X2.shape), train_Y2.expand(3, *train_Y2.shape)
            )
            model = SimpleModelListGPyTorchModel(m1, m2)
            with self.assertRaises(NotImplementedError):
                model.batch_shape
            # test same batch shape
            m2 = SimpleGPyTorchModel(
                train_X2.expand(2, *train_X2.shape), train_Y2.expand(2, *train_Y2.shape)
            )
            model = SimpleModelListGPyTorchModel(m1, m2)
            self.assertEqual(model.num_outputs, 2)
            self.assertEqual(model.batch_shape, torch.Size([2]))
            # test non-batch
            m1 = SimpleGPyTorchModel(train_X1, train_Y1)
            m2 = SimpleGPyTorchModel(train_X2, train_Y2)
            model = SimpleModelListGPyTorchModel(m1, m2)
            self.assertEqual(model.batch_shape, torch.Size([]))
            test_X = torch.rand(2, 1, **tkwargs)
            posterior = model.posterior(test_X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
            # test multioutput
            train_x_raw, train_y = get_random_data(
                batch_shape=torch.Size(), m=1, n=10, **tkwargs
            )
            task_idx = torch.cat(
                [torch.ones(5, 1, **tkwargs), torch.zeros(5, 1, **tkwargs)], dim=0
            )
            train_x = torch.cat([train_x_raw, task_idx], dim=-1)
            model_mt = MultiTaskGP(
                train_X=train_x,
                train_Y=train_y,
                task_feature=-1,
            )
            mt_posterior = model_mt.posterior(test_X)
            model = SimpleModelListGPyTorchModel(m1, model_mt, m2)
            posterior2 = model.posterior(test_X)
            expected_mean = torch.cat(
                (
                    posterior.mean[:, 0].unsqueeze(-1),
                    mt_posterior.mean,
                    posterior.mean[:, 1].unsqueeze(-1),
                ),
                dim=1,
            )
            self.assertAllClose(expected_mean, posterior2.mean)
            expected_covariance = torch.block_diag(
                posterior.covariance_matrix[:2, :2],
                mt_posterior.covariance_matrix[:2, :2],
                mt_posterior.covariance_matrix[-2:, -2:],
                posterior.covariance_matrix[-2:, -2:],
            )
            self.assertAllClose(
                expected_covariance, posterior2.covariance_matrix, atol=1e-5
            )
            # test output indices
            posterior = model.posterior(test_X)
            for output_indices in ([0], [1], [0, 1]):
                posterior_subset = model.posterior(
                    test_X, output_indices=output_indices
                )
                self.assertIsInstance(posterior_subset, GPyTorchPosterior)
                self.assertEqual(
                    posterior_subset.mean.shape, torch.Size([2, len(output_indices)])
                )
                self.assertAllClose(
                    posterior_subset.mean,
                    posterior.mean[..., output_indices],
                    atol=1e-6,
                )
                self.assertAllClose(
                    posterior_subset.variance,
                    posterior.variance[..., output_indices],
                    atol=2e-6 if dtype is torch.float else 1e-6,
                    rtol=3e-4 if dtype is torch.float else 1e-5,
                )
            # test observation noise
            model = SimpleModelListGPyTorchModel(m1, m2)
            posterior = model.posterior(test_X, observation_noise=True)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
            posterior = model.posterior(
                test_X, observation_noise=torch.rand(2, 2, **tkwargs)
            )
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
            posterior = model.posterior(
                test_X,
                output_indices=[0],
                observation_noise=torch.rand(2, 2, **tkwargs),
            )
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, torch.Size([2, 1]))

    def test_input_transform(self):
        # test that the input transforms are applied properly to individual models
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X1, train_X2 = (
                torch.rand(5, 1, **tkwargs),
                torch.rand(5, 1, **tkwargs),
            )
            train_Y1 = torch.sin(train_X1)
            train_Y2 = torch.cos(train_X2)
            # test transform on only one model
            m1 = SimpleGPyTorchModel(train_X1, train_Y1)
            m2_tf = SimpleInputTransform(True)
            m2 = SimpleGPyTorchModel(train_X2, train_Y2, input_transform=m2_tf)
            # test `input_transform.to(X)` call
            self.assertEqual(m2_tf.add_value.dtype, dtype)
            self.assertEqual(m2_tf.add_value.device.type, self.device.type)
            # train models to have the train inputs preprocessed
            for m in [m1, m2]:
                mll = ExactMarginalLogLikelihood(m.likelihood, m)
                fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 2}})
            model = SimpleModelListGPyTorchModel(m1, m2)

            test_X = torch.rand(2, 1, **tkwargs)
            model.posterior(test_X)
            # posterior calls model.forward twice, one with training inputs only,
            # other with both train and test inputs
            for m, t_X in [[m1, train_X1], [m2, train_X2]]:
                expected_train = m.transform_inputs(t_X)
                expected_test = m.transform_inputs(test_X)
                self.assertTrue(
                    torch.equal(m.transformed_call_args[-2], expected_train)
                )
                self.assertTrue(
                    torch.equal(
                        m.transformed_call_args[-1],
                        torch.cat([expected_train, expected_test], dim=0),
                    )
                )

            # different transforms on the two models
            m1_tf = ChainedInputTransform(
                tf1=SimpleInputTransform(False),
                tf2=SimpleInputTransform(True),
            )
            m1 = SimpleGPyTorchModel(train_X1, train_Y1, input_transform=m1_tf)
            m2_tf = SimpleInputTransform(False)
            m2 = SimpleGPyTorchModel(train_X2, train_Y2, input_transform=m2_tf)
            for m in [m1, m2]:
                mll = ExactMarginalLogLikelihood(m.likelihood, m)
                fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 2}})
            model = SimpleModelListGPyTorchModel(m1, m2)
            model.posterior(test_X)
            for m, t_X in [[m1, train_X1], [m2, train_X2]]:
                expected_train = m.input_transform.preprocess_transform(t_X)
                expected_test = m.transform_inputs(test_X)
                self.assertTrue(
                    torch.equal(m.transformed_call_args[-2], expected_train)
                )
                self.assertTrue(
                    torch.equal(
                        m.transformed_call_args[-1],
                        torch.cat([expected_train, expected_test], dim=0),
                    )
                )

    def test_posterior_transform(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X1, train_X2 = (
            torch.rand(5, 1, **tkwargs),
            torch.rand(5, 1, **tkwargs),
        )
        train_Y1 = torch.sin(train_X1)
        train_Y2 = torch.cos(train_X2)
        # test different batch shapes
        m1 = SimpleGPyTorchModel(train_X1, train_Y1)
        m2 = SimpleGPyTorchModel(train_X2, train_Y2)
        model = SimpleModelListGPyTorchModel(m1, m2)
        post_tf = ScalarizedPosteriorTransform(torch.ones(2, **tkwargs))
        post = model.posterior(torch.rand(3, 1, **tkwargs), posterior_transform=post_tf)
        self.assertEqual(post.mean.shape, torch.Size([3, 1]))

    def test_condition_on_observations_model_list(self):
        torch.manual_seed(12345)
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X1, train_X2 = (
                torch.rand(5, 1, **tkwargs),
                torch.rand(5, 1, **tkwargs),
            )
            train_Y1 = torch.sin(train_X1)
            train_Y2 = torch.cos(train_X2)
            m1 = SimpleGPyTorchModel(train_X1, train_Y1)
            m2 = SimpleGPyTorchModel(train_X2, train_Y2)
            model = SimpleModelListGPyTorchModel(m1, m2)

            # conditioning is not implemented (see ModelListGP for tests)
            with self.assertRaises(NotImplementedError):
                model.condition_on_observations(
                    X=torch.rand(2, 1, **tkwargs), Y=torch.rand(2, 2, **tkwargs)
                )

    def test_condition_on_observations_input_transform_consistency(self):
        """Test that input transforms are applied consistently in
         condition_on_observations.

        This addresses https://github.com/pytorch/botorch/issues/2533:
        inputs should be transformed when conditioning.
        """
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}

            # Create model with input transform
            train_X = torch.tensor([[0.0], [0.5], [1.0]], **tkwargs)
            train_Y = torch.tensor([[1.0], [2.0], [3.0]], **tkwargs)

            input_transform = SimpleInputTransform(transform_on_train=True)
            model = SimpleGPyTorchModel(
                train_X, train_Y, input_transform=input_transform
            )

            # Condition on new observations
            new_X = torch.tensor([[0.25], [0.75]], **tkwargs)
            new_Y = torch.tensor([[1.5], [2.5]], **tkwargs)

            # Get original train_inputs for comparison
            model.eval()  # Put in eval mode to see transformed inputs
            original_transformed_inputs = model.train_inputs[0].clone()
            _ = model.posterior(train_X)
            # Condition on observations
            conditioned_model = model.condition_on_observations(new_X, new_Y)
            conditioned_model.eval()  # Put in eval mode to see transformed inputs

            # Check that new inputs were transformed before being added
            expected_transformed_new_X = input_transform(new_X)
            expected_combined_inputs = torch.cat(
                [original_transformed_inputs, expected_transformed_new_X], dim=0
            )

            # NOTE This would not have passed before - the last two inputs
            # (corresponding to new_X) would not have been transformed.
            self.assertAllClose(
                conditioned_model.train_inputs[0],
                expected_combined_inputs,
            )

    def test_condition_on_observations_train_input_shapes(self):
        """Comprehensive test for condition_on_observations functionality.

        Tests input transform consistency, train/eval mode stability,
        different transform settings, and batch shape handling.
        """
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}

            # Test 1: Train/eval mode stability with transform_on_train=True
            train_X = torch.tensor([[0.0], [1.0]], **tkwargs)
            train_Y = torch.tensor([[1.0], [2.0]], **tkwargs)
            input_transform = SimpleInputTransform(transform_on_train=True)
            model = SimpleGPyTorchModel(
                train_X, train_Y, input_transform=input_transform
            )

            new_X = torch.tensor([[0.5]], **tkwargs)
            new_Y = torch.tensor([[1.5]], **tkwargs)
            _ = model.posterior(train_X)
            conditioned_model = model.condition_on_observations(new_X, new_Y)

            # Verify conditioned observations persist across train/eval modes
            conditioned_model.eval()
            self.assertEqual(conditioned_model.train_targets.shape[0], 3)
            conditioned_model.train()
            self.assertEqual(conditioned_model.train_targets.shape[0], 3)
            self.assertEqual(conditioned_model.train_inputs[0].shape[0], 3)

            # Test 2: Transform behavior with transform_on_train=False
            model2 = SimpleGPyTorchModel(
                train_X,
                train_Y,
                input_transform=SimpleInputTransform(transform_on_train=False),
            )
            _ = model2.posterior(train_X)
            conditioned_model2 = model2.condition_on_observations(new_X, new_Y)
            self.assertEqual(conditioned_model2.train_targets.shape[0], 3)

            # Verify model can make predictions after conditioning
            test_X = torch.tensor([[0.25]], **tkwargs)
            posterior = conditioned_model2.posterior(test_X)
            self.assertEqual(posterior.mean.shape, torch.Size([1, 1]))

        # Test 3: Batch shape handling and broadcasting (double precision only)
        tkwargs = {"device": self.device, "dtype": torch.double}

        # Same ndim - should update _original_train_inputs
        train_X = torch.rand(2, 2, **tkwargs)
        train_Y = torch.rand(2, 1, **tkwargs)
        model = SimpleGPyTorchModel(
            train_X, train_Y, input_transform=SimpleInputTransform(True)
        )
        _ = model.posterior(train_X)

        original_size = model._original_train_inputs.shape[0]
        fantasy_model = model.condition_on_observations(
            torch.rand(1, 2, **tkwargs), torch.rand(1, 1, **tkwargs)
        )
        self.assertEqual(
            fantasy_model._original_train_inputs.shape[0], original_size + 1
        )

        # Different ndim - should NOT update _original_train_inputs
        original_size = model._original_train_inputs.shape[0]
        fantasy_model = model.condition_on_observations(
            torch.rand(3, 2, 2, **tkwargs), torch.rand(3, 2, 1, **tkwargs)
        )

        # NOTE expected behavior is expand (2, 2) & (3, 2, 2) is expanded
        # and then concatenated along dimension -2 --> (3, 4, 2)
        self.assertEqual(
            fantasy_model._original_train_inputs.shape, torch.Size([3, 4, 2])
        )

        # Test 4: Fantasy model behavior
        model2 = SimpleGPyTorchModel(
            train_X, train_Y, input_transform=SimpleInputTransform(True)
        )
        _ = model2.posterior(train_X)
        original_size = model2._original_train_inputs.shape[0]

        fantasy_model = model2.condition_on_observations(
            torch.rand(1, 2, **tkwargs), torch.rand(1, 1, **tkwargs)
        )

        # Fantasy model gets data, original model does not
        self.assertEqual(
            fantasy_model._original_train_inputs.shape[0], original_size + 1
        )
        self.assertEqual(model2._original_train_inputs.shape[0], original_size)
