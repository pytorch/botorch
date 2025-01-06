#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models.latent_kronecker_gp import LatentKroneckerGP, MinMaxStandardize
from botorch.models.transforms import Normalize
from botorch.utils.testing import _get_random_data, BotorchTestCase
from botorch.utils.types import DEFAULT
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from linear_operator import settings
from linear_operator.utils.warnings import NumericalWarning, PerformanceWarning


def _get_data_with_missing_entries(
    n_train: int, d: int, m: int, batch_shape: torch.Size, tkwargs: dict
):
    train_X, train_Y = _get_random_data(
        batch_shape=batch_shape, m=m, d=d, n=n_train, **tkwargs
    )

    # randomly mask half of the training data
    train_Y_valid = torch.ones(n_train * m, dtype=torch.bool, device=tkwargs["device"])
    train_Y_valid[torch.randperm(n_train * m)[: n_train * m // 2]] = False
    train_Y_valid = train_Y_valid.reshape(n_train, m)
    train_Y[..., ~train_Y_valid] = torch.nan

    return train_X, train_Y, train_Y_valid


class TestLatentKroneckerGP(BotorchTestCase):
    def test_default_init(self):
        for (
            batch_shape,
            n_train,
            d,
            m,
            dtype,
            use_transforms,
        ) in itertools.product(
            (  # batch_shape
                torch.Size([]),
                torch.Size([1]),
                torch.Size([2, 3]),
            ),
            (10,),  # n_train
            (1, 2),  # d
            (1, 2),  # m
            (torch.float, torch.double),  # dtype
            (False, True),  # use_transforms
        ):
            tkwargs = {"device": self.device, "dtype": dtype}

            if use_transforms:
                intf = Normalize(d=d, batch_shape=batch_shape)
                octf = DEFAULT
            else:
                intf = None
                octf = None

            train_X, train_Y, train_Y_valid = _get_data_with_missing_entries(
                n_train=n_train, d=d, m=m, batch_shape=batch_shape, tkwargs=tkwargs
            )

            model = LatentKroneckerGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Y_valid=train_Y_valid,
                input_transform=intf,
                outcome_transform=octf,
            )
            model.to(**tkwargs)

            # test init
            train_Y_flat = train_Y.reshape(*batch_shape, -1)[
                ..., train_Y_valid.reshape(-1)
            ]
            if use_transforms:
                self.assertIsInstance(model.input_transform, Normalize)
                self.assertIsInstance(model.outcome_transform, MinMaxStandardize)
            else:
                self.assertFalse(hasattr(model, "input_transform"))
                self.assertFalse(hasattr(model, "outcome_transform"))
            train_Y_flat = (
                model.outcome_transform(train_Y_flat.unsqueeze(-1))[0].squeeze(-1)
                if use_transforms
                else train_Y_flat
            )
            self.assertAllClose(model.train_inputs[0], train_X, atol=0.0)
            self.assertAllClose(model.train_targets, train_Y_flat, atol=0.0)
            self.assertIsInstance(model.likelihood, GaussianLikelihood)
            self.assertIsInstance(model.mean_module_X, ZeroMean)
            self.assertIsInstance(model.mean_module_T, ZeroMean)
            self.assertIsInstance(model.covar_module_X, MaternKernel)
            self.assertIsInstance(model.covar_module_T, ScaleKernel)
            self.assertIsInstance(model.covar_module_T.base_kernel, MaternKernel)

    def test_custom_init(self):
        # test whether custom likelihoods and mean/covar modules are set correctly.
        for batch_shape, n_train, d, m, dtype in itertools.product(
            (  # batch_shape
                torch.Size([]),
                torch.Size([1]),
                torch.Size([2]),
                torch.Size([2, 3]),
            ),
            (10,),  # n_train
            (1, 2),  # d
            (1, 2),  # m
            (torch.float, torch.double),  # dtype
        ):
            tkwargs = {"device": self.device, "dtype": dtype}

            train_X, train_Y, train_Y_valid = _get_data_with_missing_entries(
                n_train=n_train, d=d, m=m, batch_shape=batch_shape, tkwargs=tkwargs
            )

            train_Y_valid_batched = train_Y_valid.expand(*batch_shape, n_train, m)
            if len(batch_shape) > 0:
                err_msg = (
                    "Explicit batch_shape not allowed for train_Y_valid, "
                    "because the mask must be shared across batch dimensions. "
                    f"Expected train_Y_valid with shape: {train_Y.shape[-2:]} "
                    f"(got {train_Y_valid_batched.shape})."
                )
                with self.assertRaises(BotorchTensorDimensionError) as e:
                    LatentKroneckerGP(
                        train_X=train_X,
                        train_Y=train_Y,
                        train_Y_valid=train_Y_valid_batched,
                    )
                self.assertEqual(err_msg, str(e.exception))

            T = torch.linspace(0, 1, m, **tkwargs)
            if len(batch_shape) > 0:
                expected_shape = torch.Size([*batch_shape, train_Y.shape[-1]])
                err_msg = f"Expected T with shape: {expected_shape} (got {T.shape})."
                with self.assertRaises(BotorchTensorDimensionError) as e:
                    LatentKroneckerGP(
                        train_X=train_X,
                        train_Y=train_Y,
                        T=T,
                    )
                self.assertEqual(err_msg, str(e.exception))

            T = T.expand(*batch_shape, m)

            likelihood = GaussianLikelihood(batch_shape=batch_shape)
            mean_module_X = ConstantMean(batch_shape=batch_shape)
            mean_module_T = ConstantMean(batch_shape=batch_shape)
            covar_module_X = RBFKernel(ard_num_dims=d, batch_shape=batch_shape)
            covar_module_T = RBFKernel(ard_num_dims=1, batch_shape=batch_shape)

            model = LatentKroneckerGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Y_valid=train_Y_valid,
                T=T,
                likelihood=likelihood,
                mean_module_X=mean_module_X,
                mean_module_T=mean_module_T,
                covar_module_X=covar_module_X,
                covar_module_T=covar_module_T,
            )
            model.to(**tkwargs)

            self.assertAllClose(model.T, T, atol=0.0)
            self.assertEqual(model.likelihood, likelihood)
            self.assertEqual(model.mean_module_X, mean_module_X)
            self.assertEqual(model.mean_module_T, mean_module_T)
            self.assertEqual(model.covar_module_X, covar_module_X)
            self.assertEqual(model.covar_module_T, covar_module_T)

            # check devices
            def _get_index(device):
                return device.index if device.index is not None else 0

            device_type = self.device.type
            device_idx = _get_index(self.device)

            self.assertEqual(model.train_inputs[0].device.type, device_type)
            self.assertEqual(_get_index(model.train_inputs[0].device), device_idx)
            self.assertEqual(model.train_targets.device.type, device_type)
            self.assertEqual(_get_index(model.train_targets.device), device_idx)
            self.assertEqual(model.mask.device.type, device_type)
            self.assertEqual(_get_index(model.mask.device), device_idx)
            self.assertEqual(model.T.device.type, device_type)
            self.assertEqual(_get_index(model.T.device), device_idx)
            for p in model.parameters():
                self.assertEqual(p.device.type, device_type)
                self.assertEqual(_get_index(p.device), device_idx)

    def test_custom_octf(self):
        for (
            batch_shape,
            n_train,
            d,
            m,
            dtype,
        ) in itertools.product(
            (  # batch_shape
                torch.Size([]),
                torch.Size([1]),
                torch.Size([2, 3]),
            ),
            (10,),  # n_train
            (1, 2),  # d
            (1, 2),  # m
            (torch.float, torch.double),  # dtype
        ):
            tkwargs = {"device": self.device, "dtype": dtype}

            octf = DEFAULT

            train_X, train_Y, train_Y_valid = _get_data_with_missing_entries(
                n_train=n_train, d=d, m=m, batch_shape=batch_shape, tkwargs=tkwargs
            )

            model = LatentKroneckerGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Y_valid=train_Y_valid,
                outcome_transform=octf,
            )
            model.to(**tkwargs)

            # test init
            train_Y_flat = train_Y.reshape(*batch_shape, -1)[
                ..., train_Y_valid.reshape(-1)
            ]

            self.assertIsInstance(model.outcome_transform, MinMaxStandardize)
            octf = model.outcome_transform

            # test MinMaxStandardize
            octf._is_trained = torch.tensor(False)
            # wrong batch shape
            with self.assertRaises(RuntimeError):
                octf(train_Y_flat.unsqueeze(-1).unsqueeze(0))
            octf._is_trained = torch.tensor(False)
            # wrong output dimension
            with self.assertRaises(RuntimeError):
                octf(train_Y_flat.unsqueeze(-1).repeat(1, 2))
            octf._is_trained = torch.tensor(False)
            # missing output dimension
            with self.assertRaises(ValueError):
                octf(torch.zeros(*batch_shape, 0, 1, **tkwargs))
            octf._is_trained = torch.tensor(False)
            # stdvs calculation with single observation
            octf(torch.zeros(*batch_shape, 1, 1, **tkwargs))
            self.assertAllClose(octf.stdvs, torch.ones(*batch_shape, 1, 1, **tkwargs))
            octf._is_trained = torch.tensor(False)
            # standardize specific output dimensions
            octf._outputs = []
            octf(train_Y_flat.unsqueeze(-1))
            self.assertAllClose(octf.means, torch.zeros_like(octf.means))
            self.assertAllClose(octf.stdvs, torch.ones_like(octf.stdvs))
            octf._outputs = None
            octf._is_trained = torch.tensor(False)

    def test_gp_train(self):
        for (
            batch_shape,
            n_train,
            d,
            m,
            dtype,
            use_transforms,
        ) in itertools.product(
            (  # batch_shape
                torch.Size([]),
                torch.Size([1]),
                torch.Size([2, 3]),
            ),
            (10,),  # n_train
            (1, 2),  # d
            (1, 2),  # m
            (torch.float, torch.double),  # dtype
            (False, True),  # use_transforms
        ):
            tkwargs = {"device": self.device, "dtype": dtype}

            if use_transforms:
                intf = Normalize(d=d, batch_shape=batch_shape)
                octf = DEFAULT
            else:
                intf = None
                octf = None

            train_X, train_Y, train_Y_valid = _get_data_with_missing_entries(
                n_train=n_train, d=d, m=m, batch_shape=batch_shape, tkwargs=tkwargs
            )

            model = LatentKroneckerGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Y_valid=train_Y_valid,
                input_transform=intf,
                outcome_transform=octf,
            )
            model.to(**tkwargs)

            # test optim
            model.train()
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll.to(**tkwargs)
            with warnings.catch_warnings(), model.use_iterative_methods():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                fit_gpytorch_mll(
                    mll, optimizer_kwargs={"options": {"maxiter": 1}}, max_attempts=1
                )

    def _test_gp_eval_shapes(
        self,
        batch_shape: torch.Size,
        use_transforms: bool,
        tkwargs: dict,
    ):
        n_train = 10
        n_test = 7
        d = 1
        m = 1

        if use_transforms:
            intf = Normalize(d=d, batch_shape=batch_shape)
            octf = DEFAULT
        else:
            intf = None
            octf = None

        train_X, train_Y, train_Y_valid = _get_data_with_missing_entries(
            n_train=n_train, d=d, m=m, batch_shape=batch_shape, tkwargs=tkwargs
        )

        model = LatentKroneckerGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Y_valid=train_Y_valid,
            input_transform=intf,
            outcome_transform=octf,
        )
        model.to(**tkwargs)
        model.eval()

        for test_shape in (
            torch.Size([]),
            torch.Size([3]),
            torch.Size([*batch_shape]),
            torch.Size([2, *batch_shape]),
        ):
            test_X = torch.rand(*test_shape, n_test, d, **tkwargs)

            # we expect an error if test_shape and batch_shape cannot be broadcasted
            try:
                broadcast_shape = torch.broadcast_shapes(test_shape, batch_shape)
            except RuntimeError as e:
                with self.assertRaisesRegex(RuntimeError, str(e)):
                    model.posterior(test_X)
                continue
            pred_shape = torch.Size([*broadcast_shape, n_test, m])

            # custom posterior samples
            posterior = model.posterior(test_X)
            self.assertEqual(posterior.batch_range, (0, -1))
            for sample_shape in (
                torch.Size([]),
                torch.Size([1]),
                torch.Size([2, 3]),
            ):
                # test posterior.rsample
                with warnings.catch_warnings(), model.use_iterative_methods():
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    pred_samples = posterior.rsample(sample_shape=sample_shape)
                self.assertEqual(
                    pred_samples.shape, torch.Size([*sample_shape, *pred_shape])
                )
                self.assertEqual(
                    pred_samples.shape,
                    posterior._extended_shape(torch.Size(sample_shape)),
                )
                # test posterior.rsample_from_base_samples
                base_samples = torch.randn(
                    *sample_shape,
                    *posterior.base_sample_shape,
                    **tkwargs,
                )
                with warnings.catch_warnings(), model.use_iterative_methods():
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    pred_samples = posterior.rsample_from_base_samples(
                        sample_shape, base_samples
                    )
                self.assertEqual(
                    pred_samples.shape, torch.Size([*sample_shape, *pred_shape])
                )
                # run again to test caching when using the same base samples
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    pred_samples = posterior.rsample_from_base_samples(
                        sample_shape, base_samples
                    )
                self.assertEqual(
                    pred_samples.shape, torch.Size([*sample_shape, *pred_shape])
                )
                if len(sample_shape) > 0:
                    # test incorrect base sample shape
                    incorrect_base_samples = torch.randn(
                        5,
                        *posterior.base_sample_shape,
                        **tkwargs,
                    )
                    with self.assertRaises(RuntimeError):
                        posterior.rsample_from_base_samples(
                            sample_shape, incorrect_base_samples
                        )

    def test_gp_eval_shapes_float_with_tf(self):
        use_transforms = True
        tkwargs = {"device": self.device, "dtype": torch.float}

        for batch_shape in (
            torch.Size([]),
            torch.Size([1]),
            torch.Size([2, 3]),
        ):
            self._test_gp_eval_shapes(
                batch_shape=batch_shape,
                use_transforms=use_transforms,
                tkwargs=tkwargs,
            )

    def test_gp_eval_shapes_double_with_tf(self):
        use_transforms = True
        tkwargs = {"device": self.device, "dtype": torch.double}

        for batch_shape in (
            torch.Size([]),
            torch.Size([1]),
            torch.Size([2, 3]),
        ):
            self._test_gp_eval_shapes(
                batch_shape=batch_shape,
                use_transforms=use_transforms,
                tkwargs=tkwargs,
            )

    def test_gp_eval_shapes_float_without_tf(self):
        use_transforms = False
        tkwargs = {"device": self.device, "dtype": torch.float}

        for batch_shape in (
            torch.Size([]),
            torch.Size([1]),
            torch.Size([2, 3]),
        ):
            self._test_gp_eval_shapes(
                batch_shape=batch_shape,
                use_transforms=use_transforms,
                tkwargs=tkwargs,
            )

    def test_gp_eval_shapes_double_without_tf(self):
        use_transforms = False
        tkwargs = {"device": self.device, "dtype": torch.double}

        for batch_shape in (
            torch.Size([]),
            torch.Size([1]),
            torch.Size([2, 3]),
        ):
            self._test_gp_eval_shapes(
                batch_shape=batch_shape,
                use_transforms=use_transforms,
                tkwargs=tkwargs,
            )

    def test_gp_eval_values(self):
        for (
            batch_shape,
            n_train,
            n_test,
            d,
            m,
            dtype,
            use_transforms,
        ) in itertools.product(
            (  # batch_shape
                torch.Size([]),
                torch.Size([1]),
                torch.Size([2, 3]),
            ),
            (10,),  # n_train
            (7,),  # n_test
            (1,),  # d
            (1,),  # m
            (torch.float, torch.double),  # dtype
            (False, True),  # use_transforms
        ):
            torch.manual_seed(12345)
            tkwargs = {"device": self.device, "dtype": dtype}

            if use_transforms:
                intf = Normalize(d=d, batch_shape=batch_shape)
                octf = DEFAULT
            else:
                intf = None
                octf = None

            train_X, train_Y, train_Y_valid = _get_data_with_missing_entries(
                n_train=n_train, d=d, m=m, batch_shape=batch_shape, tkwargs=tkwargs
            )

            model = LatentKroneckerGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Y_valid=train_Y_valid,
                input_transform=intf,
                outcome_transform=octf,
            )
            model.to(**tkwargs)
            model.eval()

            for test_shape in (
                torch.Size([]),
                torch.Size([3]),
                torch.Size([*batch_shape]),
                torch.Size([2, *batch_shape]),
            ):
                test_X = torch.rand(*test_shape, n_test, d, **tkwargs)

                # we expect an error if test_shape and batch_shape cannot be broadcasted
                try:
                    broadcast_shape = torch.broadcast_shapes(test_shape, batch_shape)
                except RuntimeError as e:
                    with self.assertRaisesRegex(RuntimeError, str(e)):
                        model.posterior(test_X)
                    continue
                pred_shape = torch.Size([*broadcast_shape, n_test, m])

                posterior = model.posterior(test_X)
                with warnings.catch_warnings(), model.use_iterative_methods():
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    pred_samples = posterior.rsample(sample_shape=(2048,))
                self.assertEqual(pred_samples.shape, torch.Size([2048, *pred_shape]))

                # GPyTorch predictions
                with model.use_iterative_methods():
                    pred = model(intf(test_X)) if intf is not None else model(test_X)
                pred_mean, pred_var = pred.mean, pred.variance
                pred_mean = pred_mean.reshape(*pred_mean.shape[:-1], n_test, m)
                pred_var = pred_var.reshape(*pred_var.shape[:-1], n_test, m)
                pred_mean, pred_var = (
                    model.outcome_transform.untransform(pred_mean, pred_var)
                    if octf is not None
                    else (pred_mean, pred_var)
                )
                self.assertEqual(pred_mean.shape, pred_shape)
                self.assertEqual(pred_var.shape, pred_shape)

                # check custom predictions and GPyTorch are roughly the same
                self.assertLess(
                    (pred_mean - pred_samples.mean(dim=0)).norm() / pred_mean.norm(),
                    0.1,
                )
                self.assertLess(
                    (pred_var - pred_samples.var(dim=0)).norm() / pred_var.norm(), 0.1
                )

    def test_iterative_methods(self):
        for batch_shape, n_train, d, m, dtype in itertools.product(
            (torch.Size([]),),  # batch_shape
            (10,),  # n_train
            (1,),  # d
            (1,),  # m
            (torch.float, torch.double),  # dtype
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y = _get_random_data(
                batch_shape=batch_shape, m=m, d=d, n=n_train, **tkwargs
            )

            model = LatentKroneckerGP(
                train_X=train_X,
                train_Y=train_Y,
            )
            model.to(**tkwargs)
            posterior = model.posterior(train_X)

            warn_msg = (
                "Iterative methods are disabled. Performing linear solve using "
                "full joint covariance matrix, which might be slow and require "
                "a lot of memory. Iterative methods can be enabled using "
                "'with model.use_iterative_methods():'."
            )

            with self.assertWarns(PerformanceWarning) as w:
                posterior.rsample()
            # Using this because self.assertWarnsRegex does not work for some reason
            self.assertEqual(warn_msg, str(w.warning))

            with model.use_iterative_methods():
                self.assertTrue(settings._fast_covar_root_decomposition.off())
                self.assertTrue(settings._fast_log_prob.on())
                self.assertTrue(settings._fast_solves.on())
                self.assertEqual(settings.cg_tolerance.value(), 0.01)
                self.assertEqual(settings.max_cg_iterations.value(), 10000)

    def test_not_implemented(self):
        batch_shape = torch.Size([])
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X, train_Y = _get_random_data(batch_shape=batch_shape, m=1, **tkwargs)

        model = LatentKroneckerGP(
            train_X=train_X,
            train_Y=train_Y,
        )
        model.to(**tkwargs)

        cls_name = model.__class__.__name__

        transform = ScalarizedPosteriorTransform(torch.tensor([1.0], **tkwargs))
        err_msg = f"Posterior transforms currently not supported for {cls_name}"
        with self.assertRaisesRegex(NotImplementedError, err_msg):
            model.posterior(train_X, posterior_transform=transform)

        err_msg = f"Observation noise currently not supported for {cls_name}"
        with self.assertRaisesRegex(NotImplementedError, err_msg):
            model.posterior(train_X, observation_noise=True)

        err_msg = f"Conditioning currently not supported for {cls_name}"
        with self.assertRaisesRegex(NotImplementedError, err_msg):
            model.condition_on_observations(train_X, train_Y)

        likelihood = FixedNoiseGaussianLikelihood(
            torch.tensor([1.0]), batch_shape=batch_shape, **tkwargs
        )
        model = LatentKroneckerGP(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
        )
        model.to(**tkwargs)

        err_msg = f"Only GaussianLikelihood currently supported for {cls_name}"
        with self.assertRaisesRegex(NotImplementedError, err_msg):
            model.posterior(train_X)
