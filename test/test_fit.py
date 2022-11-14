#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from contextlib import nullcontext
from copy import deepcopy
from itertools import filterfalse, product
from typing import Callable, Iterable, Optional
from unittest.mock import MagicMock, patch
from warnings import catch_warnings, warn, WarningMessage

import torch
from botorch import fit
from botorch.exceptions.errors import ModelFittingError, UnsupportedError
from botorch.exceptions.warnings import BotorchWarning, OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import (
    FixedNoiseGP,
    HeteroskedasticSingleTaskGP,
    SingleTaskGP,
    SingleTaskVariationalGP,
)
from botorch.models.converter import batched_to_model_list
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from botorch.optim.closures import get_loss_closure_with_grads
from botorch.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from botorch.optim.utils import allclose_mll, get_data_loader
from botorch.settings import debug
from botorch.utils.context_managers import (
    del_attribute_ctx,
    module_rollback_ctx,
    requires_grad_ctx,
    TensorCheckpoint,
)
from botorch.utils.dispatcher import MDNotImplementedError
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from linear_operator.utils.errors import NotPSDError

MAX_ITER_MSG = "TOTAL NO. of ITERATIONS REACHED LIMIT"


class MockOptimizer:
    def __init__(
        self,
        randomize_requires_grad: bool = True,
        warnings: Iterable[WarningMessage] = (),
        exception: Optional[BaseException] = None,
    ):
        r"""Class used to mock `optimizer` argument to `fit_gpytorch_mll."""
        self.randomize_requires_grad = randomize_requires_grad
        self.warnings = warnings
        self.exception = exception
        self.call_count = 0

    def __call__(self, mll, closure: Optional[Callable] = None):
        self.call_count += 1
        for w in self.warnings:
            warn(str(w.message), w.category)

        if self.randomize_requires_grad:
            with torch.no_grad():
                for param in mll.parameters():
                    if param.requires_grad:
                        param[...] = torch.rand_like(param)

        if self.exception is not None:
            raise self.exception

        return mll, None


class TestFitAPI(BotorchTestCase):
    r"""Unit tests for general fitting API"""

    def setUp(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.linspace(0, 1, 10).unsqueeze(-1)
            train_F = torch.sin(2 * math.pi * train_X)
            train_Y = train_F + 0.1 * torch.randn_like(train_F)

            model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                input_transform=Normalize(d=1),
                outcome_transform=Standardize(m=1),
            )
            self.mll = ExactMarginalLogLikelihood(model.likelihood, model)

    def test_fit_gyptorch_model(self):
        r"""Test support for legacy API"""

        # Test `option` argument
        options = {"foo": 0}
        with catch_warnings(), patch.object(
            fit,
            "fit_gpytorch_mll",
            new=lambda mll, optimizer_kwargs=None, **kwargs: optimizer_kwargs,
        ):
            self.assertEqual(
                {"options": options, "bar": 1},
                fit.fit_gpytorch_model(
                    self.mll,
                    options=options,
                    optimizer_kwargs={"bar": 1},
                ),
            )

        # Test `max_retries` argument
        with catch_warnings(), patch.object(
            fit,
            "fit_gpytorch_mll",
            new=lambda mll, max_attempts=None, **kwargs: max_attempts,
        ):
            self.assertEqual(100, fit.fit_gpytorch_model(self.mll, max_retries=100))

        # Test `exclude` argument
        self.assertTrue(self.mll.model.mean_module.constant.requires_grad)
        with catch_warnings(), patch.object(
            fit,
            "fit_gpytorch_mll",
            new=lambda mll, **kwargs: mll.model.mean_module.constant.requires_grad,
        ):
            self.assertFalse(
                fit.fit_gpytorch_model(
                    self.mll,
                    options=options,
                    exclude=["model.mean_module.constant"],
                )
            )
        self.assertTrue(self.mll.model.mean_module.constant.requires_grad)

        # Test collisions
        with catch_warnings(record=True) as ws, self.assertRaises(SyntaxError):
            fit.fit_gpytorch_model(
                self.mll,
                options=options,
                optimizer_kwargs={"options": {"bar": 1}},
            )
            self.assertTrue(any("marked for deprecation" in str(w.message) for w in ws))

        # Test that ModelFittingErrors are rethrown as warnings
        def mock_fit_gpytorch_mll(*args, **kwargs):
            raise ModelFittingError("foo")

        with catch_warnings(record=True) as ws, patch.object(
            fit, "fit_gpytorch_mll", new=mock_fit_gpytorch_mll
        ):
            fit.fit_gpytorch_model(self.mll)
        self.assertTrue(any("foo" in str(w.message) for w in ws))


class TestFitFallback(BotorchTestCase):
    def setUp(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.linspace(0, 1, 10).unsqueeze(-1)
            train_F = torch.sin(2 * math.pi * train_X)

            self.mlls = {}
            self.checkpoints = {}
            for model_type, output_dim in product([SingleTaskGP], [1, 2]):
                train_Y = train_F.repeat(1, output_dim)
                train_Y = train_Y + 0.1 * torch.randn_like(train_Y)
                model = model_type(
                    train_X=train_X,
                    train_Y=train_Y,
                    input_transform=Normalize(d=1),
                    outcome_transform=Standardize(m=output_dim),
                    **(
                        {}
                        if model_type is SingleTaskGP
                        else {"train_Yvar": torch.full_like(train_Y, 0.1)}
                    ),
                )
                self.assertIsInstance(model.covar_module.base_kernel, MaternKernel)
                model.covar_module.base_kernel.nu = 2.5

                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                for dtype in (torch.float32, torch.float64):
                    key = model_type, output_dim
                    self.mlls[key] = mll.to(dtype=dtype)
                    self.checkpoints[key] = {
                        k: TensorCheckpoint(
                            values=v.detach().clone(), device=v.device, dtype=v.dtype
                        )
                        for k, v in mll.state_dict().items()
                    }

    def test_main(self):
        for case, mll in self.mlls.items():
            self._test_main(mll, self.checkpoints[case])

    def test_warnings(self):
        for case, mll in self.mlls.items():
            self._test_warnings(mll, self.checkpoints[case])

    def test_exceptions(self):
        for case, mll in self.mlls.items():
            self._test_exceptions(mll, self.checkpoints[case])

    def _test_main(self, mll, ckpt):
        r"""Main test for `_fit_fallback`."""
        optimizer = MockOptimizer()
        optimizer.warnings = [
            WarningMessage("test_runtime_warning", RuntimeWarning, __file__, 0),
        ]
        for should_fail in (True, False):
            optimizer.call_count = 0
            with catch_warnings(), requires_grad_ctx(
                module=mll, assignments={"model.mean_module.constant": False}
            ), module_rollback_ctx(mll, checkpoint=ckpt):
                try:
                    fit._fit_fallback(
                        mll,
                        None,
                        None,
                        max_attempts=2,
                        optimizer=optimizer,
                        warning_handler=lambda w: not should_fail,
                    )
                except ModelFittingError:
                    failed = True
                else:
                    failed = False

                # Test control flow
                self.assertEqual(failed, should_fail)
                self.assertEqual(optimizer.call_count, 2 if should_fail else 1)

                # Test terminal state
                self.assertEqual(failed, mll.training)
                for key, vals in mll.state_dict().items():
                    if failed:
                        self.assertTrue(vals.equal(ckpt[key].values))
                    else:
                        try:
                            param = mll.get_parameter(key)
                            self.assertNotEqual(
                                param.equal(ckpt[key].values), param.requires_grad
                            )
                        except AttributeError:
                            pass

        # Test `closure_kwargs`
        with self.subTest("closure_kwargs"):
            mock_closure = MagicMock(side_effect=StopIteration("foo"))
            with self.assertRaisesRegex(StopIteration, "foo"):
                fit._fit_fallback(
                    mll, None, None, closure=mock_closure, closure_kwargs={"ab": "cd"}
                )
            mock_closure.assert_called_once_with(ab="cd")

    def _test_warnings(self, mll, ckpt):
        r"""Test warning handling for `_fit_fallback`."""
        optimizer = MockOptimizer(randomize_requires_grad=False)
        optimizer.warnings = [
            WarningMessage("test_runtime_warning", RuntimeWarning, __file__, 0),
            WarningMessage(MAX_ITER_MSG, OptimizationWarning, __file__, 0),
        ]

        warning_handlers = {
            "default": fit.DEFAULT_WARNING_HANDLER,
            "none": lambda w: False,
            "all": lambda w: True,
        }
        for case, warning_handler in warning_handlers.items():
            with (
                self.assertLogs(level="DEBUG") if case == "default" else nullcontext()
            ) as logs, catch_warnings(record=True) as ws, debug(True):
                try:
                    fit._fit_fallback(
                        mll,
                        None,
                        None,
                        max_attempts=2,
                        optimizer=optimizer,
                        warning_handler=warning_handler,
                    )
                except ModelFittingError:
                    failed = True
                else:
                    failed = False

                # Test that warnings were resolved in the expected fashion
                self.assertEqual(failed, case == "none")
                with catch_warnings(record=True) as rethrown:
                    unresolved = list(filterfalse(warning_handler, optimizer.warnings))
                    self.assertEqual(failed, len(unresolved) > 0)

                self.assertEqual(
                    {str(w.message) for w in ws},
                    {str(w.message) for w in rethrown + unresolved},
                )
                if logs:  # test that default filter logs certain warnings
                    self.assertTrue(any(MAX_ITER_MSG in log for log in logs.output))

        # Test default of retrying upon encountering an uncaught OptimizationWarning
        optimizer.warnings.append(
            WarningMessage("test_optim_warning", OptimizationWarning, __file__, 0)
        )

        with self.assertRaises(ModelFittingError), catch_warnings():
            fit._fit_fallback(
                mll,
                None,
                None,
                max_attempts=1,
                optimizer=optimizer,
            )

    def _test_exceptions(self, mll, ckpt):
        r"""Test exception handling for `_fit_fallback`."""
        optimizer = MockOptimizer(exception=NotPSDError("not_psd"))
        with catch_warnings():
            # Test behavior when encountering a caught exception
            with self.assertLogs(level="DEBUG") as logs, self.assertRaises(
                ModelFittingError
            ):
                fit._fit_fallback(
                    mll,
                    None,
                    None,
                    max_attempts=1,
                    optimizer=optimizer,
                )

            self.assertTrue(any("not_psd" in log for log in logs.output))
            self.assertTrue(  # test state rollback
                all(v.equal(ckpt[k].values) for k, v in mll.state_dict().items())
            )

            # Test behavior when encountering an uncaught exception
            with self.assertRaisesRegex(NotPSDError, "not_psd"):
                fit._fit_fallback(
                    mll,
                    None,
                    None,
                    max_attempts=1,
                    optimizer=optimizer,
                    caught_exception_types=(),
                )

            self.assertTrue(  # test state rollback
                all(v.equal(ckpt[k].values) for k, v in mll.state_dict().items())
            )


class TestFitFallbackAppoximate(BotorchTestCase):
    def setUp(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.linspace(0, 1, 10).unsqueeze(-1)
            train_F = torch.sin(2 * math.pi * train_X)
            train_Y = train_F + 0.1 * torch.randn_like(train_F)

            model = SingleTaskVariationalGP(
                train_X=train_X,
                train_Y=train_Y,
                input_transform=Normalize(d=1),
                outcome_transform=Standardize(m=1),
            )
            self.mll = mll = VariationalELBO(model.likelihood, model.model, num_data=10)
            self.data_loader = get_data_loader(mll.model, batch_size=1)
            self.closure = get_loss_closure_with_grads(
                mll=mll,
                parameters={n: p for n, p in mll.named_parameters() if p.requires_grad},
                data_loader=self.data_loader,
            )

    def test_main(self):
        # Test parameter updates
        with module_rollback_ctx(self.mll) as ckpt:
            fit._fit_fallback_approximate(
                self.mll,
                None,
                None,
                closure=self.closure,
                optimizer_kwargs={"step_limit": 3},
            )
            for name, param in self.mll.named_parameters():
                self.assertFalse(param.equal(ckpt[name].values))

        # Test dispatching pattern
        kwargs = {"full_batch_limit": float("inf")}
        with patch.object(fit, "_fit_fallback") as mock_fallback:
            fit._fit_fallback_approximate(self.mll, None, None, full_batch_limit=1)
            mock_fallback.assert_called_once_with(
                self.mll,
                None,
                None,
                closure=None,
                optimizer=fit_gpytorch_mll_torch,
            )

        with patch.object(fit, "_fit_fallback") as mock_fallback:
            fit._fit_fallback_approximate(self.mll, None, None, **kwargs)
            mock_fallback.assert_called_once_with(
                self.mll,
                None,
                None,
                closure=None,
                optimizer=fit_gpytorch_mll_scipy,
            )

        with patch.object(fit, "_fit_fallback") as mock_fallback:
            fit._fit_fallback_approximate(
                self.mll, None, None, closure=self.closure, **kwargs
            )

            mock_fallback.assert_called_once_with(
                self.mll,
                None,
                None,
                closure=self.closure,
                optimizer=fit_gpytorch_mll_torch,
            )

        with patch.object(fit, "_fit_fallback") as mock_fallback, patch.object(
            fit, "get_loss_closure_with_grads"
        ) as mock_get_closure:
            mock_get_closure.return_value = "foo"
            fit._fit_fallback_approximate(
                self.mll,
                None,
                None,
                data_loader=self.data_loader,
                **kwargs,
            )
            params = {n: p for n, p in self.mll.named_parameters() if p.requires_grad}
            mock_get_closure.assert_called_once_with(
                mll=self.mll,
                data_loader=self.data_loader,
                parameters=params,
            )
            mock_fallback.assert_called_once_with(
                self.mll,
                None,
                None,
                closure="foo",
                optimizer=fit_gpytorch_mll_torch,
            )

        # Test exception handling
        with self.assertRaisesRegex(
            UnsupportedError, "Only one of `data_loader` or `closure` may be passed."
        ):
            fit._fit_fallback_approximate(
                self.mll,
                None,
                None,
                closure=self.closure,
                data_loader=self.data_loader,
            )


class TestFitMultioutputIndependent(BotorchTestCase):
    def setUp(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.linspace(0, 1, 10).unsqueeze(-1)
            train_F = torch.sin(2 * math.pi * train_X)

        self.mlls = {}
        self.checkpoints = {}
        self.converted_mlls = {}
        for model_type, output_dim in product(
            [SingleTaskGP, FixedNoiseGP, HeteroskedasticSingleTaskGP], [1, 2]
        ):
            train_Y = train_F.repeat(1, output_dim)
            train_Y = train_Y + 0.1 * torch.randn_like(train_Y)
            model = model_type(
                train_X=train_X,
                train_Y=train_Y,
                input_transform=Normalize(d=1),
                outcome_transform=Standardize(m=output_dim),
                **(
                    {}
                    if model_type is SingleTaskGP
                    else {"train_Yvar": torch.full_like(train_Y, 0.1)}
                ),
            )
            self.assertIsInstance(model.covar_module.base_kernel, MaternKernel)
            model.covar_module.base_kernel.nu = 2.5

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            for dtype in (torch.float32, torch.float64):
                key = model_type, output_dim
                self.mlls[key] = mll.to(dtype=dtype).train()
                self.checkpoints[key] = {
                    k: TensorCheckpoint(
                        values=v.detach().clone(), device=v.device, dtype=v.dtype
                    )
                    for k, v in mll.state_dict().items()
                }
                if output_dim > 1:
                    with del_attribute_ctx(mll.model, "outcome_transform"):
                        _mll = self.converted_mlls[key] = deepcopy(mll)
                        _mll.model = deepcopy(mll.model)
                        _mll.model.covar_module.base_kernel.nu = 1.5  # break on purpose

    def test_main(self):
        for case, mll in self.mlls.items():
            self._test_main(mll, self.checkpoints[case])

    def test_unpack(self):
        for case, mll in self.mlls.items():
            if case in self.converted_mlls:
                self._test_unpack(
                    mll, self.checkpoints[case], self.converted_mlls[case]
                )

    def test_repack(self):
        for case, mll in self.mlls.items():
            if case in self.converted_mlls:
                self._test_repack(
                    mll, self.checkpoints[case], self.converted_mlls[case]
                )

    def test_exceptions(self):
        for case, mll in self.mlls.items():
            if case in self.converted_mlls:
                self._test_exceptions(
                    mll, self.checkpoints[case], self.converted_mlls[case]
                )

    def _test_main(self, mll, ckpt):
        # Test that ineligible models error out approriately, then short-circuit
        if mll.model.num_outputs == 1 or mll.likelihood is not getattr(
            mll.model, "likelihood", None
        ):
            with self.assertRaises(MDNotImplementedError):
                fit._fit_multioutput_independent(mll, None, None)

            return

        optimizer = MockOptimizer()
        with module_rollback_ctx(mll, checkpoint=ckpt), debug(
            True
        ), warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always", BotorchWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                fit._fit_multioutput_independent(
                    mll,
                    None,
                    None,
                    optimizer=optimizer,
                    warning_handler=lambda w: True,  # mark all warnings as resolved
                    max_attempts=1,
                )
            except Exception:
                pass  # exception handling tested separately
            else:
                self.assertEqual(0, len(ws))
                self.assertFalse(mll.training)
                self.assertEqual(optimizer.call_count, mll.model.num_outputs)
                self.assertTrue(
                    all(
                        v.equal(ckpt[k].values) != v.requires_grad
                        for k, v in mll.named_parameters()
                    )
                )

    def _test_unpack(self, mll, ckpt, bad_mll):
        # Test that model unpacking fails gracefully
        optimizer = MockOptimizer()
        converter = MagicMock(return_value=bad_mll.model)
        with patch.multiple(
            fit,
            batched_to_model_list=converter,
            SumMarginalLogLikelihood=MagicMock(return_value=bad_mll),
        ):
            with catch_warnings(record=True) as ws, debug(True):
                with self.assertRaises(MDNotImplementedError):
                    fit._fit_multioutput_independent(
                        mll, None, None, optimizer=optimizer, max_attempts=1
                    )

            self.assertEqual(converter.call_count, 1)
            self.assertEqual(optimizer.call_count, 0)  # should fail beforehand
            self.assertTrue(
                all(v.equal(ckpt[k].values) for k, v in mll.state_dict().items())
            )
            self.assertTrue(any("unpacked model differs" in str(w.message) for w in ws))

    def _test_repack(self, mll, ckpt, bad_mll):
        # Test that model repacking fails gracefully
        with patch.multiple(
            fit,  # skips unpacking + fitting, tests bad model repacking
            allclose_mll=lambda a, b, **kwargs: allclose_mll(a, b),
            batched_to_model_list=lambda model: model,
            SumMarginalLogLikelihood=MagicMock(return_value=mll),
            fit_gpytorch_mll=lambda mll, **kwargs: mll,
            model_list_to_batched=MagicMock(return_value=bad_mll.model),
        ):
            with catch_warnings(record=True) as ws, debug(True):
                with self.assertRaises(MDNotImplementedError):
                    fit._fit_multioutput_independent(mll, None, None, max_attempts=1)

            self.assertTrue(
                all(v.equal(ckpt[k].values) for k, v in mll.state_dict().items())
            )
            self.assertTrue(any("repacked model differs" in str(w.message) for w in ws))

    def _test_exceptions(self, mll, ckpt, bad_mll):
        for exception in (
            AttributeError("test_attribute_error"),
            RuntimeError("test_runtime_error"),
            UnsupportedError("test_unsupported_error"),
        ):
            converter = MagicMock(return_value=bad_mll.model)
            with catch_warnings(record=True) as ws, debug(True):

                def mock_fit_gpytorch_mll(*args, **kwargs):
                    raise exception

                try:
                    with patch.multiple(
                        fit,  # skip unpacking, throw exception from fit_gpytorch_mll
                        allclose_mll=lambda a, b, **kwargs: True,
                        batched_to_model_list=converter,
                        model_list_to_batched=converter,  # should not get called
                        fit_gpytorch_mll=mock_fit_gpytorch_mll,
                        SumMarginalLogLikelihood=type(mll),
                        module_rollback_ctx=lambda *args, **kwargs: nullcontext({}),
                    ):
                        fit._fit_multioutput_independent(mll, None, None)
                except MDNotImplementedError:
                    pass

            self.assertEqual(converter.call_count, 1)
            self.assertTrue(any(str(exception) in str(w.message) for w in ws))


class TestFitOther(BotorchTestCase):
    def test_fit_with_converter(self):
        # Check that sequential optimization using converter does not
        # break input transforms.
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            X = torch.rand(5, 2, **tkwargs) * 10
            Y = X**2
            intf = Normalize(2)
            model = SingleTaskGP(X, Y, input_transform=intf)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with patch(
                f"{fit_gpytorch_mll.__module__}.batched_to_model_list",
                wraps=batched_to_model_list,
            ) as wrapped_converter, warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always", BotorchWarning)
                fit_gpytorch_mll(mll)
            # Check that MLL repacking succeeded.
            self.assertFalse(
                any("Training loss of repacked model" in str(w.message) for w in ws)
            )
            wrapped_converter.assert_called_once()
            self.assertFalse(torch.allclose(intf.mins, torch.zeros(1, 2, **tkwargs)))
            self.assertFalse(torch.allclose(intf.ranges, torch.ones(1, 2, **tkwargs)))
