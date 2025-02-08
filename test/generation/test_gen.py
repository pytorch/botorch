#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
import warnings
from unittest import mock

import torch
from botorch.acquisition import qExpectedImprovement, qKnowledgeGradient
from botorch.exceptions.errors import (
    CandidateGenerationError,
    OptimizationGradientError,
)
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation.gen import (
    gen_candidates_scipy,
    gen_candidates_torch,
    get_best_candidates,
)
from botorch.models import SingleTaskGP
from botorch.utils.testing import BotorchTestCase, MockAcquisitionFunction
from gpytorch import settings as gpt_settings
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from scipy.optimize import OptimizeResult


EPS = 1e-8

NOISE = [
    [0.127],
    [-0.113],
    [-0.345],
    [-0.034],
    [-0.069],
    [-0.272],
    [0.013],
    [0.056],
    [0.087],
    [-0.081],
]


class TestBaseCandidateGeneration(BotorchTestCase):
    def _setUp(self, double=False, expand=False):
        dtype = torch.double if double else torch.float
        train_x = torch.linspace(0, 1, 10, device=self.device, dtype=dtype).unsqueeze(
            -1
        )
        train_y = torch.sin(train_x * (2 * math.pi))
        noise = torch.tensor(NOISE, device=self.device, dtype=dtype)
        self.train_x = train_x
        self.train_y = train_y + noise
        if expand:
            self.train_x = self.train_x.expand(-1, 2)
            ics = torch.tensor([[0.5, 1.0]], device=self.device, dtype=dtype)
        else:
            ics = torch.tensor([[0.5]], device=self.device, dtype=dtype)
        self.initial_conditions = ics
        self.f_best = self.train_y.max().item()
        model = SingleTaskGP(self.train_x, self.train_y)
        self.model = model.to(device=self.device, dtype=dtype)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        with warnings.catch_warnings():
            self.mll = fit_gpytorch_mll(
                self.mll, optimizer_kwargs={"options": {"maxiter": 1}}, max_attempts=1
            )


class TestGenCandidates(TestBaseCandidateGeneration):
    def test_gen_candidates(
        self, gen_candidates=gen_candidates_scipy, options=None, timeout_sec=None
    ):
        options = options or {}
        options = {**options, "maxiter": options.get("maxiter", 5)}
        for double in (True, False):
            self._setUp(double=double)
            acqfs = [
                qExpectedImprovement(self.model, best_f=self.f_best),
                qKnowledgeGradient(
                    self.model, num_fantasies=4, current_value=self.f_best
                ),
            ]
            for acqf in acqfs:
                ics = self.initial_conditions
                if isinstance(acqf, qKnowledgeGradient):
                    ics = ics.repeat(5, 1)

                kwargs = {
                    "initial_conditions": ics,
                    "acquisition_function": acqf,
                    "lower_bounds": 0,
                    "upper_bounds": 1,
                    "options": options or {},
                    "timeout_sec": timeout_sec,
                }
                if gen_candidates is gen_candidates_torch:
                    kwargs["callback"] = mock.MagicMock()
                candidates, _ = gen_candidates(**kwargs)

                if isinstance(acqf, qKnowledgeGradient):
                    candidates = acqf.extract_candidates(candidates)
                if gen_candidates is gen_candidates_torch:
                    self.assertTrue(kwargs["callback"].call_count > 0)

                self.assertTrue(-EPS <= candidates <= 1 + EPS)

    def test_gen_candidates_torch(self):
        self.test_gen_candidates(gen_candidates=gen_candidates_torch)

    def test_gen_candidates_with_none_fixed_features(
        self,
        gen_candidates=gen_candidates_scipy,
        options=None,
    ):
        options = options or {}
        options = {**options, "maxiter": 5}
        for double in (True, False):
            self._setUp(double=double, expand=True)
            acqfs = [
                qExpectedImprovement(self.model, best_f=self.f_best),
                qKnowledgeGradient(
                    self.model, num_fantasies=4, current_value=self.f_best
                ),
            ]
            for acqf in acqfs:
                ics = self.initial_conditions
                if isinstance(acqf, qKnowledgeGradient):
                    ics = ics.repeat(5, 1)
                candidates, _ = gen_candidates(
                    initial_conditions=ics,
                    acquisition_function=acqf,
                    lower_bounds=0,
                    upper_bounds=1,
                    fixed_features={1: None},
                    options=options or {},
                )
                if isinstance(acqf, qKnowledgeGradient):
                    candidates = acqf.extract_candidates(candidates)
                candidates = candidates.squeeze(0)
                self.assertTrue(-EPS <= candidates[0] <= 1 + EPS)
                self.assertTrue(candidates[1].item() == 1.0)

    def test_gen_candidates_torch_with_none_fixed_features(self):
        self.test_gen_candidates_with_none_fixed_features(
            gen_candidates=gen_candidates_torch
        )

    def test_gen_candidates_with_fixed_features(
        self, gen_candidates=gen_candidates_scipy, options=None, timeout_sec=None
    ):
        options = options or {}
        options = {**options, "maxiter": 5}
        for double in (True, False):
            self._setUp(double=double, expand=True)
            acqfs = [
                qExpectedImprovement(self.model, best_f=self.f_best),
                qKnowledgeGradient(
                    self.model, num_fantasies=4, current_value=self.f_best
                ),
            ]
            for acqf in acqfs:
                ics = self.initial_conditions
                if isinstance(acqf, qKnowledgeGradient):
                    ics = ics.repeat(5, 1)
                candidates, _ = gen_candidates(
                    initial_conditions=ics,
                    acquisition_function=acqf,
                    lower_bounds=0,
                    upper_bounds=1,
                    fixed_features={1: 0.25},
                    options=options,
                    timeout_sec=timeout_sec,
                )

                if isinstance(acqf, qKnowledgeGradient):
                    candidates = acqf.extract_candidates(candidates)

                candidates = candidates.squeeze(0)
                self.assertTrue(-EPS <= candidates[0] <= 1 + EPS)
                self.assertTrue(candidates[1].item() == 0.25)

    def test_gen_candidates_with_fixed_features_and_timeout(self):
        with self.assertLogs("botorch", level="INFO") as logs:
            self.test_gen_candidates_with_fixed_features(
                timeout_sec=1e-4,
                options={"disp": False},
            )
        self.assertTrue(any("Optimization timed out" in o for o in logs.output))

    def test_gen_candidates_torch_with_fixed_features(self):
        self.test_gen_candidates_with_fixed_features(
            gen_candidates=gen_candidates_torch
        )

    def test_gen_candidates_torch_with_fixed_features_and_timeout(self):
        with self.assertLogs("botorch", level="INFO") as logs:
            self.test_gen_candidates_with_fixed_features(
                gen_candidates=gen_candidates_torch,
                timeout_sec=1e-4,
            )
        self.assertTrue(any("Optimization timed out" in o for o in logs.output))

    def test_gen_candidates_scipy_with_fixed_features_inequality_constraints(self):
        options = {"maxiter": 5}
        for double in (True, False):
            self._setUp(double=double, expand=True)
            qEI = qExpectedImprovement(self.model, best_f=self.f_best)
            candidates, _ = gen_candidates_scipy(
                initial_conditions=self.initial_conditions.reshape(1, 1, -1),
                acquisition_function=qEI,
                inequality_constraints=[
                    (
                        torch.tensor([0], device=self.device),
                        torch.tensor([1], device=self.device),
                        0,
                    ),
                    (
                        torch.tensor([1], device=self.device),
                        torch.tensor([-1], device=self.device),
                        -1,
                    ),
                ],
                fixed_features={1: 0.25},
                options=options,
            )
            # candidates is of dimension 1 x 1 x 2
            # so we are squeezing all the singleton dimensions
            candidates = candidates.squeeze()
            self.assertTrue(-EPS <= candidates[0] <= 1 + EPS)
            self.assertTrue(candidates[1].item() == 0.25)

    def test_gen_candidates_scipy_warns_opt_failure(self):
        with warnings.catch_warnings(record=True) as ws:
            self.test_gen_candidates(options={"maxls": 1})
        expected_msg = re.compile(
            # The message changed with scipy 1.15, hence the different matching here.
            "Optimization failed within `scipy.optimize.minimize` with status 2"
            " and message ABNORMAL(|_TERMINATION_IN_LNSRCH)."
        )
        expected_warning_raised = any(
            issubclass(w.category, OptimizationWarning)
            and expected_msg.search(str(w.message))
            for w in ws
        )
        self.assertTrue(expected_warning_raised)

    def test_gen_candidates_scipy_maxiter_behavior(self):
        # Check that no warnings are raised & log produced on hitting maxiter.
        for method in ("SLSQP", "L-BFGS-B"):
            with warnings.catch_warnings(record=True) as ws, self.assertLogs(
                "botorch", level="INFO"
            ) as logs:
                self.test_gen_candidates(options={"maxiter": 1, "method": method})
            self.assertFalse(
                any(issubclass(w.category, OptimizationWarning) for w in ws)
            )
            self.assertTrue("iteration limit" in logs.output[-1])
        # Check that we handle maxfun as well.
        with warnings.catch_warnings(record=True) as ws, self.assertLogs(
            "botorch", level="INFO"
        ) as logs:
            self.test_gen_candidates(
                options={"maxiter": 100, "maxfun": 1, "method": "L-BFGS-B"}
            )
        self.assertFalse(any(issubclass(w.category, OptimizationWarning) for w in ws))
        self.assertTrue("function evaluation limit" in logs.output[-1])

    def test_gen_candidates_scipy_timeout_behavior(self):
        # Check that no warnings are raised & log produced on hitting timeout.
        for method in ("SLSQP", "L-BFGS-B"):
            with warnings.catch_warnings(record=True) as ws, self.assertLogs(
                "botorch", level="INFO"
            ) as logs:
                self.test_gen_candidates(options={"method": method}, timeout_sec=0.001)
            self.assertFalse(
                any(issubclass(w.category, OptimizationWarning) for w in ws)
            )
            self.assertTrue("Optimization timed out" in logs.output[-1])

    def test_gen_candidates_torch_timeout_behavior(self):
        # Check that no warnings are raised & log produced on hitting timeout.
        with warnings.catch_warnings(record=True) as ws, self.assertLogs(
            "botorch", level="INFO"
        ) as logs:
            self.test_gen_candidates(
                gen_candidates=gen_candidates_torch, timeout_sec=0.001
            )
        self.assertFalse(any(issubclass(w.category, OptimizationWarning) for w in ws))
        self.assertTrue("Optimization timed out" in logs.output[-1])

    def test_gen_candidates_scipy_warns_opt_no_res(self):
        ckwargs = {"dtype": torch.float, "device": self.device}

        test_ics = torch.rand(3, 1, **ckwargs)
        expected_msg = (
            "Optimization failed within `scipy.optimize.minimize` with no "
            "status returned to `res.`"
        )
        with mock.patch(
            "botorch.generation.gen.minimize_with_timeout"
        ) as mock_minimize, warnings.catch_warnings(record=True) as ws:
            mock_minimize.return_value = OptimizeResult(x=test_ics.cpu().numpy())

            gen_candidates_scipy(
                initial_conditions=test_ics,
                acquisition_function=MockAcquisitionFunction(),
            )
        expected_warning_raised = any(
            issubclass(w.category, OptimizationWarning)
            and expected_msg in str(w.message)
            for w in ws
        )
        self.assertTrue(expected_warning_raised)

    def test_gen_candidates_scipy_nan_handling(self):
        for dtype, expected_regex in [
            (torch.float, "Consider using"),
            (torch.double, "gradient array"),
        ]:
            ckwargs = {"dtype": dtype, "device": self.device}

            test_ics = torch.rand(3, 1, **ckwargs)
            test_grad = torch.tensor([0.5, 0.2, float("nan")], **ckwargs)
            # test NaN in grad
            with mock.patch("torch.autograd.grad", return_value=[test_grad]):
                with self.assertRaisesRegex(OptimizationGradientError, expected_regex):
                    gen_candidates_scipy(
                        initial_conditions=test_ics,
                        acquisition_function=mock.Mock(return_value=test_ics),
                    )

            # test NaN in `x`
            test_ics = torch.tensor([0.0, 0.0, float("nan")], **ckwargs)
            with self.assertRaisesRegex(RuntimeError, "array `x` are NaN."):
                gen_candidates_scipy(
                    initial_conditions=test_ics,
                    acquisition_function=mock.Mock(),
                )

    def test_gen_candidates_without_grad(self) -> None:
        """Test with `with_grad=False` (not supported for gen_candidates_torch)."""

        self.test_gen_candidates(
            gen_candidates=gen_candidates_scipy,
            options={"disp": False, "with_grad": False},
        )

        self.test_gen_candidates_with_fixed_features(
            gen_candidates=gen_candidates_scipy,
            options={"disp": False, "with_grad": False},
        )

        self.test_gen_candidates_with_none_fixed_features(
            gen_candidates=gen_candidates_scipy,
            options={"disp": False, "with_grad": False},
        )

    def test_gen_candidates_scipy_invalid_method(self) -> None:
        """Test with method that doesn't support constraint / bounds."""
        self._setUp(double=True, expand=True)
        acqf = qExpectedImprovement(self.model, best_f=self.f_best)
        with self.assertRaisesRegex(
            RuntimeWarning,
            "Method L-BFGS-B cannot handle constraints",
        ):
            gen_candidates_scipy(
                initial_conditions=self.initial_conditions,
                acquisition_function=acqf,
                options={"method": "L-BFGS-B"},
                inequality_constraints=[
                    (torch.tensor([0]), torch.tensor([1]), 0),
                    (torch.tensor([1]), torch.tensor([-1]), -1),
                ],
            )
        with self.assertRaisesRegex(
            RuntimeWarning,
            "Method Newton-CG cannot handle bounds",
        ):
            gen_candidates_scipy(
                initial_conditions=self.initial_conditions,
                acquisition_function=acqf,
                options={"method": "Newton-CG"},
                lower_bounds=0,
                upper_bounds=1,
            )

    def test_gen_candidates_scipy_infeasible_candidates(self) -> None:
        # Check for error when infeasible candidates are generated.
        ics = torch.rand(2, 3, 1, device=self.device)
        with mock.patch(
            "botorch.generation.gen.minimize_with_timeout",
            return_value=OptimizeResult(x=ics.view(-1).cpu().numpy()),
        ), self.assertRaisesRegex(
            CandidateGenerationError, "infeasible candidates. 2 out of 2"
        ):
            gen_candidates_scipy(
                initial_conditions=ics,
                acquisition_function=MockAcquisitionFunction(),
                inequality_constraints=[
                    (  # X[..., 0] >= 2.0, which is infeasible.
                        torch.tensor([0], device=self.device),
                        torch.tensor([1.0], device=self.device),
                        2.0,
                    )
                ],
            )


class TestRandomRestartOptimization(TestBaseCandidateGeneration):
    def test_random_restart_optimization(self):
        for double in (True, False):
            self._setUp(double=double)
            with gpt_settings.debug(False):
                best_f = self.model(self.train_x).mean.max().item()
            qEI = qExpectedImprovement(self.model, best_f=best_f)
            bounds = torch.tensor([[0.0], [1.0]]).type_as(self.train_x)
            batch_ics = torch.rand(2, 1).type_as(self.train_x)
            batch_candidates, batch_acq_values = gen_candidates_scipy(
                initial_conditions=batch_ics,
                acquisition_function=qEI,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                options={"maxiter": 3},
            )
            candidates = get_best_candidates(
                batch_candidates=batch_candidates, batch_values=batch_acq_values
            )
            self.assertTrue(-EPS <= candidates <= 1 + EPS)
