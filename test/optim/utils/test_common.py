#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import partial
from warnings import catch_warnings, warn

import numpy as np
from botorch.optim.utils import _handle_numerical_errors, _warning_handler_template
from botorch.utils.testing import BotorchTestCase
from linear_operator.utils.errors import NanError, NotPSDError


class TestUtilsCommon(BotorchTestCase):
    def test_handle_numerical_errors(self):
        x = np.zeros(1, dtype=np.float64)

        with self.assertRaisesRegex(NotPSDError, "foo"):
            _handle_numerical_errors(NotPSDError("foo"), x=x)

        for error in (
            NanError(),
            RuntimeError("singular"),
            RuntimeError("input is not positive-definite"),
        ):
            fake_loss, fake_grad = _handle_numerical_errors(error, x=x)
            self.assertTrue(np.isnan(fake_loss))
            self.assertEqual(fake_grad.shape, x.shape)
            self.assertTrue(np.isnan(fake_grad).all())

        fake_loss, fake_grad = _handle_numerical_errors(error, x=x, dtype=np.float32)
        self.assertEqual(np.float32, fake_loss.dtype)
        self.assertEqual(np.float32, fake_grad.dtype)

        with self.assertRaisesRegex(RuntimeError, "foo"):
            _handle_numerical_errors(RuntimeError("foo"), x=x)

    def test_warning_handler_template(self):
        with catch_warnings(record=True) as ws:
            warn(DeprecationWarning("foo"))
            warn(RuntimeWarning("bar"))

        self.assertFalse(any(_warning_handler_template(w) for w in ws))
        handler = partial(
            _warning_handler_template,
            debug=lambda w: issubclass(w.category, DeprecationWarning),
            rethrow=lambda w: True,
        )
        with self.assertLogs(level="DEBUG") as logs, catch_warnings(record=True) as _ws:
            self.assertTrue(all(handler(w) for w in ws))
            self.assertEqual(1, len(logs.output))
            self.assertTrue("foo" in logs.output[0])
            self.assertEqual(1, len(_ws))
            self.assertEqual("bar", str(_ws[0].message))
