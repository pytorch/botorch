#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from botorch.posteriors import Posterior

from ..botorch_test_case import BotorchTestCase


class NotSoAbstractPosterior(Posterior):
    @property
    def device(self):
        pass

    @property
    def dtype(self):
        pass

    @property
    def event_shape(self):
        pass

    def rsample(self, *args):
        pass


class TestPosterior(BotorchTestCase):
    def test_abstract_base_posterior(self):
        with self.assertRaises(TypeError):
            Posterior()

    def test_mean_var_notimplemented_error(self):
        posterior = NotSoAbstractPosterior()
        with self.assertRaises(NotImplementedError) as e:
            posterior.mean
            self.assertIn("NotSoAbstractPosterior", str(e.exception))
        with self.assertRaises(NotImplementedError) as e:
            posterior.variance
            self.assertIn("NotSoAbstractPosterior", str(e.exception))
