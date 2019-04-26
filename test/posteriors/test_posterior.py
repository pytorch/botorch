#!/usr/bin/env python3

import unittest

from botorch.posteriors import Posterior


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


class TestPosterior(unittest.TestCase):
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
