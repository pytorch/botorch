#! /usr/bin/env python3

import unittest

import torch


class TestQBatchInitialization(unittest.TestCase):
    def test_initialize_q_batch(self, cuda=False):
        # TODO (T38994767): Implement!
        pass

    def test_initialize_q_batch_cuda(self):
        if torch.cuda.is_available():
            self.test_initialize_q_batch(cuda=True)


if __name__ == "__main__":
    unittest.main()
