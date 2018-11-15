#! /usr/bin/env python3

import unittest
from test.utils.mock import MockBatchAcquisitionModule

import torch
from botorch.optim.initializers import q_batch_initialization


class TestQBatchInitialization(unittest.TestCase):
    def setUp(self):
        self.used = 0

    def test_q_batch_initialization(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")

        def fake_gen(n):
            lb = self.used * 2 * n * 1.0
            ub = (self.used + 1) * 2 * n * 1.0
            self.used = self.used + 1
            return torch.arange(lb, ub, device=device).reshape(n, 2)

        fake_acq = MockBatchAcquisitionModule()

        q = 2
        R = fake_gen(q)
        self.used = 0
        X = q_batch_initialization(
            gen_function=fake_gen, acq_function=fake_acq, q=q, multiplier=1
        )
        self.used = 0
        self.assertTrue(torch.equal(X.squeeze(0), R))

        X = q_batch_initialization(
            gen_function=fake_gen, acq_function=fake_acq, q=q, multiplier=20
        )
        self.used = 0
        self.assertTrue(
            torch.equal(
                X.squeeze(0),
                torch.tensor([[76, 77], [78, 79]], dtype=torch.float, device=device),
            )
        )

        X = q_batch_initialization(
            gen_function=fake_gen,
            acq_function=fake_acq,
            q=q,
            multiplier=20,
            torch_batches=2,
        )
        self.used = 0
        self.assertTrue(
            torch.equal(
                X,
                torch.tensor(
                    [[[76, 77], [78, 79]], [[72, 73], [74, 75]]],
                    dtype=torch.float,
                    device=device,
                ),
            )
        )

    def test_q_batch_initialization_cuda(self):
        if torch.cuda.is_available():
            self.test_q_batch_initialization(cuda=True)


if __name__ == "__main__":
    unittest.main()
