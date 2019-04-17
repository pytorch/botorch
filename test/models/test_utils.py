#! /usr/bin/env python3

import unittest

import torch
from botorch.models.utils import add_output_dim, multioutput_to_batch_mode_transform


class TestMultiOutputToBatchModeTransform(unittest.TestCase):
    def test_multioutput_to_batch_mode_transform(self, cuda=False):
        for double in (False, True):
            tkwargs = {
                "device": torch.device("cuda") if cuda else torch.device("cpu"),
                "dtype": torch.double if double else torch.float,
            }
            n = 3
            num_outputs = 1
            train_X = torch.rand(n, 1, **tkwargs)
            train_Y = torch.rand(n, **tkwargs)
            train_Yvar = torch.rand(n, **tkwargs)
            # num_outputs = 1 and train_Y has shape `n`
            X_out, Y_out, Yvar_out = multioutput_to_batch_mode_transform(
                train_X=train_X,
                train_Y=train_Y,
                num_outputs=num_outputs,
                train_Yvar=train_Yvar,
            )
            self.assertTrue(torch.equal(X_out, train_X))
            self.assertTrue(torch.equal(Y_out, train_Y))
            self.assertTrue(torch.equal(Yvar_out, train_Yvar))
            # num_outputs = 1 and train_Y has shape `n x 1`
            X_out, Y_out, Yvar_out = multioutput_to_batch_mode_transform(
                train_X=train_X,
                train_Y=train_Y.view(-1, 1),
                num_outputs=num_outputs,
                train_Yvar=train_Yvar.view(-1, 1),
            )
            self.assertTrue(torch.equal(X_out, train_X))
            self.assertTrue(torch.equal(Y_out, train_Y))
            self.assertTrue(torch.equal(Yvar_out, train_Yvar))
            # num_outputs > 1
            num_outputs = 2
            train_Y = torch.rand(n, num_outputs, **tkwargs)
            train_Yvar = torch.rand(n, num_outputs, **tkwargs)
            X_out, Y_out, Yvar_out = multioutput_to_batch_mode_transform(
                train_X=train_X,
                train_Y=train_Y,
                num_outputs=num_outputs,
                train_Yvar=train_Yvar,
            )
            expected_X_out = train_X.unsqueeze(0).expand(num_outputs, -1, 1)
            self.assertTrue(torch.equal(X_out, expected_X_out))
            self.assertTrue(torch.equal(Y_out, train_Y.transpose(0, 1)))
            self.assertTrue(torch.equal(Yvar_out, train_Yvar.transpose(0, 1)))

    def test_multioutput_to_batch_mode_transform_cuda(self):
        if torch.cuda.is_available():
            self.test_multioutput_to_batch_mode_transform(cuda=True)


class TestAddOutputDim(unittest.TestCase):
    def test_add_output_dim(self, cuda=False):
        for double in (False, True):
            tkwargs = {
                "device": torch.device("cuda") if cuda else torch.device("cpu"),
                "dtype": torch.double if double else torch.float,
            }
            original_batch_shape = torch.Size([2])
            # check exception is raised
            X = torch.rand(2, 1, **tkwargs)
            with self.assertRaises(ValueError):
                add_output_dim(X=X, original_batch_shape=original_batch_shape)
            # test no new batch dims
            X = torch.rand(2, 2, 1, **tkwargs)
            X_out, output_dim_idx = add_output_dim(
                X=X, original_batch_shape=original_batch_shape
            )
            self.assertTrue(torch.equal(X_out, X.unsqueeze(0)))
            self.assertEqual(output_dim_idx, 0)
            # test new batch dims
            X = torch.rand(3, 2, 2, 1, **tkwargs)
            X_out, output_dim_idx = add_output_dim(
                X=X, original_batch_shape=original_batch_shape
            )
            self.assertTrue(torch.equal(X_out, X.unsqueeze(1)))
            self.assertEqual(output_dim_idx, 1)

    def test_add_output_dim_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_add_output_dim(cuda=True)
