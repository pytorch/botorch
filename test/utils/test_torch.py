#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import OrderedDict

import torch
from botorch.utils.testing import BotorchTestCase
from botorch.utils.torch import BufferDict


class TestBufferDict(BotorchTestCase):
    def test_BufferDict(self):
        buffers = OrderedDict(
            [
                ("b1", torch.randn(10, 10)),
                ("b2", torch.randn(10, 10)),
                ("b3", torch.randn(10, 10)),
            ]
        )

        buffer_dict = BufferDict(buffers)

        def check():
            self.assertEqual(len(buffer_dict), len(buffers))
            for k1, m2 in zip(buffers, buffer_dict.buffers()):
                self.assertIs(buffers[k1], m2)
            for k1, k2 in zip(buffers, buffer_dict):
                self.assertIs(buffers[k1], buffer_dict[k2])
            for k in buffer_dict:
                self.assertIs(buffer_dict[k], buffers[k])
            for k in buffer_dict.keys():
                self.assertIs(buffer_dict[k], buffers[k])
            for k, v in buffer_dict.items():
                self.assertIs(v, buffers[k])
            for k1, m2 in zip(buffers, buffer_dict.values()):
                self.assertIs(buffers[k1], m2)
            for k in buffers.keys():
                self.assertTrue(k in buffer_dict)

        check()

        buffers["b4"] = torch.randn(10, 10)
        buffer_dict["b4"] = buffers["b4"]
        check()

        next_buffers = [("b5", torch.randn(10, 10)), ("b2", torch.randn(10, 10))]
        buffers.update(next_buffers)
        buffer_dict.update(next_buffers)
        check()

        next_buffers = OrderedDict(
            [("b6", torch.randn(10, 10)), ("b5", torch.randn(10, 10))]
        )
        buffers.update(next_buffers)
        buffer_dict.update(next_buffers)
        check()

        next_buffers = {"b8": torch.randn(10, 10), "b7": torch.randn(10, 10)}
        buffers.update(sorted(next_buffers.items()))
        buffer_dict.update(next_buffers)
        check()

        del buffer_dict["b3"]
        del buffers["b3"]
        check()

        with self.assertRaises(TypeError):
            buffer_dict.update(1)

        with self.assertRaises(TypeError):
            buffer_dict.update([1])

        with self.assertRaises(ValueError):
            buffer_dict.update(torch.randn(10, 10))

        with self.assertRaises(TypeError):
            buffer_dict[1] = torch.randn(10, 10)

        p_pop = buffer_dict.pop("b4")
        self.assertIs(p_pop, buffers["b4"])
        buffers.pop("b4")
        check()

        buffer_dict.clear()
        self.assertEqual(len(buffer_dict), 0)
        buffers.clear()
        check()

        # test extra repr
        buffer_dict = BufferDict(
            OrderedDict(
                [
                    ("b1", torch.randn(10, 10)),
                    ("b2", torch.randn(10, 10)),
                    ("b3", torch.randn(10, 10)),
                ]
            )
        )
        self.assertEqual(
            buffer_dict.extra_repr(),
            "  (b1): Buffer containing: [torch.FloatTensor of size 10x10]\n"
            "  (b2): Buffer containing: [torch.FloatTensor of size 10x10]\n"
            "  (b3): Buffer containing: [torch.FloatTensor of size 10x10]",
        )
        # test that calling a buffer dict raises an exception
        with self.assertRaises(RuntimeError):
            buffer_dict(1)
