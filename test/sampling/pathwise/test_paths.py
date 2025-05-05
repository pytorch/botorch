#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.pathwise.paths import PathDict, PathList, SamplePath
from botorch.utils.testing import BotorchTestCase
from torch.nn import ModuleDict, ModuleList


class IdentityPath(SamplePath):
    """Simple path that returns input unchanged, used for testing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestGenericPaths(BotorchTestCase):
    def test_path_dict(self):
        """Test PathDict functionality including:
        - Initialization with different path types
        - Forward pass with and without reducer
        - Dictionary-like operations
        - Error handling for invalid configurations
        """
        # Test error when output_transform provided without reducer
        with self.assertRaisesRegex(
            UnsupportedError, "must be preceded by a `reducer`"
        ):
            PathDict(output_transform="foo")

        # Create test paths
        A = IdentityPath()
        B = IdentityPath()

        # Test initialization with dict vs ModuleList
        module_dict = ModuleDict({"0": A, "1": B})
        path_dict = PathDict(paths={"0": A, "1": B})
        # Verify new ModuleDict is created
        self.assertTrue(path_dict._paths_dict is not module_dict)

        # Test initialization with existing ModuleDict
        path_dict = PathDict(paths=module_dict)
        # Verify existing ModuleDict is reused
        self.assertIs(path_dict._paths_dict, module_dict)

        # Test forward pass without reducer
        x = torch.rand(3, device=self.device)
        output = path_dict(x)
        self.assertIsInstance(output, dict)
        # Verify each path returns input unchanged
        self.assertTrue(x.equal(output.pop("0")))
        self.assertTrue(x.equal(output.pop("1")))
        self.assertTrue(not output)

        # Test forward pass with reducer
        path_dict.reducer = torch.stack
        output = path_dict(x)
        self.assertIsInstance(output, torch.Tensor)
        # Verify stacked output shape and values
        self.assertEqual(output.shape, (2,) + x.shape)
        self.assertTrue(output.eq(x).all())

        # Test dictionary operations
        self.assertEqual(len(path_dict), 2)
        # Verify consistent behavior across different access methods
        for key, val, (key_0, val_0), (key_1, val_1), key_2 in zip(
            path_dict,
            path_dict.values(),
            path_dict.items(),
            path_dict._paths_dict.items(),
            path_dict.keys(),
        ):
            self.assertEqual(1, len({key, key_0, key_1, key_2}))
            self.assertEqual(1, len({val, val_0, val_1, path_dict[key]}))

        # Test item assignment
        path_dict["1"] = A  # test __setitem__
        self.assertIs(path_dict._paths_dict["1"], A)

        # Test item deletion
        del path_dict["1"]  # test __delitem__
        self.assertEqual(("0",), tuple(path_dict))

    def test_path_list(self):
        """Test PathList functionality including:
        - Initialization with different path types
        - Forward pass with and without reducer
        - List-like operations
        - Error handling for invalid configurations
        """
        # Test error when output_transform provided without reducer
        with self.assertRaisesRegex(
            UnsupportedError, "must be preceded by a `reducer`"
        ):
            PathList(output_transform="foo")

        # Create test paths
        A = IdentityPath()
        B = IdentityPath()

        # Test initialization with list vs ModuleList
        module_list = ModuleList((A, B))
        path_list = PathList(paths=list(module_list))
        # Verify new ModuleList is created
        self.assertTrue(path_list._paths_list is not module_list)

        # Test initialization with existing ModuleList
        path_list = PathList(paths=module_list)
        # Verify existing ModuleList is reused
        self.assertIs(path_list._paths_list, module_list)

        # Test forward pass without reducer
        x = torch.rand(3, device=self.device)
        output = path_list(x)
        self.assertIsInstance(output, list)
        # Verify each path returns input unchanged
        self.assertTrue(x.equal(output.pop()))
        self.assertTrue(x.equal(output.pop()))
        self.assertTrue(not output)

        # Test forward pass with reducer
        path_list.reducer = torch.stack
        output = path_list(x)
        self.assertIsInstance(output, torch.Tensor)
        # Verify stacked output shape and values
        self.assertEqual(output.shape, (2,) + x.shape)
        self.assertTrue(output.eq(x).all())

        # Test list operations
        self.assertEqual(len(path_list), 2)
        # Verify consistent behavior across different access methods
        for key, (path, path_0) in enumerate(zip(path_list, path_list._paths_list)):
            self.assertEqual(1, len({path, path_0, path_list[key]}))

        # Test item assignment
        path_list[1] = A  # test __setitem__
        self.assertIs(path_list._paths_list[1], A)

        # Test item deletion
        del path_list[1]  # test __delitem__
        self.assertEqual((A,), tuple(path_list))
