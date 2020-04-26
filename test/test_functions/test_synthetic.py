#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from botorch.test_functions.synthetic import (
    Ackley,
    Beale,
    Branin,
    Bukin,
    Cosine8,
    DixonPrice,
    DropWave,
    EggHolder,
    Griewank,
    Hartmann,
    HolderTable,
    Levy,
    Michalewicz,
    Powell,
    Rastrigin,
    Rosenbrock,
    Shekel,
    SixHumpCamel,
    StyblinskiTang,
    ThreeHumpCamel,
)
from botorch.utils.testing import BotorchTestCase, SyntheticTestFunctionBaseTestCase


class TestAckley(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [Ackley(), Ackley(negate=True), Ackley(noise_std=0.1), Ackley(dim=3)]


class TestBeale(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [Beale(), Beale(negate=True), Beale(noise_std=0.1)]


class TestBranin(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [Branin(), Branin(negate=True), Branin(noise_std=0.1)]


class TestBukin(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [Bukin(), Bukin(negate=True), Bukin(noise_std=0.1)]


class TestCosine8(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [Cosine8(), Cosine8(negate=True), Cosine8(noise_std=0.1)]


class TestDropWave(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [DropWave(), DropWave(negate=True), DropWave(noise_std=0.1)]


class TestDixonPrice(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [
        DixonPrice(),
        DixonPrice(negate=True),
        DixonPrice(noise_std=0.1),
        DixonPrice(dim=3),
    ]


class TestEggHolder(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [EggHolder(), EggHolder(negate=True), EggHolder(noise_std=0.1)]


class TestGriewank(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [
        Griewank(),
        Griewank(negate=True),
        Griewank(noise_std=0.1),
        Griewank(dim=4),
    ]


class TestHartmann(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [
        Hartmann(),
        Hartmann(negate=True),
        Hartmann(noise_std=0.1),
        Hartmann(dim=3),
        Hartmann(dim=3, negate=True),
        Hartmann(dim=3, noise_std=0.1),
        Hartmann(dim=4),
        Hartmann(dim=4, negate=True),
        Hartmann(dim=4, noise_std=0.1),
    ]

    def test_dimension(self):
        with self.assertRaises(ValueError):
            Hartmann(dim=2)


class TestHolderTable(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [HolderTable(), HolderTable(negate=True), HolderTable(noise_std=0.1)]


class TestLevy(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [
        Levy(),
        Levy(negate=True),
        Levy(noise_std=0.1),
        Levy(dim=3),
        Levy(dim=3, negate=True),
        Levy(dim=3, noise_std=0.1),
    ]


class TestMichalewicz(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [
        Michalewicz(),
        Michalewicz(negate=True),
        Michalewicz(noise_std=0.1),
        Michalewicz(dim=5),
        Michalewicz(dim=5, negate=True),
        Michalewicz(dim=5, noise_std=0.1),
        Michalewicz(dim=10),
        Michalewicz(dim=10, negate=True),
        Michalewicz(dim=10, noise_std=0.1),
    ]


class TestPowell(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [Powell(), Powell(negate=True), Powell(noise_std=0.1)]


class TestRastrigin(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [
        Rastrigin(),
        Rastrigin(negate=True),
        Rastrigin(noise_std=0.1),
        Rastrigin(dim=3),
        Rastrigin(dim=3, negate=True),
        Rastrigin(dim=3, noise_std=0.1),
    ]


class TestRosenbrock(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [
        Rosenbrock(),
        Rosenbrock(negate=True),
        Rosenbrock(noise_std=0.1),
        Rosenbrock(dim=3),
        Rosenbrock(dim=3, negate=True),
        Rosenbrock(dim=3, noise_std=0.1),
    ]


class TestShekel(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [Shekel(), Shekel(negate=True), Shekel(noise_std=0.1)]


class TestSixHumpCamel(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [SixHumpCamel(), SixHumpCamel(negate=True), SixHumpCamel(noise_std=0.1)]


class TestStyblinskiTang(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [
        StyblinskiTang(),
        StyblinskiTang(negate=True),
        StyblinskiTang(noise_std=0.1),
        StyblinskiTang(dim=3),
        StyblinskiTang(dim=3, negate=True),
        StyblinskiTang(dim=3, noise_std=0.1),
    ]


class TestThreeHumpCamel(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [
        ThreeHumpCamel(),
        ThreeHumpCamel(negate=True),
        ThreeHumpCamel(noise_std=0.1),
    ]
