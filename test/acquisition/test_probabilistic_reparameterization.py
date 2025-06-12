import itertools
from typing import Any

import torch
from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
from botorch.acquisition.probabilistic_reparameterization import (
    AnalyticProbabilisticReparameterization,
    MCProbabilisticReparameterization,
)
from botorch.generation.gen import gen_candidates_scipy, gen_candidates_torch
from botorch.models.transforms.factory import (
    get_probabilistic_reparameterization_input_transform,
    get_rounding_input_transform,
)
from botorch.models.transforms.input import OneHotToNumeric
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.test_functions.synthetic import Ackley, AckleyMixed
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.test_helpers import get_model
from botorch.utils.testing import BotorchTestCase


def get_categorical_features_dict(feature_to_num_categories: dict[int, int]):
    r"""Get the mapping of starting index in one-hot space to cardinality.

    This mapping is used to construct the OneHotToNumeric transform. This
    requires that all of the categorical parameters are the rightmost elements.

    Args:
        feature_to_num_categories: Mapping of feature index to cardinality in the
            untransformed space.

    """
    start = None
    categorical_features = {}
    for idx, cardinality in sorted(
        feature_to_num_categories.items(), key=lambda kv: kv[0]
    ):
        if start is None:
            start = idx
        categorical_features[start] = cardinality
        # add cardinality to start
        start += cardinality
    return categorical_features


class TestProbabilisticReparameterizationInputTransform(BotorchTestCase):
    def test_probabilistic_reparameterization_input_transform(self):
        # TODO: test this functionality in factory
        # test the actual transform here.
        _ = get_probabilistic_reparameterization_input_transform()


class TestProbabilisticReparameterization(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.tkwargs: dict[str, Any] = {"device": self.device, "dtype": torch.double}

    def test_probabilistic_reparameterization_binary(
        self,
        base_acq_func_cls=LogExpectedImprovement,
    ):
        torch.manual_seed(0)
        f = AckleyMixed(dim=6, randomize_optimum=True)
        train_X = torch.rand((10, f.dim), **self.tkwargs)
        train_X[:, f.discrete_inds] = train_X[:, f.discrete_inds].round()
        train_Y = f(train_X).unsqueeze(-1)
        model = get_model(train_X, train_Y)
        base_acq_func = base_acq_func_cls(model, best_f=train_Y.max())

        pr_acq_func_params = dict(
            acq_function=base_acq_func,
            one_hot_bounds=f.bounds,
            integer_indices=f.discrete_inds,
            batch_limit=32,
        )
        optimize_acqf_params = dict(
            bounds=f.bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={
                "batch_limit": 5,
                "maxiter": 200,
                "rel_tol": float("-inf"),
            },
        )

        pr_analytic_acq_func = AnalyticProbabilisticReparameterization(
            **pr_acq_func_params
        )

        pr_mc_acq_func = MCProbabilisticReparameterization(**pr_acq_func_params)

        candidate_analytic, acq_values_analytic = optimize_acqf(
            acq_function=pr_analytic_acq_func,
            gen_candidates=gen_candidates_scipy,
            **optimize_acqf_params,
        )

        candidate_mc, acq_values_mc = optimize_acqf(
            acq_function=pr_mc_acq_func,
            gen_candidates=gen_candidates_torch,
            **optimize_acqf_params,
        )

        fixed_features_list = [
            {feat_dim: val for feat_dim, val in enumerate(vals)}
            for vals in itertools.product([0, 1], repeat=len(f.discrete_inds))
        ]
        candidate_exhaustive, acq_values_exhaustive = optimize_acqf_mixed(
            acq_function=base_acq_func,
            fixed_features_list=fixed_features_list,
            **optimize_acqf_params,
        )

        self.assertTrue(candidate_analytic.shape == (1, f.dim))
        self.assertTrue(candidate_mc.shape == (1, f.dim))
        self.assertAllClose(candidate_analytic, candidate_mc)
        self.assertAllClose(acq_values_analytic, acq_values_mc)

    def test_probabilistic_reparameterization_binary_qLogEI(self):
        self.test_probabilistic_reparameterization_binary(
            base_acq_func_cls=qLogExpectedImprovement,
        )

    def test_probabilistic_reparameterization_categorical(
        self,
        pr_acq_func_cls=AnalyticProbabilisticReparameterization,
        base_acq_func_cls=LogExpectedImprovement,
    ):
        torch.manual_seed(0)
        # we use Ackley here to ensure the categorical features are the
        # rightmost elements
        dim = 5
        bounds = [(0.0, 1.0)] * 5
        f = Ackley(dim=dim, bounds=bounds)
        # convert the continuous features into categorical features
        feature_to_num_categories = {3: 3, 4: 5}
        for feature_idx, num_categories in feature_to_num_categories.items():
            f.bounds[1, feature_idx] = num_categories - 1

        categorical_features = get_categorical_features_dict(feature_to_num_categories)
        one_hot_bounds = torch.zeros(
            2, 3 + sum(categorical_features.values()), **self.tkwargs
        )
        one_hot_bounds[1, :] = 1.0
        init_exact_rounding_func = get_rounding_input_transform(
            one_hot_bounds=one_hot_bounds,
            categorical_features=categorical_features,
            initialization=True,
        )
        one_hot_to_numeric = OneHotToNumeric(
            dim=one_hot_bounds.shape[1], categorical_features=categorical_features
        ).to(**self.tkwargs)

        raw_X = (
            draw_sobol_samples(one_hot_bounds, n=10, q=1).squeeze(-2).to(**self.tkwargs)
        )
        train_X = init_exact_rounding_func(raw_X)
        train_Y = f(one_hot_to_numeric(train_X)).unsqueeze(-1)
        model = get_model(train_X, train_Y)
        base_acq_func = base_acq_func_cls(model, best_f=train_Y.max())

        pr_acq_func = pr_acq_func_cls(
            acq_function=base_acq_func,
            one_hot_bounds=one_hot_bounds,
            categorical_features=categorical_features,
            integer_indices=None,
            batch_limit=32,
        )

        raw_candidate, _ = optimize_acqf(
            acq_function=pr_acq_func,
            bounds=one_hot_bounds,
            q=1,
            num_restarts=10,
            raw_samples=20,
            options={"maxiter": 5},
            # gen_candidates=gen_candidates_scipy,
        )
        # candidates are generated in the one-hot space
        candidate = one_hot_to_numeric(raw_candidate)
        self.assertTrue(candidate.shape == (1, f.dim))

    def test_probabilistic_reparameterization_categorical_analytic_qLogEI(self):
        self.test_probabilistic_reparameterization_categorical(
            pr_acq_func_cls=AnalyticProbabilisticReparameterization,
            base_acq_func_cls=qLogExpectedImprovement,
        )

    def test_probabilistic_reparameterization_categorical_MC_LogEI(self):
        self.test_probabilistic_reparameterization_categorical(
            pr_acq_func_cls=MCProbabilisticReparameterization,
            base_acq_func_cls=LogExpectedImprovement,
        )

    def test_probabilistic_reparameterization_categorical_MC_qLogEI(self):
        self.test_probabilistic_reparameterization_categorical(
            pr_acq_func_cls=MCProbabilisticReparameterization,
            base_acq_func_cls=qLogExpectedImprovement,
        )
