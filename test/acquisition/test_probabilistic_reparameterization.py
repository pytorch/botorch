import torch
from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
from botorch.acquisition.probabilistic_reparameterization import (
    AnalyticProbabilisticReparameterization,
    MCProbabilisticReparameterization,
)
from botorch.optim import optimize_acqf
from botorch.test_functions.synthetic import AckleyMixed
from botorch.utils.test_helpers import get_model
from botorch.utils.testing import BotorchTestCase


class TestProbabilisticReparameterizationInputTransform(BotorchTestCase):
    def test_probabilistic_reparameterization_input_transform(self):
        pass


class TestProbabilisticReparameterization(BotorchTestCase):
    def test_probabilistic_reparameterization_binary(
        self,
        pr_acq_func_cls=AnalyticProbabilisticReparameterization,
        base_acq_func_cls=LogExpectedImprovement,
    ):
        torch.manual_seed(0)
        f = AckleyMixed(dim=13, randomize_optimum=True)
        train_X = torch.rand((10, f.dim), dtype=torch.float64)
        train_X[:, f.discrete_inds] = train_X[:, f.discrete_inds].round()
        train_Y = f(train_X).unsqueeze(-1)
        model = get_model(train_X, train_Y)
        base_acq_func = base_acq_func_cls(model, best_f=train_Y.max())

        pr_acq_func = pr_acq_func_cls(
            acq_function=base_acq_func,
            one_hot_bounds=f.bounds,
            integer_indices=f.discrete_inds,
            batch_limit=32,
        )

        candidate, _ = optimize_acqf(
            acq_function=pr_acq_func,
            bounds=f.bounds,
            q=1,
            num_restarts=10,
            raw_samples=20,
            options={"maxiter": 5},
        )

        self.assertTrue(candidate.shape == (1, f.dim))

    def test_probabilistic_reparameterization_binary_analytic_qLogEI(self):
        self.test_probabilistic_reparameterization_binary(
            pr_acq_func_cls=AnalyticProbabilisticReparameterization,
            base_acq_func_cls=qLogExpectedImprovement,
        )

    def test_probabilistic_reparameterization_binary_MC_LogEI(self):
        self.test_probabilistic_reparameterization_binary(
            pr_acq_func_cls=MCProbabilisticReparameterization,
            base_acq_func_cls=LogExpectedImprovement,
        )

    def test_probabilistic_reparameterization_binary_MC_qLogEI(self):
        self.test_probabilistic_reparameterization_binary(
            pr_acq_func_cls=MCProbabilisticReparameterization,
            base_acq_func_cls=qLogExpectedImprovement,
        )

    def test_probabilistic_reparameterization_categorical(self):
        pass
