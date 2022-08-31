from botorch.utils.testing import BotorchTestCase
from run_profiling import fn_names, run_and_profile_fn


class TestProfiling(BotorchTestCase):
    def test_run_problem(self):
        additional_args = {
            "run_test_memory_fn": (1,),
            "run_qnei": (1,),
            "run_qnehvi": (1,),
            "run_fit_fully_bayesian_model_nuts": (2, 2),
            "run_large_t_batch_posterior_sampling": (1,),
        }
        self.assertTrue(additional_args.keys() == fn_names.keys())
        for problem_name, args in additional_args.items():
            with self.subTest(problem_name=problem_name, args=args):
                run_and_profile_fn(problem_name, *args)
