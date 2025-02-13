from typing import Optional

import torch
from torch import Tensor

from botorch.posteriors import Posterior


class BLLPosterior(Posterior):
    def __init__(self, posterior, model, X, output_dim):
        super().__init__()
        self.posterior = posterior
        self.model = model
        self.old_model = model
        self.output_dim = output_dim
        self.X = X

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
    ) -> Tensor:
        """
        For VBLLs, we need to sample from W and then create the generalized linear model to get posterior samples.
        """
        n_samples = (
            1 if sample_shape is None else torch.tensor(sample_shape).prod().item()
        )
        samples_list = [self.model.sample()(self.X) for _ in range(n_samples)]
        samples = torch.stack(samples_list, dim=0)
        new_shape = samples.shape[:-1]
        return samples.reshape(*new_shape, -1, self.output_dim)

    @property
    def mean(self) -> Tensor:
        r"""The posterior mean."""
        post_mean = self.posterior.mean.squeeze(-1)
        shape = post_mean.shape
        return post_mean.reshape(*shape[:-1], -1, self.output_dim)

    @property
    def variance(self) -> Tensor:
        r"""The posterior variance."""
        post_var = self.posterior.variance.squeeze(-1)
        shape = post_var.shape
        return post_var.reshape(*shape[:-1], -1, self.output_dim)

    @property
    def device(self) -> torch.device:
        return self.posterior.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the distribution."""
        return self.posterior.dtype
