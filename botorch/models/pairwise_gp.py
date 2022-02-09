#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Preference Learning with Gaussian Process

.. [Chu2005preference]
    Wei Chu, and Zoubin Ghahramani. Preference learning with Gaussian processes.
    Proceedings of the 22nd international conference on Machine learning. 2005.

.. [Brochu2010tutorial]
    Eric Brochu, Vlad M. Cora, and Nando De Freitas.
    A tutorial on Bayesian optimization of expensive cost functions,
    with application to active user modeling and hierarchical reinforcement learning.
    arXiv preprint arXiv:1012.2599 (2010).
"""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from gpytorch import settings
from gpytorch.constraints import Positive
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.lazy.lazy_tensor import LazyTensor
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.models.gp import GP
from gpytorch.module import Module
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.utils.cholesky import psd_safe_cholesky
from scipy import optimize
from torch import float32, float64, Tensor


class PairwiseGP(Model, GP):
    r"""Probit GP for preference learning with Laplace approximation

    Implementation is based on [Chu2005preference]_.
    Also see [Brochu2010tutorial]_ for additional reference.

    Note that in [Chu2005preference]_ the likelihood of a pairwise comparison
    is :math:`\left(\frac{f(x_1) - f(x_2)}{\sqrt{2}\sigma}\right)`, i.e. a scale is
    used in the denominator. To maintain consistency with usage of kernels
    elsewhere in botorch, we instead do not include :math:`\sigma` in the code
    (implicitly setting it to 1) and use ScaleKernel to scale the function.
    """

    def __init__(
        self,
        datapoints: Tensor,
        comparisons: Tensor,
        covar_module: Optional[Module] = None,
        input_transform: Optional[InputTransform] = None,
        **kwargs,
    ) -> None:
        r"""A probit-likelihood GP with Laplace approximation model that learns via
            pairwise comparison data. By default it uses a scaled RBF kernel.

        Args:
            datapoints: A `batch_shape x n x d` tensor of training features.
            comparisons: A `batch_shape x m x 2` training comparisons;
                comparisons[i] is a noisy indicator suggesting the utility value
                of comparisons[i, 0]-th is greater than comparisons[i, 1]-th.
            covar_module: Covariance module.
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        super().__init__()

        if input_transform is not None:
            input_transform.to(datapoints)
            # input transformation is applied in set_train_data
            self.input_transform = input_transform

        # Compatibility variables with fit_gpytorch_*: Dummy likelihood
        # Likelihood is tightly tied with this model and
        # it doesn't make much sense to keep it separate
        self.likelihood = None

        # TODO: remove these variables from `state_dict()` so that when calling
        #       `load_state_dict()`, only the hyperparameters are copied over
        self.register_buffer("datapoints", None)
        self.register_buffer("comparisons", None)
        self.register_buffer("D", None)
        self.register_buffer("DT", None)
        self.register_buffer("utility", None)
        self.register_buffer("covar_chol", None)
        self.register_buffer("likelihood_hess", None)
        self.register_buffer("hlcov_eye", None)
        self.register_buffer("covar", None)
        self.register_buffer("covar_inv", None)

        self.train_inputs = []
        self.train_targets = None

        self.pred_cov_fac_need_update = True
        self.dim = None

        # See set_train_data for additional compatibility variables.
        # Not that the datapoints here are not transformed even if input_transform
        # is not None to avoid double transformation during model fitting.
        # self.transform_inputs is called in `forward`
        self.set_train_data(datapoints, comparisons, update_model=False)

        # Set optional parameters
        # jitter to add for numerical stability
        self._jitter = kwargs.get("jitter", 1e-6)
        # Clamping z lim for better numerical stability. See self._calc_z for detail
        # norm_cdf(z=3) ~= 0.999, top 0.1% percent
        self._zlim = kwargs.get("zlim", 3)
        # Stopping creteria in scipy.optimize.fsolve used to find f_map in _update()
        # If None, set to 1e-6 by default in _update
        self._xtol = kwargs.get("xtol")
        # The maximum number of calls to the function in scipy.optimize.fsolve
        # If None, set to 100 by default in _update
        # If zero, then 100*(N+1) is used by default by fsolve;
        self._maxfev = kwargs.get("maxfev")

        # Set hyperparameters
        # Do not set the batch_shape explicitly so mean_module can operate in both mode
        # once fsolve used in _update can run in batch mode, we should explicitly set
        # the bacth shape here
        self.mean_module = ConstantMean()
        # Do not optimize constant mean prior
        for param in self.mean_module.parameters():
            param.requires_grad = False

        # set covariance module
        # the default outputscale here is only a rule of thumb, meant to keep
        # estimates away from scale value that would make Phi(f(x)) saturate
        # at 0 or 1
        if covar_module is None:
            ls_prior = GammaPrior(1.2, 0.5)
            ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate
            covar_module = ScaleKernel(
                RBFKernel(
                    batch_shape=self.batch_shape,
                    ard_num_dims=self.dim,
                    lengthscale_prior=ls_prior,
                    lengthscale_constraint=Positive(
                        transform=None, initial_value=ls_prior_mode
                    ),
                ),
                outputscale_prior=SmoothedBoxPrior(a=1, b=4),
            )

        self.covar_module = covar_module

        self._x0 = None  # will store temporary results for warm-starting
        if self.datapoints is not None and self.comparisons is not None:
            self.to(dtype=self.datapoints.dtype, device=self.datapoints.device)
            # Find f_map for initial parameters with transformed datapoints
            transformed_dp = self.transform_inputs(datapoints)
            self._update(transformed_dp)

        self.to(self.datapoints)

    def __deepcopy__(self, memo) -> PairwiseGP:
        attrs = (
            "datapoints",
            "comparisons",
            "covar",
            "covar_inv",
            "covar_chol",
            "likelihood_hess",
            "utility",
            "hlcov_eye",
        )
        if any(getattr(self, attr) is not None for attr in attrs):
            # Temporarily remove non-leaf tensors so that pytorch allows deepcopy
            old_attr = {}
            for attr in attrs:
                old_attr[attr] = getattr(self, attr)
                setattr(self, attr, None)
            new_model = deepcopy(self, memo)
            # now set things back
            for attr in attrs:
                setattr(self, attr, old_attr[attr])
            return new_model
        else:
            dcp = self.__deepcopy__
            # make sure we don't fall into the infinite recursive loop
            self.__deepcopy__ = None
            new_model = deepcopy(self, memo)
            self.__deepcopy__ = dcp
            return new_model

    def _has_no_data(self):
        r"""Return true if the model does not have both datapoints and comparisons"""
        return (
            self.datapoints is None
            or len(self.datapoints.size()) == 0
            or self.comparisons is None
        )

    def _calc_covar(self, X1: Tensor, X2: Tensor) -> Union[Tensor, LazyTensor]:
        r"""Calculate the covariance matrix given two sets of datapoints"""
        covar = self.covar_module(X1, X2)
        return covar.evaluate()

    def _batch_chol_inv(self, mat_chol: Tensor) -> Tensor:
        r"""Wrapper to perform (batched) cholesky inverse"""
        # TODO: get rid of this once cholesky_inverse supports batch mode
        batch_eye = torch.eye(
            mat_chol.shape[-1],
            dtype=self.datapoints.dtype,
            device=self.datapoints.device,
        )

        if len(mat_chol.shape) == 2:
            mat_inv = torch.cholesky_inverse(mat_chol)
        elif len(mat_chol.shape) > 2 and (mat_chol.shape[-1] == mat_chol.shape[-2]):
            batch_eye = batch_eye.repeat(*(mat_chol.shape[:-2]), 1, 1)
            chol_inv = torch.triangular_solve(batch_eye, mat_chol, upper=False).solution
            mat_inv = chol_inv.transpose(-1, -2) @ chol_inv

        return mat_inv

    def _update_covar(self, datapoints: Tensor) -> None:
        r"""Update values derived from the data and hyperparameters

        covar, covar_chol, and covar_inv will be of shape batch_shape x n x n

        Args:
            datapoints: (Transformed) datapoints for finding f_max
        """
        self.covar = self._calc_covar(datapoints, datapoints)
        self.covar_chol = psd_safe_cholesky(self.covar)
        self.covar_inv = self._batch_chol_inv(self.covar_chol)

    def _prior_mean(self, X: Tensor) -> Union[Tensor, LazyTensor]:
        r"""Return point prediction using prior only

        Args:
            X: A `batch_size x n' x d`-dim Tensor at which to evaluate prior

        Returns:
            Prior mean prediction
        """
        return self.mean_module(X)

    def _prior_predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Predict utility based on prior info only

        Args:
            X: A `batch_size x n' x d`-dim Tensor at which to evaluate prior

        Returns:
            pred_mean: predictive mean
            pred_covar: predictive covariance
        """
        pred_mean = self._prior_mean(X)
        pred_covar = self._calc_covar(X, X)
        return pred_mean, pred_covar

    def _add_jitter(self, X: Tensor) -> Tensor:
        jitter_prev = 0
        Eye = torch.eye(X.size(-1), device=X.device, dtype=X.dtype).expand(X.shape)
        for i in range(3):
            jitter_new = self._jitter * (10 ** i)
            X = X + (jitter_new - jitter_prev) * Eye
            jitter_prev = jitter_new
            # This may be VERY slow given upstream pytorch issue:
            # https://github.com/pytorch/pytorch/issues/34272
            try:
                _ = torch.linalg.cholesky(X)
                warnings.warn(
                    "X is not a p.d. matrix; "
                    f"Added jitter of {jitter_new:.2e} to the diagonal",
                    RuntimeWarning,
                )
                return X
            except RuntimeError:
                continue
        warnings.warn(
            f"Failed to render X p.d. after adding {jitter_new:.2e} jitter",
            RuntimeWarning,
        )
        return X

    def _calc_z(
        self, utility: Tensor, D: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Calculate z score.

        Calculate z score as in [Chu2005preference]_: the standarized difference

        Args:
            utility: A Tensor of shape `batch_size x n`, the utility at MAP point
            D: as in self.D

        Returns:
            z: z score calculated as in [Chu2005preference]_.
            z_logpdf: log PDF of z
            z_logcdf: log CDF of z
            hazard: hazard function defined as pdf(z)/cdf(z)
        """

        scaled_util = (utility / math.sqrt(2)).unsqueeze(-1).to(D)
        z = (D @ scaled_util).squeeze(-1)
        std_norm = torch.distributions.normal.Normal(
            torch.zeros(1, dtype=z.dtype, device=z.device),
            torch.ones(1, dtype=z.dtype, device=z.device),
        )
        # Clamp z for stable log transformation. This also prevent extreme values
        # from appearing in the hess matrix, which should help with numerical
        # stability and avoid extreme fitted hyperparameters
        z = z.clamp(-self._zlim, self._zlim)
        z_logpdf = std_norm.log_prob(z)
        z_cdf = std_norm.cdf(z)
        z_logcdf = torch.log(z_cdf)
        hazard = torch.exp(z_logpdf - z_logcdf)
        return z, z_logpdf, z_logcdf, hazard

    def _grad_likelihood_f_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        r"""Compute the sum over of grad. of negative Log-LH wrt utility f.
        Original grad should be of dimension m x n, as in (6) from [Chu2005preference]_.
        Sum over the first dimension and return a tensor of shape n
        Needed for calculating utility value at f_map to fill in torch gradient

        Args:
            utility: A Tensor of shape `batch_size x n`
            D: A Tensor of shape `batch_size x m x n` as in self.D

        Returns:
            The sum over the first dimension of grad. of negative Log-LH wrt utility f
        """
        _, _, _, h = self._calc_z(utility, D)
        h_factor = (h / math.sqrt(2)).unsqueeze(-2)
        grad = (h_factor @ (-D)).squeeze(-2)

        return grad

    def _hess_likelihood_f_sum(self, utility: Tensor, D: Tensor, DT: Tensor) -> Tensor:
        r"""Compute the sum over of hessian of neg. Log-LH wrt utility f.

        Original hess should be of dimension m x n x n, as in (7) from
        [Chu2005preference]_ Sum over the first dimension and return a tensor of
        shape n x n.

        Args:
            utility: A Tensor of shape `batch_size x n`
            D: A Tensor of shape `batch_size x m x n` as in self.D
            DT: Transpose of D. A Tensor of shape `batch_size x n x m` as in self.DT

        Returns:
            The sum over the first dimension of hess. of negative Log-LH wrt utility f
        """
        z, _, _, h = self._calc_z(utility, D)
        mul_factor = h * (h + z) / 2
        weighted_DT = DT * mul_factor.unsqueeze(-2).expand(*DT.size())
        hess = weighted_DT @ D

        return hess

    def _grad_posterior_f(
        self,
        utility: Union[Tensor, np.ndarray],
        datapoints: Tensor,
        D: Tensor,
        DT: Tensor,
        covar_chol: Tensor,
        covar_inv: Tensor,
        ret_np: bool = False,
    ) -> Union[Tensor, np.ndarray]:
        r"""Compute the gradient of S loss wrt to f/utility in [Chu2005preference]_.

        For finding f_map, which is negative of the log posterior, i.e., -log(p(f|D))
        Derivative of (10) in [Chu2005preference]_.
        Also see [Brochu2010tutorial]_ page 26. This is needed for estimating f_map.

        Args:
            utility: A Tensor of shape `batch_size x n`
            datapoints: A Tensor of shape `batch_size x n x d` as in self.datapoints
            D: A Tensor of shape `batch_size x m x n` as in self.D
            DT: Transpose of D. A Tensor of shape `batch_size x n x m` as in self.DT
            covar_chol: A Tensor of shape `batch_size x n x n`, as in self.covar_chol
            covar_inv: A Tensor of shape `batch_size x n x n`, as in self.covar_inv
            ret_np: return a numpy array if true, otherwise a Tensor
        """
        prior_mean = self._prior_mean(datapoints)

        if ret_np:
            utility = torch.tensor(utility, dtype=self.datapoints.dtype)
            prior_mean = prior_mean.cpu()

        b = self._grad_likelihood_f_sum(utility, D)

        # g_ = covar_inv x (utility - pred_prior)
        p = (utility - prior_mean).unsqueeze(-1).to(covar_chol)
        g_ = torch.cholesky_solve(p, covar_chol).squeeze(-1)
        g = g_ + b

        if ret_np:
            return g.cpu().numpy()
        else:
            return g

    def _hess_posterior_f(
        self,
        utility: Union[Tensor, np.ndarray],
        datapoints: Tensor,
        D: Tensor,
        DT: Tensor,
        covar_chol: Tensor,
        covar_inv: Tensor,
        ret_np: bool = False,
    ) -> Union[Tensor, np.ndarray]:
        r"""Compute the hessian of S loss wrt utility for finding f_map.

        which is negative of the log posterior, i.e., -log(p(f|D))
        Following [Chu2005preference]_ section 2.2.1.
        This is needed for estimating f_map

        Args:
            utility: A Tensor of shape `batch_size x n`
            datapoints: A Tensor of shape `batch_size x n x d` as in self.datapoints
            D: A Tensor of shape `batch_size x m x n` as in self.D
            DT: Transpose of D. A Tensor of shape `batch_size x n x m` as in self.DT
            covar_chol: A Tensor of shape `batch_size x n x n`, as in self.covar_chol
            covar_inv: A Tensor of shape `batch_size x n x n`, as in self.covar_inv
            ret_np: return a numpy array if true, otherwise a Tensor
        """
        if ret_np:
            utility = torch.tensor(utility, dtype=self.datapoints.dtype)

        hl = self._hess_likelihood_f_sum(utility, D, DT)
        hess = hl + covar_inv
        return hess.numpy() if ret_np else hess

    def _posterior_f(self, utility: Union[Tensor, np.ndarray]) -> Tensor:
        r"""Calculate the negative of the log posterior, i.e., -log(P(f|D)).

        This is the S loss function as in equation (10) of [Chu2005preference]_.

        Args:
            utility: A Tensor of shape `batch_size x n`
        """
        _, _, z_logcdf, _ = self._calc_z(utility, self.D)
        loss1 = -(torch.sum(z_logcdf, dim=-1))
        inv_prod = torch.cholesky_solve(utility.unsqueeze(-1), self.covar_chol)
        loss2 = 0.5 * (utility.unsqueeze(-2) @ inv_prod).squeeze(-1).squeeze(-1)
        loss = loss1 + loss2
        loss = loss.clamp(min=0)

        return loss

    def _update_utility_derived_values(self) -> None:
        r"""Calculate utility-derived values not needed during optimization

        Using subsitution method for better numerical stability
        Let `pred_cov_fac = (covar + hl^-1)`, which is needed for calculate
        predictive covariance = `K - k.T @ pred_cov_fac^-1 @ k`
        (Also see posterior mode in `forward`)
        Instead of inverting `pred_cov_fac`, let `hlcov_eye = (hl @ covar + I)`
        Then we can obtain `pred_cov_fac^-1 @ k` by solving for p in
        `(hl @ k) p = hlcov_eye`
        `hlcov_eye p = hl @ k`
        """
        hl = self.likelihood_hess  # "C" from page 27, [Brochu2010tutorial]_
        hlcov = hl @ self.covar
        eye = torch.eye(
            hlcov.size(-1), dtype=self.datapoints.dtype, device=self.datapoints.device
        ).expand(hlcov.shape)
        self.hlcov_eye = hlcov + eye

        self.pred_cov_fac_need_update = False

    def _update(self, datapoints: Tensor, **kwargs) -> None:
        r"""Update the model by updating the covar matrix and MAP utility values

        Update the model by
        1. Re-evaluating the covar matrix as the data or hyperparams may have changed
        2. Approximating maximum a posteriori of the utility function f using fsolve

        Should be called after data or hyperparameters are changed to update
        f_map and related values

        self._xtol and self._maxfev are passed to fsolve as xtol and maxfev
        to control stopping criteria

        Args:
            datapoints: (transformed) datapoints for finding f_max
        """

        xtol = 1e-6 if self._xtol is None else self._xtol
        maxfev = 100 if self._maxfev is None else self._maxfev

        # Using the latest param for covariance before calculating f_map
        self._update_covar(datapoints)

        # scipy newton raphson
        with torch.no_grad():
            # warm start
            init_x0_size = self.batch_shape + torch.Size([self.n])
            if self._x0 is None or torch.Size(self._x0.shape) != init_x0_size:
                x0 = np.random.rand(*init_x0_size)
            else:
                x0 = self._x0

            if len(self.batch_shape) > 0:
                # batch mode, do optimize.fsolve sequentially on CPU
                # TODO: enable vectorization/parallelization here
                x0 = x0.reshape(-1, self.n)
                dp_v = datapoints.view(-1, self.n, self.dim).cpu()
                D_v = self.D.view(-1, self.m, self.n).cpu()
                DT_v = self.DT.view(-1, self.n, self.m).cpu()
                ch_v = self.covar_chol.view(-1, self.n, self.n).cpu()
                ci_v = self.covar_inv.view(-1, self.n, self.n).cpu()
                x = np.empty(x0.shape)
                for i in range(x0.shape[0]):
                    fsolve_args = (dp_v[i], D_v[i], DT_v[i], ch_v[i], ci_v[i], True)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        x[i] = optimize.fsolve(
                            x0=x0[i],
                            func=self._grad_posterior_f,
                            fprime=self._hess_posterior_f,
                            xtol=xtol,
                            maxfev=maxfev,
                            args=fsolve_args,
                            **kwargs,
                        )
                x = x.reshape(*init_x0_size)
            else:
                # fsolve only works on CPU
                fsolve_args = (
                    datapoints.cpu(),
                    self.D.cpu(),
                    self.DT.cpu(),
                    self.covar_chol.cpu(),
                    self.covar_inv.cpu(),
                    True,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    x = optimize.fsolve(
                        x0=x0,
                        func=self._grad_posterior_f,
                        fprime=self._hess_posterior_f,
                        xtol=xtol,
                        maxfev=maxfev,
                        args=fsolve_args,
                        **kwargs,
                    )

            self._x0 = x.copy()  # save for warm-starting
            f = torch.tensor(x, dtype=datapoints.dtype, device=datapoints.device)

        # To perform hyperparameter optimization, this need to be recalculated
        # when calling forward() in order to obtain correct gradients
        # self.likelihood_hess is updated here is for the rare case where we
        # do not want to call forward()
        self.likelihood_hess = self._hess_likelihood_f_sum(f, self.D, self.DT)

        # Lazy update hlcov_eye, which is used in calculating posterior during training
        self.pred_cov_fac_need_update = True
        # fill in dummy values for hlcov_eye so that load_state_dict can function
        hlcov_eye_size = torch.Size((*self.likelihood_hess.shape[:-2], self.n, self.n))
        self.hlcov_eye = torch.empty(hlcov_eye_size)

        self.utility = f.clone().requires_grad_(True)

    def _transform_batch_shape(self, X: Tensor, X_new: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Transform X and X_new into the same shape

        Transform the batch shape of X to be compatible
        with `X_new` to calculate the posterior.
        If X has the same batch size as `X_new`, return it as is.
        If one is in batch mode and the other one is not, convert both
        into batch mode.
        If both are in batch mode, this will only work if X_batch_shape
        can propagate to X_new_batch_shape

        Args:
            X: A `batch_shape x q x d`-dim or `(1 x) q x d`-dim Tensor
            X_new: A `batch_shape x q x d`-dim Tensor

        Returns:
            Transformed X and X_new pair
        """
        X_bs = X.shape[:-2]  # X batch shape
        X_new_bs = X_new.shape[:-2]  # X_new batch shape
        if X_new_bs == X_bs:
            # if batch shapes match, there's no need to transform
            # X_new may or may not have batch_shape dimensions
            return X, X_new
        elif len(X_new_bs) < len(X_bs):
            # if X_new has fewer dimension, try to expand it to X's shape
            return X, X_new.expand(X_bs + X_new.shape[-2:])
        else:
            # if X has fewer dimension, try to expand it to X_new's shape
            return X.expand(X_new_bs + X.shape[-2:]), X_new

    def _util_newton_updates(self, dp, x0, max_iter=1, xtol=None) -> Tensor:
        r"""Make `max_iter` newton updates on utility.

        This is used in `forward` to calculate and fill in gradient into tensors.
        Instead of doing utility -= H^-1 @ g, use substition method.
        See more explanation in _update_utility_derived_values.dd
        By default only need to run one iteration just to fill the the gradients.

        Args:
            dp: (Transformed) datapoints.
            x0: A `batch_size x n` dimension tensor, initial values.
            max_iter: Max number of iterations.
            xtol: Stop creteria. If `None`, do not stop until
                finishing `max_iter` updates.
        """
        xtol = float("-Inf") if xtol is None else xtol
        D, DT, ch, ci = (
            self.D,
            self.DT,
            self.covar_chol,
            self.covar_inv,
        )
        covar = self.covar
        diff = float("Inf")
        i = 0
        x = x0
        eye = None
        while i < max_iter and diff > xtol:
            hl = self._hess_likelihood_f_sum(x, D, DT)
            cov_hl = covar @ hl
            if eye is None:
                eye = torch.eye(
                    cov_hl.size(-1),
                    dtype=dp.dtype,
                    device=dp.device,
                ).expand(cov_hl.shape)
            cov_hl = cov_hl + eye  # add 1 to cov_hl
            g = self._grad_posterior_f(x, dp, D, DT, ch, ci)
            cov_g = covar @ g.unsqueeze(-1)
            x_update = torch.linalg.solve(cov_hl, cov_g).squeeze(-1)
            x_next = x - x_update
            diff = torch.norm(x - x_next)
            x = x_next
            i += 1
        return x

    # ============== public APIs ==============

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return self._num_outputs

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        if self.datapoints is None:
            # this could happen in prior mode
            return torch.Size()
        else:
            return self.datapoints.shape[:-2]

    def set_train_data(
        self,
        datapoints: Tensor = None,
        comparisons: Tensor = None,
        strict: bool = False,
        update_model: bool = True,
    ) -> None:
        r"""Set datapoints and comparisons and update model properties if needed

        Args:
            datapoints: A `batch_shape x n x d` dimension tensor X. If there are input
                transformations, assume the datapoints are not transformed
            comparisons: A tensor of size `batch_shape x m x 2`. (i, j) means
                f_i is preferred over f_j.
            strict: `strict` argument as in gpytorch.models.exact_gp for compatibility
                when using fit_gpytorch_model with input_transform.
            update_model: True if we want to refit the model (see _update) after
                re-setting the data.
        """
        # When datapoints and/or comparisons are None, we are constructing
        # a prior-only model
        if datapoints is None or comparisons is None:
            return

        # following gpytorch.models.exact_gp.set_train_data
        if datapoints is not None:
            if torch.is_tensor(datapoints):
                inputs = (datapoints,)

            inputs = tuple(
                input_.unsqueeze(-1) if input_.ndimension() == 1 else input_
                for input_ in inputs
            )
            if strict:
                for input_, t_input in zip(inputs, self.train_inputs or (None,)):
                    for attr in {"shape", "dtype", "device"}:
                        expected_attr = getattr(t_input, attr, None)
                        found_attr = getattr(input_, attr, None)
                        if expected_attr != found_attr:
                            msg = (
                                "Cannot modify {attr} of inputs "
                                "(expected {e_attr}, found {f_attr})."
                            )
                            msg = msg.format(
                                attr=attr, e_attr=expected_attr, f_attr=found_attr
                            )
                            raise RuntimeError(msg)
            self.datapoints = inputs[0]
            # Compatibility variables with fit_gpytorch_*
            # alias for datapoints ("train_inputs")
            self.train_inputs = inputs

        if comparisons is not None:
            if strict:
                for attr in {"shape", "dtype", "device"}:
                    expected_attr = getattr(self.train_targets, attr, None)
                    found_attr = getattr(comparisons, attr, None)
                    if expected_attr != found_attr:
                        msg = (
                            "Cannot modify {attr} of targets "
                            "(expected {e_attr}, found {f_attr})."
                        )
                        msg = msg.format(
                            attr=attr, e_attr=expected_attr, f_attr=found_attr
                        )
                        raise RuntimeError(msg)
            # convert to long so that it can be used as index and
            # compatible with Tensor.scatter_
            self.comparisons = comparisons.long()
            # Compatibility variables with fit_gpytorch_*
            # alias for comparisons ("train_targets" here)
            self.train_targets = self.comparisons

        # Compatibility variables with optimize_acqf
        self._dtype = self.datapoints.dtype
        self._num_outputs = 1  # 1 latent value output per observation

        self.dim = self.datapoints.shape[-1]  # feature dimensions
        self.n = self.datapoints.shape[-2]  # num datapoints
        self.m = self.comparisons.shape[-2]  # num pairwise comparisons
        self.utility = None
        # D is batch_sizem x n or num_comparison x num_datapoints.
        # D_k_i is the s_k(x_i) value as in equation (6) in [Chu2005preference]_
        # D will usually be very sparse as well
        # TODO swap out scatter_ so that comparisons could be int instead of long
        # TODO: make D a sparse matrix once pytorch has better support for
        #       sparse tensors
        D_size = torch.Size((*(self.batch_shape), self.m, self.n))
        self.D = torch.zeros(
            D_size, dtype=self.datapoints.dtype, device=self.datapoints.device
        )
        comp_view = self.comparisons.view(-1, self.m, 2).long()
        for i, sub_D in enumerate(self.D.view(-1, self.m, self.n)):
            sub_D.scatter_(1, comp_view[i, :, [0]], 1)
            sub_D.scatter_(1, comp_view[i, :, [1]], -1)
        self.DT = self.D.transpose(-1, -2)

        if update_model:
            transformed_dp = self.transform_inputs(datapoints)
            self._update(transformed_dp)

        self.to(self.datapoints)

    def forward(self, datapoints: Tensor) -> MultivariateNormal:
        r"""Calculate a posterior or prior prediction.

        During training mode, forward implemented solely for gradient-based
        hyperparam opt. Essentially what it does is to re-calculate the utility
        f using its analytical form at f_map so that we are able to obtain
        gradients of the hyperparameters.

        Args:
            datapoints: A `batch_shape x n x d` Tensor,
                should be the same as self.datapoints during training

        Returns:
            A MultivariateNormal object, being one of the followings:
                1. Posterior centered at MAP points for training data (training mode)
                2. Prior predictions (prior mode)
                3. Predictive posterior (eval mode)
        """

        # Training mode: optimizing
        if self.training:
            if self._has_no_data():
                raise RuntimeError(
                    "datapoints and comparisons cannot be None in training mode. "
                    "Call .eval() for prior predictions, "
                    "or call .set_train_data() to add training data."
                )

            if datapoints is not self.datapoints:
                raise RuntimeError("Must train on training data")

            transformed_dp = self.transform_inputs(datapoints)

            # We pass in the untransformed datapoints into set_train_data
            # as we will be setting self.datapoints as the untransformed datapoints
            # self.transform_inputs will be called inside before calling _update()
            self.set_train_data(datapoints, self.comparisons, update_model=True)

            # Take a newton step on the posterior MAP point to fill
            # in gradients for pytorch
            self.utility = self._util_newton_updates(
                transformed_dp, self.utility, max_iter=1
            )

            hl = self.likelihood_hess = self._hess_likelihood_f_sum(
                self.utility, self.D, self.DT
            )
            covar = self.covar
            # Apply matrix inversion lemma on eq. in page 27 of [Brochu2010tutorial]_
            # (A + B)^-1 = A^-1 - A^-1 @ (I + BA^-1)^-1 @ BA^-1
            # where A = covar_inv, B = hl
            hl_cov = hl @ covar
            eye = torch.eye(
                hl_cov.size(-1),
                dtype=self.datapoints.dtype,
                device=self.datapoints.device,
            ).expand(hl_cov.shape)
            hl_cov_I = hl_cov + eye  # add I to hl_cov
            train_covar_map = covar - covar @ torch.linalg.solve(hl_cov_I, hl_cov)
            output_mean, output_covar = self.utility, train_covar_map

        # Prior mode
        elif settings.prior_mode.on() or self._has_no_data():
            transformed_new_dp = self.transform_inputs(datapoints)
            # if we don't have any data yet, use prior GP to make predictions
            output_mean, output_covar = self._prior_predict(transformed_new_dp)

        # Posterior mode
        else:
            transformed_dp = self.transform_inputs(self.datapoints)
            transformed_new_dp = self.transform_inputs(datapoints).to(transformed_dp)

            # self.utility might be None if exception was raised and _update
            # was failed to be called during hyperparameter optimization
            # procedures (e.g., fit_gpytorch_scipy)
            if self.utility is None:
                self._update(transformed_dp)

            if self.pred_cov_fac_need_update:
                self._update_utility_derived_values()

            X, X_new = self._transform_batch_shape(transformed_dp, transformed_new_dp)
            covar_chol, _ = self._transform_batch_shape(self.covar_chol, X_new)
            hl, _ = self._transform_batch_shape(self.likelihood_hess, X_new)
            hlcov_eye, _ = self._transform_batch_shape(self.hlcov_eye, X_new)

            # otherwise compute predictive mean and covariance
            covar_xnew_x = self._calc_covar(X_new, X)
            covar_x_xnew = covar_xnew_x.transpose(-1, -2)
            covar_xnew = self._calc_covar(X_new, X_new)
            p = self.utility - self._prior_mean(X)

            covar_inv_p = torch.cholesky_solve(p.unsqueeze(-1), covar_chol)
            pred_mean = (covar_xnew_x @ covar_inv_p).squeeze(-1)
            pred_mean = pred_mean + self._prior_mean(X_new)

            # [Brochu2010tutorial]_ page 27
            # Preictive covariance fatcor: hlcov_eye = (K + C^-1)
            # fac = (K + C^-1)^-1 @ k = pred_cov_fac_inv @ covar_x_xnew
            # used substitution method here to calculate fac
            fac = torch.linalg.solve(hlcov_eye, hl @ covar_x_xnew)
            pred_covar = covar_xnew - (covar_xnew_x @ fac)

            output_mean, output_covar = pred_mean, pred_covar

        try:
            if self.datapoints is None:
                diag_jitter = torch.eye(output_covar.size(-1))
            else:
                diag_jitter = torch.eye(
                    output_covar.size(-1),
                    dtype=self.datapoints.dtype,
                    device=self.datapoints.device,
                )
            diag_jitter = diag_jitter.expand(output_covar.shape)
            diag_jitter = diag_jitter * self._jitter
            # Preemptively adding jitter to diagonal to prevent the use of _add_jitter
            # given that torch.cholesky may be very slow on non-pd matrix input
            # See https://github.com/pytorch/pytorch/issues/34272
            # TODO: remove this once torch.cholesky issue is resolved
            output_covar = output_covar + diag_jitter
            post = MultivariateNormal(output_mean, output_covar)
        except RuntimeError:
            output_covar = self._add_jitter(output_covar)
            post = MultivariateNormal(output_mean, output_covar)

        return post

    # ============== botorch.models.model.Model interfaces ==============
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> Posterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `batch_shape x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered jointly.
            output_indices: As defined in parent Model class, not used for this model.
            observation_noise: Ignored (since noise is not identifiable from scale
                in probit models).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `Posterior` object, representing joint
                distributions over `q` points.
        """
        self.eval()  # make sure model is in eval mode

        if output_indices is not None:
            raise RuntimeError(
                "output_indices is not None. PairwiseGP should not be a"
                "multi-output model."
            )

        post = self(X)
        posterior = GPyTorchPosterior(post)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        else:
            return posterior

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        r"""Condition the model on new observations.

        Note that unlike other BoTorch models, PairwiseGP requires Y to be
        pairwise comparisons

        Args:
            X: A `batch_shape x n x d` dimension tensor X
            Y: A tensor of size `batch_shape x m x 2`. (i, j) means
                f_i is preferred over f_j

        Returns:
            A (deepcopied) `Model` object of the same type, representing the
            original model conditioned on the new observations `(X, Y)`.
        """
        new_model = deepcopy(self)

        if self._has_no_data():
            # If the model previously has no data, set X and Y as the data directly
            new_model.set_train_data(X, Y, update_model=True)
        else:
            # Can only condition on pairwise comparisons instead of the directly
            # observed values. Raise a RuntimeError if Y is not a tensor presenting
            # pairwise comparisons
            if Y.dtype in (float32, float64) or Y.shape[-1] != 2:
                raise RuntimeError(
                    "Conditioning on non-pairwise comparison observations."
                )

            # Reshaping datapoints and comparisons by batches
            Y_new_batch_shape = Y.shape[:-2]
            new_datapoints = self.datapoints.expand(
                Y_new_batch_shape + self.datapoints.shape[-2:]
            )
            new_comparisons = self.comparisons.expand(
                Y_new_batch_shape + self.comparisons.shape[-2:]
            )
            # Reshape X since Y may have additional batch dim. from fantasy models
            X = X.expand(Y_new_batch_shape + X.shape[-2:])

            new_datapoints = torch.cat((new_datapoints, X.to(new_datapoints)), dim=-2)

            shifted_comp = Y.to(new_comparisons) + self.n
            new_comparisons = torch.cat((new_comparisons, shifted_comp), dim=-2)

            # TODO: be smart about how we can update covar matrix here
            new_model.set_train_data(new_datapoints, new_comparisons, update_model=True)
        return new_model


class PairwiseLaplaceMarginalLogLikelihood(MarginalLogLikelihood):
    def __init__(self, model: PairwiseGP) -> None:
        r"""Laplace-approximated marginal log likelihood/evidence for PairwiseGP

        See (12) from [Chu2005preference]_.

        Args:
            model: A model using laplace approximation (currently only supports
                `PairwiseGP`)
        """
        # Do not use likelihood module here as it's implicitly included in the model
        super().__init__(None, model)

    def forward(self, post: Posterior, comp: Tensor) -> Tensor:
        r"""Calculate approximated log evidence, i.e., log(P(D|theta))

        Args:
            post: training posterior distribution from self.model
            comp: Comparisons pairs, see PairwiseGP.__init__ for more details

        Returns:
            The approximated evidence, i.e., the marginal log likelihood
        """

        model = self.model
        if comp is not model.comparisons:
            raise RuntimeError("Must train on training data")

        f_max = post.mean
        log_posterior = model._posterior_f(f_max)
        part1 = -log_posterior

        part2 = model.covar @ model.likelihood_hess
        eye = torch.eye(
            part2.size(-1), dtype=model.datapoints.dtype, device=model.datapoints.device
        ).expand(part2.shape)
        part2 = part2 + eye
        part2 = -0.5 * torch.logdet(part2)

        evidence = part1 + part2

        # Sum up mll first so that when adding prior probs it won't
        # propagate and double count
        evidence = evidence.sum()

        # Add log probs of priors on the (functions of) parameters
        for _, module, prior, closure, _ in self.named_priors():
            evidence = evidence.add(prior.log_prob(closure(module)).sum())

        return evidence
