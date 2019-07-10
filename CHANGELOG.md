# Changelog

The release log for BoTorch.


## [0.1.2] - July 9, 2019

Maintenance release

#### Bug fixes
* Avoid [PyTorch bug]((https://github.com/pytorch/pytorch/issues/22353)
  resulting in bad gradients on GPU by requiring GPyTorch >= 0.3.4
* Fixes to resampling behavior in MCSamplers (#204)

#### Experimental Features
* Linear truncated kernel for multi-fidelity bayesian optimization (#192)
* SingleTaskMultiFidelityGP for GP models that have fidelity parameters (#181)


## [0.1.1] - June 27, 2019

API updates, more robust model fitting

#### Breaking changes
* rename `botorch.qmc` to `botorch.sampling`, move MC samplers from
  `acquisition.sampler` to `botorch.sampling.samplers` (#172)

#### New Features
* Add `condition_on_observations` and `fantasize` to the Model level API (#173)
* Support pending observations generically for all `MCAcqusitionFunctions` (#176)
* Add fidelity kernel for training iterations/training data points (#178)
* Support for optimization constraints across `q`-batches (to support things like
  sample budget constraints) (2a95a6c3f80e751d5cf8bc7240ca9f5b1529ec5b)
* Add ModelList <-> Batched Model converter (#187)
* New test functions
    * basic: `neg_ackley`, `cosine8`, `neg_levy`, `neg_rosenbrock`, `neg_shekel`
      (e26dc7576c7bf5fa2ba4cb8fbcf45849b95d324b)
    * for multi-fidelity BO: `neg_aug_branin`, `neg_aug_hartmann6`,
      `neg_aug_rosenbrock` (ec4aca744f65ca19847dc368f9fee4cc297533da)

#### Improved functionality:
* More robust model fitting
    * Catch gpytorch numerical issues and return `NaN` to the optimizer (#184)
    * Restart optimization upon failure by sampling hyperparameters from their prior (#188)
    * Sequentially fit batched and `ModelListGP` models by default (#189)
    * Change minimum inferred noise level (e2c64fef1e76d526a33951c5eb75ac38d5581257)
* Introduce optional batch limit in `joint_optimize` to increases scalability of
  parallel optimization (baab5786e8eaec02d37a511df04442471c632f8a)
* Change constructor of `ModelListGP` to comply with GPyTorch’s `IndependentModelList`
  constructor (a6cf739e769c75319a67c7525a023ece8806b15d)
* Use `torch.random` to set default seed for samplers (rather than `random`) to
  making sampling reproducible when setting `torch.manual_seed`
  (ae507ad97255d35f02c878f50ba68a2e27017815)

####  Performance Improvements
* Use `einsum` in `LinearMCObjective` (22ca29535717cda0fcf7493a43bdf3dda324c22d)
* Change default Sobol sample size for `MCAquisitionFunctions` to be base-2 for
  better MC integration performance (5d8e81866a23d6bfe4158f8c9b30ea14dd82e032)
* Add ability to fit models in `SumMarginalLogLikelihood` sequentially (and make
  that the default setting) (#183)
* Do not construct the full covariance matrix when computing posterior of
  single-output BatchedMultiOutputGPyTorchModel (#185)

#### Bug fixes
* Properly handle observation_noise kwarg for BatchedMultiOutputGPyTorchModels (#182)
* Fix a issue where `f_best` was always max for NoisyExpectedImprovement
  (de8544a75b58873c449b41840a335f6732754c77)
* Fix bug and numerical issues in `initialize_q_batch`
  (844dcd1dc8f418ae42639e211c6bb8e31a75d8bf)
* Fix numerical issues with `inv_transform` for qMC sampling (#162)

#### Other
* Bump GPyTorch minimum requirement to 0.3.3



## [0.1.0] - April 30, 2019

First public beta release.
