# Changelog

The release log for BoTorch.


## [0.3.0] - July 6, 2020

Multi-Objective Bayesian Optimization

#### New Features
* Multi-Objective Acquisition Functions (#466)
  * q-Expected Hypervolume Improvement
  * q-ParEGO
  * Analytic Expected Hypervolume Improvement with auto-differentiation
* Multi-Objective Utilities (#466)
  * Pareto Computation
  * Hypervolume Calculation
  * Box Decomposition algorithm
* Multi-Objective Test Functions (#466)
  * Suite of synthetic test functions for multi-objective, constrained
  optimzation
* Multi-Objective Tutorial (#468)
* Abstract ConstrainedBaseTestProblem (#454)
* Add optimize_acqf_list method for sequentially, greedily optimizing 1 candidate
from each provided acquisition function (d10aec911b241b208c59c192beb9e4d572a092cd)


#### Bug fixes
* Fixed re-arranging mean in MultiTask MO models (#450).

#### Other changes
* Move gpt_posterior_settings into models.utils (#449)
* Allow specifications of batch dims to collapse in samplers (#457)
* Remove outcome transform before model-fitting for sequential model fitting
in MO models (#458)


## [0.2.5] - May 14, 2020

Bugfix Release

#### Bug fixes
* Fixed issue with broken wheel build (#444).

#### Other changes
* Changed code style to use absolute imports throughout (#443).


## [0.2.4] - May 12, 2020

Bugfix Release

#### Bug fixes
* There was a mysterious issue with the 0.2.3 wheel on pypi, where part of the
  `botorch/optim/utils.py` file was not included, which resulted in an `ImportError` for
  many central components of the code. Interestingly, the source dist (built with the
  same command) did not have this issue.
* Preserve order in ChainedOutcomeTransform (#440).

#### New Features
* Utilities for estimating the feasible volume under outcome constraints (#437).


## [0.2.3] - Apr 27, 2020

Pairwise GP for Preference Learning, Sampling Strategies.

#### Compatibility
* Require PyTorch >=1.5 (#423).
* Require GPyTorch >=1.1.1 (#425).

#### New Features
* Add `PairwiseGP` for preference learning with pair-wise comparison data (#388).
* Add `SamplingStrategy` abstraction for sampling-based generation strategies, including
  `MaxPosteriorSampling` (i.e. Thompson Sampling) and `BoltzmannSampling` (#218, #407).

#### Deprecations
* The existing `botorch.gen` module is moved to `botorch.generation.gen` and imports
  from `botorch.gen` will raise a warning (an error in the next release) (#218).

#### Bug fixes
* Fix & update a number of tutorials (#394, #398, #393, #399, #403).
* Fix CUDA tests (#404).
* Fix sobol maxdim limitation in `prune_baseline` (#419).

#### Other changes
* Better stopping criteria for stochastic optimization (#392).
* Improve numerical stability of `LinearTruncatedFidelityKernel` (#409).
* Allow batched `best_f` in `qExpectedImprovement` and `qProbabilityOfImprovement`
  (#411).
* Introduce new logger framework (#412).
* Faster indexing in some situations (#414).
* More generic `BaseTestProblem` (9e604fe2188ac85294c143d249872415c4d95823).


## [0.2.2] - Mar 6, 2020

Require PyTorch 1.4, Python 3.7 and new features for active learning,
multi-fidelity optimization, and a number of bug fixes.

#### Compatibility
* Require PyTorch >=1.4 (#379).
* Require Python >=3.7 (#378).

#### New Features
* Add `qNegIntegratedPosteriorVariance` for Bayesian active learning (#377).
* Add `FixedNoiseMultiFidelityGP`, analogous to `SingleTaskMultiFidelityGP` (#386).
* Support `scalarize_posterior` for m>1 and q>1 posteriors (#374).
* Support `subset_output` method on multi-fidelity models (#372).
* Add utilities for sampling from simplex and hypersphere (#369).

#### Bug fixes
* Fix `TestLoader` local test discovery (#376).
* Fix batch-list conversion of `SingleTaskMultiFidelityGP` (#370).
* Validate tensor args before checking input scaling for more
  informative error messaages (#368).
* Fix flaky `qNoisyExpectedImprovement` test (#362).
* Fix test function in closed-loop tutorial (#360).
* Fix num_output attribute in BoTorch/Ax tutorial (#355).

#### Other changes
* Require output dimension in `MultiTaskGP` (#383).
* Update code of conduct (#380).
* Remove deprecated `joint_optimize` and `sequential_optimize` (#363).


## [0.2.1] - Jan 15, 2020

Minor bug fix release.

#### New Features
* Add a static method for getting batch shapes for batched MO models (#346).

#### Bug fixes
* Revamp qKG constructor to avoid issue with missing objective (#351).
* Make sure MVES can support sampled costs like KG (#352).

#### Other changes
* Allow custom module-to-array handling in fit_gpytorch_scipy (#341).


## [0.2.0] - Dec 20, 2019

Max-value entropy acquisition function, cost-aware / multi-fidelity optimization,
subsetting models, outcome transforms.

#### Compatibility
* Require PyTorch >=1.3.1 (#313).
* Require GPyTorch >=1.0 (#342).

#### New Features
* Add cost-aware KnowledgeGradient (`qMultiFidelityKnowledgeGradient`) for
  multi-fidelity optimization (#292).
* Add `qMaxValueEntropy` and `qMultiFidelityMaxValueEntropy` max-value entropy
  search acquisition functions (#298).
* Add `subset_output` functionality to (most) models (#324).
* Add outcome transforms and input transforms (#321).
* Add `outcome_transform` kwarg to model constructors for automatic outcome
  transformation and un-transformation (#327).
* Add cost-aware utilities for cost-sensitive acquisiiton functions (#289).
* Add `DeterminsticModel` and `DetermisticPosterior` abstractions (#288).
* Add `AffineFidelityCostModel` (f838eacb4258f570c3086d7cbd9aa3cf9ce67904).
* Add `project_to_target_fidelity` and `expand_trace_observations` utilties for
  use in multi-fidelity optimization (1ca12ac0736e39939fff650cae617680c1a16933).

#### Performance Improvements
* New `prune_baseline` option for pruning `X_baseline` in
  `qNoisyExpectedImprovement` (#287).
* Do not use approximate MLL computation for deterministic fitting (#314).
* Avoid re-evaluating the acquisition function in `gen_candidates_torch` (#319).
* Use CPU where possible in `gen_batch_initial_conditions` to avoid memory
  issues on the GPU (#323).

#### Bug fixes
* Properly register `NoiseModelAddedLossTerm` in `HeteroskedasticSingleTaskGP`
  (671c93a203b03ef03592ce322209fc5e71f23a74).
* Fix batch mode for `MultiTaskGPyTorchModel` (#316).
* Honor `propagate_grads` argument in `fantasize` of `FixedNoiseGP` (#303).
* Properly handle `diag` arg in `LinearTruncatedFidelityKernel` (#320).

#### Other changes
* Consolidate and simplify multi-fidelity models (#308).
* New license header style (#309).
* Validate shape of `best_f` in `qExpectedImprovement` (#299).
* Support specifying observation noise explicitly for all models (#256).
* Add `num_outputs` property to the `Model` API (#330).
* Validate output shape of models upon instantiating acquisition functions (#331).

#### Tests
* Silence warnings outside of explicit tests (#290).
* Enforce full sphinx docs coverage in CI (#294).


## [0.1.4] - Oct 1, 2019

Knowledge Gradient acquisition function (one-shot), various maintenance

#### Breaking Changes
* Require explicit output dimensions in BoTorch models (#238)
* Make `joint_optimize` / `sequential_optimize` return acquisition function
  values (#149) [note deprecation notice below]
* `standardize` now works on the second to last dimension (#263)
* Refactor synthetic test functions (#273)

#### New Features
* Add `qKnowledgeGradient` acquisition function (#272, #276)
* Add input scaling check to standard models (#267)
* Add `cyclic_optimize`, convergence criterion class (#269)
* Add `settings.debug` context manager (#242)

#### Deprecations
* Consolidate `sequential_optimize` and `joint_optimize` into `optimize_acqf`
  (#150)

#### Bug fixes
* Properly pass noise levels to GPs using a `FixedNoiseGaussianLikelihood` (#241)
  [requires gpytorch > 0.3.5]
* Fix q-batch dimension issue in `ConstrainedExpectedImprovement`
  (6c067185f56d3a244c4093393b8a97388fb1c0b3)
* Fix parameter constraint issues on GPU (#260)

#### Minor changes
* Add decorator for concatenating pending points (#240)
* Draw independent sample from prior for each hyperparameter (#244)
* Allow `dim > 1111` for `gen_batch_initial_conditions` (#249)
* Allow `optimize_acqf` to use `q>1` for `AnalyticAcquisitionFunction` (#257)
* Allow excluding parameters in fit functions (#259)
* Track the final iteration objective value in `fit_gpytorch_scipy` (#258)
* Error out on unexpected dims in parameter constraint generation (#270)
* Compute acquisition values in gen_ functions w/o grad (#274)

#### Tests
* Introduce BotorchTestCase to simplify test code (#243)
* Refactor tests to have monolithic cuda tests (#261)


## [0.1.3] - Aug 9, 2019

Compatibility & maintenance release

#### Compatibility
* Updates to support breaking changes in PyTorch to boolean masks and tensor
  comparisons (#224).
* Require PyTorch >=1.2 (#225).
* Require GPyTorch >=0.3.5 (itself a compatibility release).

#### New Features
* Add `FixedFeatureAcquisitionFunction` wrapper that simplifies optimizing
  acquisition functions over a subset of input features (#219).
* Add `ScalarizedObjective` for scalarizing posteriors (#210).
* Change default optimization behavior to use L-BFGS-B by for box constraints
  (#207).

#### Bug fixes
* Add validation to candidate generation (#213), making sure constraints are
  strictly satisfied (rater than just up to numerical accuracy of the optimizer).

#### Minor changes
* Introduce `AcquisitionObjective` base class (#220).
* Add propagate_grads context manager, replacing the `propagate_grads` kwarg in
  model `posterior()` calls (#221)
* Add `batch_initial_conditions` argument to `joint_optimize()` for
  warm-starting the optimization (ec3365a37ed02319e0d2bb9bea03aee89b7d9caa).
* Add `return_best_only` argument to `joint_optimize()` (#216). Useful for
  implementing advanced warm-starting procedures.


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
