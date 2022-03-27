# Changelog

The release log for BoTorch.

## [0.6.3] - Mar 28, 2022

#### New Features
* Implement SAASBO - `SaasFullyBayesianSingleTaskGP` model for sample-efficient high-dimensional Bayesian optimization (#1123).
* Add SAASBO tutorial (#1127).
* Add `LearnedObjective` (#1131), `AnalyticExpectedUtilityOfBestOption` acquisition function (#1135), and a few auxiliary classes to support Bayesian optimization with preference exploration (BOPE).
* Add BOPE tutorial (#1138).

#### Other Changes
* Use `qKG.evaluate` in `optimize_acqf_mixed` (#1133).
* Add `construct_inputs` to SAASBO (#1136).

#### Bug Fixes
* Fix "Constraint Active Search" tutorial (#1124).
* Update "Discrete Multi-Fidelity BO" tutorial (#1134).


## [0.6.2] - Mar 9, 2022

#### New Features
* Use `BOTORCH_MODULAR` in tutorials with Ax (#1105).
* Add `optimize_acqf_discrete_local_search` for discrete search spaces (#1111).

#### Bug Fixes
* Fix missing `posterior_transform` in qNEI and `get_acquisition_function` (#1113).


## [0.6.1] - Feb 28, 2022

#### New Features
* Add `Standardize` input transform (#1053).
* Low-rank Cholesky updates for NEI (#1056).
* Add support for non-linear input constraints (#1067).
* New MOO problems: MW7 (#1077), disc brake (#1078), penicillin (#1079), RobustToy (#1082), GMM (#1083).

#### Other Changes
* Support multi-output models in MES using `PosteriorTransform` (#904).
* Add `Dispatcher` (#1009).
* Modify qNEHVI to support deterministic models (#1026).
* Store tensor attributes of input transforms as buffers (#1035).
* Modify NEHVI to support MTGPs (#1037).
* Make `Normalize` input transform input column-specific (#1047).
* Improve `find_interior_point` (#1049).
* Remove deprecated `botorch.distributions` module (#1061).
* Avoid costly application of posterior transform in Kronecker & HOGP models (#1076).
* Support heteroscedastic perturbations in `InputPerturbations` (#1088).

#### Performance Improvements
* Make risk measures more memory efficient (#1034).

#### Bug Fixes
* Properly handle empty `fixed_features` in optimization (#1029).
* Fix missing weights in `VaR` risk measure (#1038).
* Fix `find_interior_point` for negative variables & allow unbounded problems (#1045).
* Filter out indefinite bounds in constraint utilities (#1048).
* Make non-interleaved base samples use intuitive shape (#1057).
* Pad small diagonalization with zeros for `KroneckerMultitaskGP` (#1071).
* Disable learning of bounds in `preprocess_transform` (#1089).
* Fix `gen_candidates_torch` (4079164489613d436d19c7b2df97677d97dfa8dc).
* Catch runtime errors with ill-conditioned covar (#1095).
* Fix `compare_mc_analytic_acquisition` tutorial (#1099).


## [0.6.0] - Dec 8, 2021

#### Compatibility
* Require PyTorch >=1.9 (#1011).
* Require GPyTorch >=1.6 (#1011).

#### New Features
* New `ApproximateGPyTorchModel` wrapper for various (variational) approximate GP models (#1012).
* New `SingleTaskVariationalGP` stochastic variational Gaussian Process model (#1012).
* Support for Multi-Output Risk Measures (#906, #965).
* Introduce `ModelList` and `PosteriorList` (#829).
* New Constraint Active Search tutorial (#1010).
* Add additional multi-objective optimization test problems (#958).

#### Other Changes
* Add `covar_module` as an optional input of `MultiTaskGP` models (#941).
* Add `min_range` argument to `Normalize` transform to prevent division by zero (#931).
* Add initialization heuristic for acquisition function optimization that samples around best points (#987).
* Update initialization heuristic to perturb a subset of the dimensions of the best points if the dimension is > 20 (#988).
* Modify `apply_constraints` utility to work with multi-output objectives (#994).
* Short-cut `t_batch_mode_transform` decorator on non-tensor inputs (#991).

#### Performance Improvements
* Use lazy covariance matrix in `BatchedMultiOutputGPyTorchModel.posterior` (#976).
* Fast low-rank Cholesky updates for `qNoisyExpectedHypervolumeImprovement` (#747, #995, #996).

#### Bug Fixes
* Update error handling to new PyTorch linear algebra messages (#940).
* Avoid test failures on Ampere devices (#944).
* Fixes to the `Griewank` test function (#972).
* Handle empty base_sample_shape in `Posterior.rsample` (#986).
* Handle `NotPSDError` and hitting `maxiter` in `fit_gpytorch_model` (#1007).
* Use TransformedPosterior for subclasses of GPyTorchPosterior (#983).
* Propagate `best_f` argument to `qProbabilityOfImprovement` in input constructors (f5a5f8b6dc20413e67c6234e31783ac340797a8d).


## [0.5.1] - Sep 2, 2021

#### Compatibility
* Require GPyTorch >=1.5.1 (#928).

#### New Features
* Add `HigherOrderGP` composite Bayesian Optimization tutorial notebook (#864).
* Add Multi-Task Bayesian Optimziation tutorial (#867).
* New multi-objective test problems from (#876).
* Add `PenalizedMCObjective` and `L1PenaltyObjective` (#913).
* Add a `ProximalAcquisitionFunction` for regularizing new candidates towards previously generated ones (#919, #924).
* Add a `Power` outcome transform (#925).

#### Bug Fixes
* Batch mode fix for `HigherOrderGP` initialization (#856).
* Improve `CategoricalKernel` precision (#857).
* Fix an issue with `qMultiFidelityKnowledgeGradient.evaluate` (#858).
* Fix an issue with transforms with `HigherOrderGP`. (#889)
* Fix initial candidate generation when parameter constraints are on different device (#897).
* Fix bad in-place op in `_generate_unfixed_lin_constraints` (#901).
* Fix an input transform bug in `fantasize` call (#902).
* Fix outcome transform bug in `batched_to_model_list` (#917).

#### Other Changes
* Make variance optional for `TransformedPosterior.mean` (#855).
* Support transforms in `DeterministicModel` (#869).
* Support `batch_shape` in `RandomFourierFeatures` (#877).
* Add a `maximize` flag to `PosteriorMean` (#881).
* Ignore categorical dimensions when validating training inputs in `MixedSingleTaskGP` (#882).
* Refactor `HigherOrderGPPosterior` for memory efficiency (#883).
* Support negative weights for minimization objectives in `get_chebyshev_scalarization` (#884).
* Move `train_inputs` transforms to `model.train/eval` calls (#894).


## [0.5.0] - Jun 29, 2021

#### Compatibility
* Require PyTorch >=1.8.1 (#832).
* Require GPyTorch >=1.5 (#848).
* Changes to how input transforms are applied: `transform_inputs` is applied in `model.forward` if the model is in `train` mode, otherwise it is applied in the `posterior` call (#819, #835).

#### New Features
* Improved multi-objective optimization capabilities:
  * `qNoisyExpectedHypervolumeImprovement` acquisition function that improves on `qExpectedHypervolumeImprovement` in terms of tolerating observation noise and speeding up computation for large `q`-batches (#797, #822).
  * `qMultiObjectiveMaxValueEntropy` acqusition function (913aa0e510dde10568c2b4b911124cdd626f6905, #760).
  * Heuristic for reference point selection (#830).
  * `FastNondominatedPartitioning` for Hypervolume computations (#699).
  * `DominatedPartitioning` for partitioning the dominated space (#726).
  * `BoxDecompositionList` for handling box decompositions of varying sizes (#712).
  * Direct, batched dominated partitioning for the two-outcome case (#739).
  * `get_default_partitioning_alpha` utility providing heuristic for selecting approximation level for partitioning algorithms (#793).
  * New method for computing Pareto Frontiers with less memory overhead (#842, #846).
* New `qLowerBoundMaxValueEntropy` acquisition function (a.k.a. GIBBON), a lightweight variant of Multi-fidelity Max-Value Entropy Search using a Determinantal Point Process approximation (#724, #737, #749).
* Support for discrete and mixed input domains:
  * `CategoricalKernel` for categorical inputs (#771).
  * `MixedSingleTaskGP` for mixed search spaces (containing both categorical and ordinal parameters) (#772, #847).
  * `optimize_acqf_discrete` for optimizing acquisition functions over fully discrete domains (#777).
  * Extend `optimize_acqf_mixed` to allow batch optimization (#804).
* Support for robust / risk-aware optimization:
  * Risk measures for robust / risk-averse optimization (#821).
  * `AppendFeatures` transform (#820).
  * `InputPerturbation` input transform for for risk averse BO with implementation errors (#827).
  * Tutorial notebook for Bayesian Optimization of risk measures (#823).
  * Tutorial notebook for risk-averse Bayesian Optimization under input perturbations (#828).
* More scalable multi-task modeling and sampling:
  * `KroneckerMultiTaskGP` model for efficient multi-task modeling for block-design settings (all tasks observed at all inputs) (#637).
  * Support for transforms in Multi-Task GP models (#681).
  * Posterior sampling based on Matheron's rule for Multi-Task GP models (#841).
* Various changes to simplify and streamline integration with Ax:
  * Handle non-block designs in `TrainingData` (#794).
  * Acquisition function input constructor registry (#788, #802, #845).
* Random Fourier Feature (RFF) utilties for fast (approximate) GP function sampling (#750).
* `DelaunayPolytopeSampler` for fast uniform sampling from (simple) polytopes (#741).
* Add `evaluate` method to `ScalarizedObjective` (#795).

#### Bug Fixes
* Handle the case when all features are fixed in `optimize_acqf` (#770).
* Pass `fixed_features` to initial candidate generation functions (#806).
* Handle batch empty pareto frontier in `FastPartitioning` (#740).
* Handle empty pareto set in `is_non_dominated` (#743).
* Handle edge case of no or a single observation in `get_chebyshev_scalarization` (#762).
* Fix an issue in `gen_candidates_torch` that caused problems with acqusition functions using fantasy models (#766).
* Fix `HigherOrderGP` `dtype` bug (#728).
* Normalize before clamping in `Warp` input warping transform (#722).
* Fix bug in GP sampling (#764).

#### Other Changes
* Modify input transforms to support one-to-many transforms (#819, #835).
* Make initial conditions for acquisition function optimization honor parameter constraints (#752).
* Perform optimization only over unfixed features if `fixed_features` is passed (#839).
* Refactor Max Value Entropy Search Methods (#734).
* Use Linear Algebra functions from the `torch.linalg` module (#735).
* Use PyTorch's `Kumaraswamy` distribution (#746).
* Improved capabilities and some bugfixes for batched models (#723, #767).
* Pass `callback` argument to `scipy.optim.minimize` in `gen_candidates_scipy` (#744).
* Modify behavior of `X_pending` in in multi-objective acqusiition functions (#747).
* Allow multi-dimensional batch shapes in test functions (#757).
* Utility for converting batched multi-output models into batched single-output models (#759).
* Explicitly raise `NotPSDError` in `_scipy_objective_and_grad` (#787).
* Make `raw_samples` optional if `batch_initial_conditions` is passed (#801).
* Use powers of 2 in qMC docstrings & examples (#812).


## [0.4.0] - Feb 23, 2021

#### Compatibility
* Require PyTorch >=1.7.1 (#714).
* Require GPyTorch >=1.4 (#714).

#### New Features
* `HigherOrderGP` - High-Order Gaussian Process (HOGP) model for
  high-dimensional output regression (#631, #646, #648, #680).
* `qMultiStepLookahead` acquisition function for general look-ahead
  optimization approaches (#611, #659).
* `ScalarizedPosteriorMean` and `project_to_sample_points` for more
  advanced MFKG functionality (#645).
* Large-scale Thompson sampling tutorial (#654, #713).
* Tutorial for optimizing mixed continuous/discrete domains (application
  to multi-fidelity KG with discrete fidelities) (#716).
* `GPDraw` utility for sampling from (exact) GP priors (#655).
* Add `X` as optional arg to call signature of `MCAcqusitionObjective` (#487).
* `OSY` synthetic test problem (#679).

#### Bug Fixes
* Fix matrix multiplication in `scalarize_posterior` (#638).
* Set `X_pending` in `get_acquisition_function` in `qEHVI` (#662).
* Make contextual kernel device-aware (#666).
* Do not use an `MCSampler` in `MaxPosteriorSampling` (#701).
* Add ability to subset outcome transforms (#711).

#### Performance Improvements
* Batchify box decomposition for 2d case (#642).

#### Other Changes
* Use scipy distribution in MES quantile bisect (#633).
* Use new closure definition for GPyTorch priors (#634).
* Allow enabling of approximate root decomposition in `posterior` calls (#652).
* Support for upcoming 21201-dimensional PyTorch `SobolEngine` (#672, #674).
* Refactored various MOO utilities to allow future additions (#656, #657, #658, #661).
* Support input_transform in PairwiseGP (#632).
* Output shape checks for t_batch_mode_transform (#577).
* Check for NaN in `gen_candidates_scipy` (#688).
* Introduce `base_sample_shape` property to `Posterior` objects (#718).


## [0.3.3] - Dec 8, 2020

Contextual Bayesian Optimization, Input Warping, TuRBO, sampling from polytopes.

#### Compatibility
* Require PyTorch >=1.7 (#614).
* Require GPyTorch >=1.3 (#614).

#### New Features
* Models (LCE-A, LCE-M and SAC ) for Contextual Bayesian Optimziation (#581).
   * Implements core models from:
     [High-Dimensional Contextual Policy Search with Unknown Context Rewards using Bayesian Optimization](https://proceedings.neurips.cc/paper/2020/hash/faff959d885ec0ecf70741a846c34d1d-Abstract.html).
      Q. Feng, B. Letham, H. Mao, E. Bakshy. NeurIPS 2020.
    * See Ax for usage of these models.
* Hit and run sampler for uniform sampling from a polytope (#592).
* Input warping:
  * Core functionality (#607).
  * Kumaraswamy Distribution (#606).
  * Tutorial (8f34871652042219c57b799669a679aab5eed7e3).
* TuRBO-1 tutorial (#598).
  * Implements the method from [Scalable Global Optimization via
Local Bayesian Optimization](https://proceedings.neurips.cc/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf).
    D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, M. Poloczek. NeurIPS 2019.

#### Bug fixes
* Fix bounds of `HolderTable` synthetic function (#596).
* Fix `device` issue in MOO tutorial (#621).

#### Other changes
* Add `train_inputs` option to `qMaxValueEntropy` (#593).
* Enable gpytorch settings to override BoTorch defaults for `fast_pred_var` and `debug` (#595).
* Rename `set_train_data_transform` -> `preprocess_transform` (#575).
* Modify `_expand_bounds()` shape checks to work with >2-dim bounds (#604).
* Add `batch_shape` property to models (#588).
* Modify `qMultiFidelityKnowledgeGradient.evaluate()` to work with `project`, `expand` and `cost_aware_utility` (#594).
* Add list of papers using BoTorch to website docs (#617).


## [0.3.2] - Oct 23, 2020

Maintenance Release

#### New Features
* Add `PenalizedAcquisitionFunction` wrapper (#585)
* Input transforms
  * Reversible input transform (#550)
  * Rounding input transform (#562)
  * Log input transform (#563)
* Differentiable approximate rounding for integers (#561)

#### Bug fixes
* Fix sign error in UCB when `maximize=False` (a4bfacbfb2109d3b89107d171d2101e1995822bb)
* Fix batch_range sample shape logic (#574)

#### Other changes
* Better support for two stage sampling in preference learning
  (0cd13d0cb49b1ac8d0971e42f1f0e9dd6126fd9a)
* Remove noise term in `PairwiseGP` and add `ScaleKernel` by default (#571)
* Rename `prior` to `task_covar_prior` in `MultiTaskGP` and `FixedNoiseMultiTaskGP`
  (8e42ea82856b165a7df9db2a9b6f43ebd7328fc4)
* Support only transforming inputs on training or evaluation (#551)
* Add `equals` method for `InputTransform` (#552)


## [0.3.1] - Sep 15, 2020

Maintenance Release

#### New Features
* Constrained Multi-Objective tutorial (#493)
* Multi-fidelity Knowledge Gradient tutorial (#509)
* Support for batch qMC sampling (#510)
* New `evaluate` method for `qKnowledgeGradient` (#515)

#### Compatibility
* Require PyTorch >=1.6 (#535)
* Require GPyTorch >=1.2 (#535)
* Remove deprecated `botorch.gen module` (#532)

#### Bug fixes
* Fix bad backward-indexing of task_feature in `MultiTaskGP` (#485)
* Fix bounds in constrained Branin-Currin test function (#491)
* Fix max_hv for C2DTLZ2 and make Hypervolume always return a float (#494)
* Fix bug in `draw_sobol_samples` that did not use the proper effective dimension (#505)
* Fix constraints for `q>1` in `qExpectedHypervolumeImprovement` (c80c4fdb0f83f0e4f12e4ec4090d0478b1a8b532)
* Only use feasible observations in partitioning for `qExpectedHypervolumeImprovement`
  in `get_acquisition_function` (#523)
* Improved GPU compatibility for `PairwiseGP` (#537)

#### Performance Improvements
* Reduce memory footprint in `qExpectedHypervolumeImprovement` (#522)
* Add `(q)ExpectedHypervolumeImprovement` to nonnegative functions
  [for better initialization] (#496)

#### Other changes
* Support batched `best_f` in `qExpectedImprovement` (#487)
* Allow to return full tree of solutions in `OneShotAcquisitionFunction` (#488)
* Added `construct_inputs` class method to models to programmatically construct the
  inputs to the constructor from a standardized `TrainingData` representation
  (#477, #482, 3621198d02195b723195b043e86738cd5c3b8e40)
* Acquisition function constructors now accept catch-all `**kwargs` options
  (#478, e5b69352954bb10df19a59efe9221a72932bfe6c)
* Use `psd_safe_cholesky` in `qMaxValueEntropy` for better numerical stabilty (#518)
* Added `WeightedMCMultiOutputObjective` (81d91fd2e115774e561c8282b724457233b6d49f)
* Add ability to specify `outcomes` to all multi-output objectives (#524)
* Return optimization output in `info_dict` for `fit_gpytorch_scipy` (#534)
* Use `setuptools_scm` for versioning (#539)


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
  * Suite of synthetic test functions for multi-objective, constrained optimization
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
