# Pareto decisions and experimental FPBI: evidence and reproduction

This companion to [the notebook](pareto_decision_making.ipynb) documents the
proposed knee/preference selector and experimental focus point boundary
intersection (FPBI) sampler from [Schmitt et al. (2022)](https://doi.org/10.3390/inventions7030046).
The proposal is [draft PR #3531](https://github.com/meta-pytorch/botorch/pull/3531),
with [scope issue #3530](https://github.com/meta-pytorch/botorch/issues/3530).
It is a community research proposal, not an accepted core API or a claim of
general superiority.

## What worked

Analytical tests reproduce the signed knee-distance equation, preference
plane, asymmetric FPBI grid and unrestricted ray parameter. Functional tests
exercise duplicate/dominated/flat outcomes, batching, dtype/device handling,
original indices, negative ray parameters, constrained continuous solves and
failed-solver rejection. Posterior-mean adapters work with compatible model
interfaces. Neither the selector nor FPBI learns a person's preferences,
supplies a BO exploration incentive, or guarantees MPC stability.

Historical validation includes 35 OpenBO tests passing with one CUDA skip in
its CPU environment, 21 Rust Python decision tests, eight native Rust decision
tests, 18 initial upstream selector CPU/CUDA tests, and 144 exact mask/index
comparisons across both library CUDA facades. The external HITL repository's
complete suite passed 1,250 tests, with 11 runtime skips and one
collection-skipped module; five new replay tests and static paper checks
passed. Full LaTeX and Docusaurus builds were not performed.

The expanded selector/FPBI/harness CPU check passed 36 tests and ten subtests,
with ten CUDA tests skipped in that environment (46 test functions). A
dedicated FPBI CUDA boundary test passed; the existing selector CPU/CUDA suite
passed all 18 tests. Both ported CLIs completed functional smokes: all three
FPBI arms on a convex seed-19 case, and 12 CPU selector configurations with
exact parity. These validate the later port, not a rerun of old performance
measurements on new FPBI module bytes. See the
[FPBI port validation record](evidence/validation/fpbi-port-validation.json)
and [current check summary](evidence/validation/current-upstream-checks.json).

## What did not improve

### Final decision quality under simulated error

The fixed-archive replay uses 100 existing logs from BraninCurrin, ZDT1,
DTLZ2 and VehicleSafety: qLogNEHVI, seeds 12–16, 50 trials each. Twenty logs
are clean optimization runs; 80 contain Gaussian objective errors of one
landscape SD, onset 0/20, and objective-error correlation 0/.85. Selectors and
counterfactuals share the same candidate archive. No new objective evaluations,
GP fits or search runs occur. There are 98,100 scored rows.

For each seed, preferences include 24 Dirichlet vectors, equal weights and
each objective axis. Separately simulated log-weight noise has SD 0/.25/1;
it preserves exact zero weights. Ground-truth normalized costs `Z` use the
true Pareto ranges within each archive and are frozen across decision rules
and error conditions. Linear cost is `sum(p * Z)`; quadratic distance cost
is `sum(p * Z**2)`. Each has its matching ordinary reference. Regret is the
chosen true cost minus the best true cost among the visited designs.

The following pools the noisy archives with exact preference weights:

| Declared utility | Reference | Reference regret | Knee regret | Knee gain | Knee clean cost |
| --- | --- | ---: | ---: | ---: | ---: |
| Linear | Weighted sum | 0.237706 | 0.332108 | -0.094402 | +0.072301 |
| Quadratic distance | Weighted nearest ideal | 0.452719 | 0.547495 | -0.094777 | +0.032571 |

For completeness, the third rule is also retained. Nearest ideal under
linear truth has noisy regret 0.269813, gain -0.032106 and clean cost
+0.033651. Weighted selection under quadratic truth has regret 0.448362,
gain +0.004357 and clean cost +0.031272. Its small descriptive gain does
not establish superiority; it comes with a clean loss.

Gain is reference noisy regret minus treatment noisy regret. Clean cost is
treatment regret minus reference regret with exact outcomes and true weights
on the **same archive**, rather than a different optimization trajectory.
Units are within-archive true-front-normalized final selection regret; they
are not hypervolume loss, global regret or measured human utility.

The linear knee gain is negative on each of the four problems. Among random
Dirichlet preferences alone, ordinary weighted regret is 0.237492 versus
0.324670 for the knee rule, with clean cost 0.066320. Equal supplied weights
give the same weighted and knee-weighted decision, as expected from the
formula. Preference-weight corruption alone did not establish a general
knee robustness benefit either.

See [summary](evidence/selection/comparison_summary.csv),
[every problem](evidence/selection/comparison_per_problem.csv),
[preference groups](evidence/selection/preference_groups.csv), and
[protocol](evidence/selection/protocol.json). Four correlated problem clusters
support descriptive comparisons only. Their bootstrap treats clusters as
independent; no significance stars or p-values are reported. A person who
explicitly wants a compromise restriction has a different decision target.

The replay's `oracle_best_excluded` diagnostic includes removal by perceived
Pareto filtering and by the additional knee threshold; it does not isolate
the latter. `knee_fraction` uses all 50 visited designs as its denominator.

### FPBI front generation

Thirty-six CPU double-precision runs cover convex 2D, concave 2D, constrained
2D and sphere 3D cases, with seeds 0–2. Both methods optimize exact analytic
costs through real deterministic-model interfaces. The comparator uses
BoTorch's NSGA-II wrapper. One arm shares an output cap; another rounds the
measured FPBI objective-point budget to a population-sized generation.

Median HV ratio and IGD favored NSGA-II on all four cases in both arms.
Under approximately matched point budgets:

| Case | FPBI HV ratio | NSGA-II HV ratio | FPBI seconds | NSGA-II seconds |
| --- | ---: | ---: | ---: | ---: |
| Convex 2D | 0.9220 | 0.9519 | 0.3219 | 0.1220 |
| Concave 2D | 0.8502 | 0.8693 | 0.7665 | 0.2282 |
| Constrained 2D | 0.8940 | 0.9228 | 0.4854 | 0.1204 |
| Sphere 3D | 0.5575 | 0.7287 | 0.9405 | 0.2728 |

No complete runs failed. All 72 nominal FPBI rays succeeded, but five of 369
restart attempts were rejected. The maximum returned constraint violation
was approximately 2.8e-16. Output cardinalities were capped, not identical: 2D medians
are six FPBI points versus seven NSGA-II points, and sphere medians are 11
versus 12. Every decision vector counts as an evaluated point; a population
batch counts many points but one posterior call. Finite differences and
diagnostic evaluations are counted. Generation time excludes imports,
reference metrics, common output capping and serialization.

Quality uses known theoretical objective ranges, reference `(1.1,...,1.1)`
and exact continuous-front hypervolume. IGD uses 4,097 fixed reference points,
plus sphere axes; its reference sampling measure is part of the protocol.
See [summary with ranges](evidence/fpbi/summary.csv),
[per-run metrics](evidence/fpbi/runs.csv),
[fronts and solver diagnostics](evidence/fpbi/runs.json), and
[protocol](evidence/fpbi/protocol.json). These cheap, low-dimensional cases
favor vectorized population evaluation. They are not a general optimizer
ranking, noisy BO experiment or global-optimality certificate.

### Selector runtime and memory

All 396 indices and 79,104 knee-mask entries match exactly in 48 paired
function configurations. Full selection improved in 11/12 CPU and 4/12 CUDA
configurations; best observed gains were 4.07x and 26.73x respectively.
The worst full-selection case was CUDA float32 `(1,1024,3)`: 6.862 ms to
165.3 ms, approximately **24.1x slower**. Incremental live CUDA allocation in
that case fell from 6,314,496 to 53,760 bytes as BoTorch's memory-bounded
Pareto dispatch traded memory for latency. Memory did not improve everywhere.

The knee-mask-only maximum regression was about 40.1x slower, distinct from
the full selector above. See [all paired timings and allocator counts](evidence/selector/timings.csv),
[individual repetitions](evidence/selector/timings.json), and
[protocol](evidence/selector/protocol.json). This was one machine, two warmups,
five repetitions and observed independent background workloads. The results
are descriptive ranges, not isolated-machine or cross-hardware speed claims.
No implementation path was retuned after seeing the results.

The baseline is the earlier Torch prototype; the measured candidate is the
vectorized upstream selector. The archived `after.py.txt` hash matches
`botorch_community/utils/multi_objective/decision_making.py` at packaging
time. The later upstream FPBI port is a separate module.

## Applicable uses

Use ordinary weighted selection for a declared linear utility and weighted
nearest ideal for that declared distance objective. Use a knee restriction
only when excluding some extremes is itself wanted; `alpha=0` does not
generally disable the signed filter. Keep FPBI experimental for inexpensive
objectives or surrogate means with few objectives, check solver diagnostics,
and compare generators in the application. Keep acquisition, uncertainty
handling and preference learning separate.

## Reproduce new upstream experiments

From the BoTorch repository root, using its development environment and the
optional `pymoo` comparator dependency:

```bash
python -m notebooks_community.pareto_decision_making.experiments.benchmark_fpbi \
  --output-dir fpbi-results-new --seeds 0 1 2 \
  --cases convex_2d concave_2d constrained_2d sphere_3d \
  --steps-2d 5 --steps-3d 3 --restarts 2 --maxiter 100 \
  --population 32 --output-generations 20 --max-budget-generations 300 --fpbi-api model

python -m notebooks_community.pareto_decision_making.experiments.benchmark_selector \
  --before notebooks_community/pareto_decision_making/evidence/selector/source/before.py.txt \
  --after botorch_community/utils/multi_objective/decision_making.py \
  --output-dir selector-results-new
```

Check each harness's `--help` and current tests before running. Use fresh
output directories. The FPBI command runs the **later upstream port**, not
the historical OpenBO package. The selector command loads the checked-out
candidate, whose hash matched the archived measured candidate at packaging
time. Preserve or check those hashes if reproducing later. Changed software,
hardware or background load can change timings.

The combined current CPU check can be repeated with:

```bash
python -m pytest test_community/utils/multi_objective/test_decision_making.py \
  test_community/utils/multi_objective/test_fpbi.py \
  test_community/utils/multi_objective/test_fpbi_benchmark.py \
  test_community/utils/multi_objective/test_selector_benchmark.py -q
```

## Historical provenance and source snapshots

The [checksum index](evidence/packaged-file-hashes.json) verifies copied source
snapshots against the original provenance hashes. Historical Python sources
use `.py.txt` names to keep formatting, linting and test discovery from changing
their bytes. This is archival source, distinct from runnable, maintained
`experiments/*.py` harnesses. The selector harness can load the historical
baseline snapshot explicitly. To use a historical package elsewhere, restore
the original `.py` filenames in a separate checkout.

The old FPBI and replay measurements used OpenBO's historical implementation,
whose package bytes are in `evidence/fpbi/source/src/openbo/decision_making/`.
The original FPBI benchmark source is in `evidence/fpbi/source/scripts/`.
The before/after selector and its original timing harness are in
`evidence/selector/source/`. Their original hashes are retained even when
later upstream code evolves. The replay harness is under
`evidence/selection/source/`.

Published metadata JSONs are explicitly wrapped sanitized copies: machine
paths were replaced with role labels. Each wrapper preserves the original
metadata file's SHA256; the checksum index separately hashes the sanitized
published file. Original source snapshots retain exact bytes and their
original source hashes. Reproduction helper hashes captured during packaging
are labeled separately from the measurement-time provenance.

## External replay data and dependencies

The selection replay is **not a standalone upstream runnable experiment**.
It depends on the external
[HITL robustness repository](https://github.com/M-Colley/hitl-mobo-robustness-tests),
including `scripts/analyse_boba_adaptations.py`, `scripts/replay_mo_front.py`
and their helper dependencies. Raw MOBO archives are not bundled here and
may not be available from a repository clone; obtain the 100 CSVs listed in
[the sanitized provenance](evidence/selection/provenance.json) from the
authors or an accessible archive. Their filenames and original SHA256 hashes
allow identity checks. Summary CSVs alone cannot recreate the original choices.

After obtaining that repository and its logs, restore the historical replay
source to its `scripts/replay_schmitt_decisions.py`, and provide a compatible
OpenBO checkout. From the external repository root, with sibling checkout
layout, run:

```bash
python scripts/replay_schmitt_decisions.py --input-dir output-boba-mo-halo \
  --openbo-src ../openbo/src --output-dir replay-results-new --run
```

The historical helper source hashes are recorded in
`evidence/selection/reproduction-helper-hashes.json`; the numerical versions
and original selector hashes are in the provenance. The complete external
suite command was `python -m pytest tests/ -q --durations=10`; the focused
command was `python -m pytest tests/test_replay_schmitt_decisions.py -q`.
See [validation status](evidence/selection/validation.json) and
[archive integrity](evidence/selection/archive_integrity.json). This bundle
does not package participant data, raw MOBO log archives or generated website
MDX files.
