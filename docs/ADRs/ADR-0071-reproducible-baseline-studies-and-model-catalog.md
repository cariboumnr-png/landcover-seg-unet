# ADR-0071: Reproducible Baseline Studies and Model Catalog

**Status:** Proposed<br>
**Date:** 2026-11-27<br>
**Target release:** v0.7.0<br>
**Related:**
[ADR-0026](ADR-0026-study-layer-sweep-orchestration.md),
[ADR-0052](ADR-0052-replay-buffer-continual-learning.md),
[ADR-0063](ADR-0063-single-file-artifact-envelopes-and-schema-standardization.md),
[ADR-0067](ADR-0067-decoupled-stage-io-and-auxiliary-experiment-root.md),
[ADR-0070](ADR-0070-artifact-path-handles-and-final-checkpoint-resolution.md)<br>

---

## 1. Context & Problem Statement

As of v0.6.0, the framework can ingest, prepare, train, evaluate, sweep and
register models on Databricks compute. ADR-0070 gave each run a canonical
final checkpoint. What the project cannot yet do is produce a **baseline**:
a result that can be repeated, compared against later framework or
architecture changes, reloaded, and traced back to the exact data, code,
configuration and environment that produced it.

The gaps below were found by reading `main` at `18fd2d6`. None of them has
been reproduced by a training run yet.

### 1.1. Training Is Not Seeded
* The only training-adjacent seed is `study.sweep.seed`, which seeds
  Optuna's `TPESampler` in `study/sweep/optimize.py`.
* No code under `session`, `runtime`, `execution` or the session config
  schema seeds Python, NumPy, PyTorch or DataLoader workers.
* Two runs with the same configuration are therefore not expected to
  produce the same weights or metrics, and seed-to-seed variance cannot be
  measured deliberately.

### 1.2. A Run Folder Does Not Describe Its Own Provenance
* `SessionPaths` declares `checkpoints/`, `logs/`, `plots/`, `previews/`,
  `checkpoints/status.json`, `evaluation.json` and `step_results.json`.
* None of these records the resolved configuration, the `landseg` version
  and git commit, the prepared-data identity, the seed, the compute
  environment, or the tracker run ID.
* Provenance therefore depends on MLflow and notebook state outliving the
  run folder, which ADR-0067's relocatable experiment roots do not
  guarantee.

### 1.3. Study Ranking and Checkpoint Selection Disagree
* `study.sweep.make_objective` returns the validation value of the **last**
  step a trial reported.
* `CheckpointCatalog.final` resolves the **best** checkpoint of the last
  phase (`last_phase_best`).
* A trial can therefore be ranked by one model and represented by another.
  `study.rank_trials` and `mlops.storage` then promote a checkpoint whose
  metric is not the one the study optimized.

### 1.4. Trial and Run Status Is Not Truthful
* In `execution/workflows/study_sweep.py::_build_runner`, the `except`
  branch sets `FAILED` and the `finally` branch then unconditionally sets
  `SUCCESS`, so a failed trial is logged as successful.
* A pruned trial leaves the generator suspended; closing it is expected to
  run the same `finally`, which also records `SUCCESS`.
* The session runners' `finally` (`continuous.py`, `curriculum.py`) calls
  `on_session_end`, which reaches `MLFlowTracker.close` and
  `mlflow.end_run()`. A `KeyboardInterrupt` is therefore expected to be
  recorded as `FINISHED`, which leaves the interrupted-run tagging in
  `runtime.interactive.safe_training_session` unreachable.

### 1.5. Registration Does Not Produce a Reloadable, Linked Model
* `mlops.registry.register_unity_catalog_model` logs the raw `.pt` file as
  an artifact directory and registers `runs:/<id>/model_checkpoint`. That
  URI has no `MLmodel` file, so it is not an MLflow model, and Unity
  Catalog registration is expected to reject it or produce a version that
  `mlflow.pyfunc.load_model` cannot load.
* Unless a run is already active, registration opens a new run named
  `unity_catalog_registration`, so the model version is not linked to the
  training run that produced it.
* `_probe_manifest_hash` looks for `artifacts/manifest.{json,yaml}`,
  `input/manifest.*` and `_hash.json`. None of these match the ADR-0067
  layout (`ingested_data/.../schema.json`, prepared block-split manifests),
  so `manifest_sha256` is likely always `unspecified`.

### 1.6. No Baseline Concept Exists
* `execution/workflows/study_analysis.py` writes the top five trials by
  Optuna value. It does not repeat a configuration across seeds, summarize
  variance, evaluate on the test split, or keep a history.
* There is no durable record that says "baseline B for dataset D under
  framework version V scored X ± s", so a later framework change has
  nothing fixed to be compared against.

---

## 2. Decision & Governing Invariants

We will add a **baseline study** workflow and an **append-only model
catalog**, and harden the run, study and publication layers they depend
on. Hyperparameter search stays available, but it is not required to
produce a baseline.

Subsystem ownership follows ADR-0069 and ADR-0070:

| Subsystem | Owns |
| --- | --- |
| `runtime` | seeding and determinism controls; environment probes |
| `artifacts` | every file and folder name, including run records and catalog entries |
| `session` | writing the run record during training; checkpoint selection |
| `study` | baseline specs, repeated-seed execution, aggregation, catalog entries |
| `mlops` | publishing a catalog entry to MLflow / Unity Catalog |

### Invariant 1: Every Training Run Writes a Run Record
Each run folder holds one `run_record.json` (an ADR-0063 envelope). It is
written when the run is allocated and finalized when the run ends. It
contains:
* the resolved configuration and its fingerprint (`compute_fingerprint`);
* the `landseg` version, git commit and a dirty-tree flag;
* the data identity: fingerprints of the prepared-splits manifest and the
  prepared-data `schema.json`;
* the seed and determinism mode;
* the environment: Databricks runtime, cluster ID, device, torch and CUDA
  versions, and the resolved AMP dtype;
* the terminal status, the final checkpoint (run-relative path), its
  selection metric name and value, and the tracker run ID.

Identity fields are written once and never change; only the status and
result fields are filled in at finalization.

### Invariant 2: Seeds Are Explicit and Recorded
* `session.seed` (required to be an integer once this ADR is implemented)
  seeds Python, NumPy, PyTorch, the DataLoader generator and worker init
  through one `runtime` function.
* An optional `session.deterministic` flag enables PyTorch deterministic
  algorithms. The flag is recorded, because GPU determinism is not
  guaranteed for every kernel.

### Invariant 3: The Study Optimizes What the Catalog Keeps
* The sweep objective returns the selection metric of
  `CheckpointCatalog.final` for the trial's run, not the last reported step.
* Intermediate reports to Optuna for pruning are unchanged.
* The metric name is declared once in configuration and stored in the trial
  user attributes and the run record.

### Invariant 4: Terminal Status Is Truthful Everywhere
A run ends in exactly one of `COMPLETED`, `PRUNED`, `FAILED` or
`INTERRUPTED`. The same value appears in the run record, the session log
summary, `step_results.json` and the MLflow run status.

### Invariant 5: A Baseline Is a Versioned Spec, Not a Search
A baseline spec names a fixed configuration, a list of seeds, the required
data identity, the selection and evaluation metrics, and a representative
policy. Running it trains one run per seed and evaluates each final
checkpoint on the **test** split. The test split is never used for
checkpoint or trial selection.

### Invariant 6: The Catalog Is Append-Only
* Each catalog entry lists its per-seed run records, the mean and standard
  deviation of each metric, the representative run, the spec fingerprint,
  the data identity and the framework version.
* An entry is never edited. A newer entry may reference an older one with
  `supersedes`.
* Each entry states its comparison class: `data_baseline` (new data),
  `framework_regression` (same spec rerun on a new framework version) or
  `architecture_experiment` (changed model or training spec).
* Runs are grouped into one entry only if their configuration fingerprints
  are equal once the seed is excluded and their data identities match.

### Invariant 7: Publication Consumes the Catalog
`mlops` publishes a catalog entry's representative run as an MLflow
`pyfunc` model that bundles the checkpoint and resolved configuration. The
model version is created from the training run, so its lineage points back
to it, and its tags carry the run-record and catalog-entry fingerprints.

---

## 3. Proposed Design

### 3.1. Run Records (`artifacts`, `session`, `runtime`)
* `artifacts.paths.SessionPaths.run_record` → `run_record.json`.
* A `RunRecord` dataclass and a schema-versioned envelope
  (`run_record/v0.1.0`), with relocatable run-relative POSIX paths, as for
  `CheckpointCatalog`.
* `runtime.seed_everything(seed, deterministic=False)` plus a DataLoader
  `worker_init_fn` / `generator` factory.
* `BaseRunner` writes identity fields on start and finalizes status and
  results in its `finally`, with the status passed explicitly rather than
  inferred.

### 3.2. Study Hardening (`study.sweep`, `execution.workflows`)
* The objective reads `CheckpointCatalog.from_run(run).final` and returns
  that checkpoint's recorded metric.
* `_build_runner` maps the outcome to a status: success → `COMPLETED`,
  `optuna.TrialPruned` / `GeneratorExit` → `PRUNED`, other exceptions →
  `FAILED`, `KeyboardInterrupt` → `INTERRUPTED`.
* `MLFlowTracker.close` accepts that status and ends the MLflow run with it.

### 3.3. Baseline Study & Catalog (`study.baseline`, `study.catalog`)
* `configs/study_baseline.yaml` recipe and a `study.baseline` config
  schema: `name`, `seeds`, `selection_metric`, `evaluation_metrics`,
  `representative_policy` (default `median_seed`), `comparison_class`,
  `supersedes`.
* `study.run_baseline(spec, runner_builder)` trains one run per seed,
  skipping seeds that already have a `COMPLETED` run record with the same
  fingerprint (`artifacts.find_run_collision`), so an interrupted baseline
  resumes instead of restarting.
* `study.BaselineCatalog` writes entries under
  `<exp_root>/artifacts/catalog/entries/<entry_id>.json`, plus an
  `index.json` that can be rebuilt from the entries.
* `execution.workflows.study_baseline` provides the CLI entry point, and a
  notebook drives it on Databricks (see §8).

### 3.4. Publication (`mlops.registry`)
* `register_unity_catalog_model` takes a catalog entry ID instead of
  probing for a checkpoint.
* It logs a `pyfunc` wrapper with a signature and input example, inside the
  representative training run (resumed by its tracker run ID).
* `_probe_manifest_hash` is removed; data identity comes from the run
  record.

### 3.5. Implementation Phases
Each phase is one or more scoped commits that leave `main` releasable.
1. Run record, seeding and truthful status (§3.1, §3.2 status mapping).
2. Objective alignment (§3.2).
3. Baseline study and catalog (§3.3).
4. Publication from the catalog (§3.4).
5. On-compute acceptance run and the first `data_baseline` catalog entry.

ADR-0052 (replay buffer) stays Proposed. It needs a catalog entry to use as
its Tier 1 base model and as the historical reference for forgetting.

---

## 4. Acceptance Criteria

Unit tests live in `tests/unit` and integration tests in
`tests/integration` using `landseg.testing.dummy_data`, per
`docs/repo_conventions.md`. The full unit suite must show no new failures
compared with a run on the parent commit before each phase.

| ID | Criterion | Verified by |
| --- | --- | --- |
| AC-1 | Every `ModelTraining` run writes `run_record.json` with all Invariant 1 fields; identity fields are byte-identical before and after finalization. | unit + integration |
| AC-2 | Two CPU runs on dummy data with the same seed and `deterministic=true` produce identical `step_results.json` metrics; two different seeds do not. | integration |
| AC-3 | Success, exception, pruning and `KeyboardInterrupt` each yield the matching status in the run record, session summary and MLflow run (MLflow via a local file tracking URI). | unit |
| AC-4 | For every completed trial, the Optuna trial value equals the `best_metric` of `CheckpointCatalog.final` for that trial's run. | unit (fake runner) |
| AC-5 | A three-seed baseline spec produces three run records and one catalog entry with per-metric mean and standard deviation and a representative run chosen by policy. | integration |
| AC-6 | Rerunning the same spec trains nothing and adds no entry; after an interruption it trains only the missing seeds. | integration |
| AC-7 | The catalog refuses to group runs whose seed-excluded config fingerprints or data identities differ, and refuses to overwrite an existing entry. | unit |
| AC-8 | Selection never reads test-split metrics: a test asserts the selection path has no access to them. | unit |
| AC-9 | A published model version loads in a fresh process with `mlflow.pyfunc.load_model`, and its prediction on a dummy patch matches direct checkpoint inference. Its source run is the training run, and its tags include the run-record and entry fingerprints. | integration (local MLflow) + Databricks |
| AC-10 | `tests/unit/test_boundaries.py` asserts that `study` does not import `mlops`, and that catalog and run-record file names are declared only in `artifacts`. | unit |
| AC-11 | On the target Databricks cluster, the baseline notebook runs end to end with Run All on a UC Volume experiment root, without manual repair; the first `data_baseline` entry is written and published, and wall time per seed is recorded. | manual milestone |
| AC-12 | Rerunning AC-11's spec on the same commit and cluster reproduces the per-seed selection metric within a tolerance stated in the entry (exact on CPU; a documented tolerance on GPU). | manual milestone |

AC-11 and AC-12 together are the definition of "ready for baseline model
training". The ADR moves to Accepted when AC-1 to AC-10 pass in CI, and to
Implemented when AC-11 and AC-12 have been run and recorded.

---

## 5. Consequences

### Positive
* Each result can be traced from a registered model back through the catalog
  entry and the run record to data, code, configuration and environment,
  without relying on notebook or MLflow state.
* A framework change can be judged by rerunning a fixed spec
  (`framework_regression`) instead of by impression.
* Seed variance is measured, so a small difference between two baselines
  can be read against noise.
* ADR-0052 gets a concrete base model and a forgetting reference.

### Trade-offs & Neutral Notes
* Repeated seeds multiply compute cost; the default seed count should stay
  small (three) for the 4-vCPU cluster.
* Requiring a seed is a breaking configuration change, so this ADR targets a
  minor version bump.
* Full determinism on GPU may cost speed; it stays opt-in and recorded.
* Old run folders have no run record. They remain readable through ADR-0070
  handles but cannot join a catalog entry.

---

## 6. Alternatives Considered
* **Use the Optuna study as the catalog.** Rejected: Optuna stores trial
  parameters and values, not data identity, code version or evaluation, and
  a study is tied to one search space.
* **Use MLflow as the only source of truth.** Rejected as primary:
  experiment roots must stay relocatable and usable offline (ADR-0067).
  MLflow stays the publication and browsing layer.
* **Adopt an external tool (for example DVC) for lineage.** Deferred: it
  adds a second storage model on top of UC Volumes; the run record covers
  the immediate need.
* **Tune hyperparameters before establishing a baseline.** Rejected as the
  first step: without seeded, repeatable runs, sweep rankings cannot be told
  apart from noise.

---

## 7. Verification Plan
* Phase 1-4 commits each list the AC IDs they satisfy, with the pytest
  command and counts, in the PR "Verification & Test Suite Results" section.
* AC-11 and AC-12 results (cluster ID, runtime, commit, wall time, entry
  ID, metric values) are appended to this ADR when it is marked Implemented.

---

## 8. Open Questions
1. **Representative policy.** Is `median_seed` (the run closest to the
   median selection metric) acceptable, or should it be the best seed?
2. **Dirty git tree.** Should a baseline refuse to run on uncommitted code,
   or only record the flag?
3. **Catalog location.** Is `<exp_root>/artifacts/catalog/` enough, or
   should entries also be mirrored to a Unity Catalog table for browsing?
4. **Notebook.** Should baselines get a new `04_baseline_study` notebook, or
   a mode in `03_study_sweep`?
5. **Sweep runner mode.** Sweeps currently force `continuous` mode with a
   single phase. Should baseline specs allow curriculum runs from the start?
6. **Tolerance.** What per-seed tolerance should AC-12 accept on GPU?
