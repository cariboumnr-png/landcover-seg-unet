# ADR-0049: Introduce Unit Test System

**Status:** Accepted
**Date:** 2026-07-08

## 1. Context

As the `landcover-seg-unet` project has grown in complexity and feature set, ensuring
code correctness and preventing regressions during refactoring has become critical.
The project lacked a systematic test suite.

Furthermore:
* A clean separation of test code and source code is preferred, utilizing a
  root-level `tests/` directory rather than colocating test scripts next to source
  code files.
* The codebase consists of both highly stable elements (utilities, core data
  specifications) and rapidly changing, experimental elements (sweeps, model
  architectures). A one-size-fits-all testing strategy would either be too loose
  for stable components or too brittle for volatile components.

## 2. Decision

We have introduced a structured testing framework based on the following decisions:

### 2.1. Framework Selection and Project Integration
* We use **pytest** as the core testing runner and framework due to its
  minimal boilerplate, powerful fixture system, and extensive ecosystem (e.g., mock
  support).
* Project test dependencies (`pytest`, `pytest-mock`, `pytest-cov`) are declared
  under `[project.optional-dependencies]` inside `pyproject.toml`.
* Pytest is configured in `pyproject.toml` to map the `src/` directory
  directly to `sys.path` (`pythonpath = ["src"]`). This allows executing the test
  suite with a simple `pytest` command without requiring editable installation
  during local testing.

### 2.2. Directory Structure
* A root-level `tests/` directory is established.
* The layout of `tests/` mirrors the layout of `src/landseg/` (e.g., unit tests
  for `src/landseg/utils/multip.py` live in `tests/unit/utils/test_multip.py`).

### 2.3. Testing Strategy and Boundaries
To optimize developer velocity and test effectiveness, we classify tests into
three distinct tiers:

1. **Tier 1: Unit Tests (Fast, In-Memory)**: Target stable packages like `utils/`
   and `core/` (e.g., `tests/unit/utils/`, `tests/unit/core/`, and `tests/unit/geopipe/core/`).
   These tests execute in milliseconds, do not interact with external
   assets, and require minimal mocking.
2. **Tier 2: Sub-System/Integration Tests (Medium, Mocked I/O)**: Target
   orchestration components and pipelines (e.g., `tests/unit/geopipe/foundation/data_blocks/`).
   These will not perform heavy model training or read large files. Instead, they use dynamic
   test fixtures (defined in `tests/conftest.py`) or pre-generated dummy files to test integrations.
3. **Tier 3: End-to-End CLI Smoke Tests (Slow)**: Target the CLI interfaces and execution layer pipelines (e.g.,
   `tests/unit/adapters/cli/test_cli.py` and `tests/unit/execution/pipelines/test_data_ingest.py`) to ensure the entry points, configurations (Hydra), and full orchestration layers wire correctly and run without crashing.

### 2.4. Handling Volatile vs. Stable Interfaces
* **Stable Interfaces**: We write detailed, high-coverage unit tests verifying
  edge cases, error conditions, and exact return values.
* **Volatile/Experimental Interfaces**: We write "light-touch" smoke tests.
  These tests assert that the code runs without raising an exception when
  initialized or invoked, avoiding assert statements on exact numerical values
  that are subject to rapid change.

## 3. Consequences

### Positive
* **Regression Protection**: Changes to core utility and datastructure modules
  can be validated immediately.
* **Separation of Concerns**: Production code is kept clean of testing code.
* **No Repository Bloat**: Using dynamic fixtures for GeoTIFF and tensor
  generation ensures that no large binary datasets are checked into version control.
* **Low Boilerplate**: Pytest fixtures and plain assert statements make test
  writing fast and readable.

### Negative
* **Maintenance Overhead**: Adding new features requires updating corresponding
  project dependencies and test code.
* **Initial Setup Execution**: Requires configuring developer environments to
  install test dependencies.

## 4. Implementation Strategy & Modular Roadmap

To balance code quality with developer velocity and avoid maintaining long-lived
branches, this testing framework is deployed incrementally. Rather than attempting
all-encompassing coverage in a single release, we focus on establishing blueprints
and backfilling coverage module-by-module.

### 4.1. Blueprint Representatives in Initial Release
For this initial deployment, we have implemented representative test suites for each
tier to serve as architectural blueprints for future test additions:
* **Tier 1 Blueprint**: Core specifications and utilities (`tests/unit/core/`, `tests/unit/utils/`, and `tests/unit/geopipe/core/`).
* **Tier 2 Blueprint**: Foundation data blocks building pipeline (`tests/unit/geopipe/foundation/data_blocks/`).
* **Tier 3 Blueprint**: CLI entrypoint configuration smoke test (`tests/unit/adapters/cli/test_cli.py`) and E2E execution pipeline test (`tests/unit/execution/pipelines/test_data_ingest.py`).

All initial blueprint tests must pass successfully in the local/remote environment
before this framework is merged to `main`.

### 4.2. Post-Merge / Incremental Module Roadmap
Subsequent test coverage will be added gradually through isolated feature branches. Future testing efforts will be organized and integrated incrementally module-by-module. Pull requests that introduce new features, refactor existing code, or address bugs should prioritize introducing corresponding test coverage for the target module to gradually expand project-wide coverage.
