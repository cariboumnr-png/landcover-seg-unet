# ADR-0049: Introduce Unit Test System

**Status:** Proposed
**Date:** 2026-07-03

## 1. Context

As the `landcover-seg-unet` project has grown in complexity and feature set, ensuring
code correctness and preventing regressions during refactoring has become critical.
The project currently lacks a systematic test suite.

Furthermore:
* A clean separation of test code and source code is preferred, utilizing a
  root-level `tests/` directory rather than colocating test scripts next to source
  code files.
* The codebase consists of both highly stable elements (utilities, core data
  specifications) and rapidly changing, experimental elements (sweeps, model
  architectures). A one-size-fits-all testing strategy would either be too loose
  for stable components or too brittle for volatile components.

## 2. Decision

We will introduce a structured testing framework based on the following decisions:

### 2.1. Framework Selection and Project Integration
* We will use **pytest** as the core testing runner and framework due to its
  minimal boilerplate, powerful fixture system, and extensive ecosystem (e.g., mock
  support).
* Project test dependencies (`pytest`, `pytest-mock`, `pytest-cov`) will be declared
  under `[project.optional-dependencies]` inside `pyproject.toml`.
* Pytest will be configured in `pyproject.toml` to map the `src/` directory
  directly to `sys.path` (`pythonpath = ["src"]`). This allows executing the test
  suite with a simple `pytest` command without requiring editable installation
  during local testing.

### 2.2. Directory Structure
* A root-level `tests/` directory will be created.
* The layout of `tests/` will mirror the layout of `src/landseg/` (e.g., unit tests
  for `src/landseg/utils/multip.py` will live in `tests/unit/utils/test_multip.py`).

### 2.3. Testing Strategy and Boundaries
To optimize developer velocity and test effectiveness, we will classify tests into
three tiers:

1. **Tier 1: Unit Tests (Fast, In-Memory)**: Target stable packages like `utils/`
   and `core/`. These tests execute in milliseconds, do not interact with external
   assets, and require minimal mocking.
2. **Tier 2: Sub-System/Integration Tests (Medium, Mocked I/O)**: Target
   orchestration components like configurators. These will not perform heavy model
   training or read large files. Instead, they will use dynamic test fixtures
   (defined in `tests/conftest.py`) to generate tiny, in-memory dummy GeoTIFFs
   and data structures.
3. **Tier 3: End-to-End CLI Smoke Tests (Slow)**: Target the CLI interfaces to
   ensure the entry points wire correctly and run without crashing on toy
   configurations.

### 2.4. Handling Volatile vs. Stable Interfaces
* **Stable Interfaces**: We will write detailed, high-coverage unit tests verifying
  edge cases, error conditions, and exact return values.
* **Volatile/Experimental Interfaces**: We will write "light-touch" smoke tests.
  These tests will assert that the code runs without raising an exception when
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
