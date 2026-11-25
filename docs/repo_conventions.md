# Repository Hygiene & Engineering Conventions

This document establishes the unified coding standards, architectural boundaries, documentation guidelines, and repository hygiene practices for the `landseg` project.

These conventions apply to all contributions—whether written by human developers or autonomous AI agents—to ensure the codebase remains maintainable, consistent, readable, and legally compliant.

---

## Table of Contents

1. [Crown Copyright & Legal Compliance](#1-crown-copyright--legal-compliance)
2. [Code Formatting & Structural Style](#2-code-formatting--structural-style)
   - [Line Length Limits](#line-length-limits)
   - [Imports Organization](#imports-organization)
   - [Multi-Line Wrapping & Package Namespaces](#multi-line-wrapping--package-namespaces)
   - [Vertical Spacing & Blank Lines](#vertical-spacing--blank-lines)
   - [Section Delimiters & Script Ordering](#section-delimiters--script-ordering)
   - [Comments Formatting](#comments-formatting)
3. [Documentation & Docstring Specifications](#3-documentation--docstring-specifications)
   - [Module-Level Docstrings](#module-level-docstrings)
   - [Public Functions & Methods](#public-functions--methods)
   - [Dataclasses & Private Helpers](#dataclasses--private-helpers)
4. [Typing Protocols & Static Analysis](#4-typing-protocols--static-analysis)
   - [Typing Protocols](#typing-protocols)
   - [Pylint & Code Quality Philosophy](#pylint--code-quality-philosophy)
5. [Testing Architecture & Standards](#5-testing-architecture--standards)
   - [Unit vs. Integration Boundary](#unit-vs-integration-boundary)
   - [No "Mock Theater"](#no-mock-theater)
   - [Fixture Scope & Placement](#fixture-scope--placement)
   - [Pathlib Contracts](#pathlib-contracts)
   - [Centralized Lazy Import Testing](#centralized-lazy-import-testing)
   - [Targeted Test Execution](#targeted-test-execution)
6. [Repository Hygiene & Contribution Workflow](#6-repository-hygiene--contribution-workflow)
   - [Virtual Environment Execution](#virtual-environment-execution)
   - [Scratch & Analysis Space (`sandbox/`)](#scratch--analysis-space-sandbox)
   - [Diff Hygiene & Scoped Changes](#diff-hygiene--scoped-changes)
   - [Architectural Decision Records (ADRs)](#architectural-decision-records-adrs)
   - [Git Commit Message Conventions](#git-commit-message-conventions)
   - [Pull Request (PR) Descriptions](#pull-request-pr-descriptions)
7. [Quick Reference Checklist](#7-quick-reference-checklist)

---

## 1. Crown Copyright & Legal Compliance

Every Python source file (`.py`) in this repository must begin with the verbatim 20-line Ontario Crown Copyright header block.

* **Strict Requirement**: The header block must remain **completely untouched and verbatim** at lines 1–20 of every file.
* **No Modifications**: Do not alter phrasing, reformat borders, update years, or remove license text.

```python
# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #
```

---

## 2. Code Formatting & Structural Style

### Line Length Limits

* **Code Lines**: Must not exceed **80 characters**.
  * *Pragmatic exception*: If a line exceeds 80 characters by only a couple characters (e.g., 81–83 characters) and wrapping would severely impair readability, leave it unwrapped rather than introducing awkward line breaks.
* **Docstrings and Comments**: Must not exceed **72 characters**.
  * Docstrings and comments must wrap cleanly to preserve terminal rendering and side-by-side diff legibility.

### Imports Organization

Imports must be organized into exactly three categories separated by explicit, lowercase category comments:

1. `# standard imports`
2. `# third-party imports`
3. `# local imports`

#### Rules & Constraints:
* **No blank lines between categories**: Category comments immediately follow the preceding import block without empty lines.
* **No `np` or `pd` Aliases**: Never use abbreviated import aliases like `import numpy as np` or `import pandas as pd`. Always use direct module imports: `import numpy` and `import pandas`.
* **Trailing Spacing after Imports**:
  * If followed by a **top-level function, fixture, class, or section delimiter**: Leave **exactly two blank lines**.
  * If followed by a **module-level global variable or alias**: Leave **exactly one blank line** (then leave two blank lines after the variable before any functions or section delimiters).

#### Example:
```python
# standard imports
import pathlib
import typing
# third-party imports
import numpy
import pandas
import torch
# local imports
import landseg.configs as configs
import landseg.runtime as runtime


# ----- public functions
def compute_metrics(...) -> dict:
    ...
```

### Multi-Line Wrapping & Package Namespaces

#### One-or-Vertical Rule (No Line Bin-Packing):
Never bin-pack multiple items across wrapped lines to fill the 80-character limit. For `from ... import ...` statements, `_lazy.attach(...)` symbol lists, and `submodules` lists:

* **0 or 1 item (within 80 characters)**: Keep inline on a single line (e.g., `from .logger import Logger`, `'.logger': ['Logger'],`).
* **2 or more items (or 1 item > 80 characters)**: Expand to multi-line with **strictly one item per line**, sorted alphabetically, with a **trailing comma** on every item (including the last).

#### Package `__init__.py` Namespaces (`__all__` & `landseg._lazy.attach`):
* **`__all__` Formatting & Ordering**: Always format `__all__` multi-line with **one symbol per line** and trailing commas (even for single-item lists). Do not add `# group` category comments; sort entries in **flat ASCII alphabetical order** (`sorted()`), which naturally sequences `UPPER_SNAKE` constants, `PascalCase` classes/types, and `snake_case` functions/submodules.
* **Lazy Resolution (`_lazy.attach`)**: Package `__init__.py` files exposing submodules or symbols must use `_lazy.attach` from `landseg._lazy` rather than hand-written `__getattr__` chains:
  * Import `import landseg._lazy as _lazy` under `# local imports` (include `from __future__ import annotations` only if runtime annotations reference `TYPE_CHECKING`-only types).
  * Mirror exports under `if typing.TYPE_CHECKING:` ordered alphabetically by relative module path and formatted per the One-or-Vertical rule (including multi-submodule `from . import (...)`).
  * Order `exports` dictionary keys alphabetically by relative module path (`'.mod': ...`), placing each entry on its own line with its symbol list formatted per the One-or-Vertical rule.
  * When `submodules` is passed (or `exports` is `{}`), wrap `_lazy.attach(...)` arguments one per line and apply the One-or-Vertical rule to `submodules=[...]`.

#### Example:
```python
# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'CORE_N',
    'Logger',
    'get_device_name',
    'resolve_device',
    'validator',
]


# for static check
if typing.TYPE_CHECKING:
    from . import validator
    from .concurrency import CORE_N
    from .hardware import (
        get_device_name,
        resolve_device,
    )
    from .logger import Logger


__getattr__, __dir__ = _lazy.attach(
    __name__,
    {
        '.concurrency': ['CORE_N'],
        '.hardware': [
            'get_device_name',
            'resolve_device',
        ],
        '.logger': ['Logger'],
    },
    submodules=['validator'],
)
```

### Vertical Spacing & Blank Lines

* **Top-Level Definitions**: Leave **exactly two blank lines** between top-level classes, functions, fixtures, and section delimiters.
* **Class Methods**: Leave **exactly one blank line** between methods within a class definition.
* **Docstring Trailing Spacing**: Leave **no blank lines** immediately after a function/method docstring before the first line of executable code.

```python
def example_function(value: int) -> int:
    '''
    Calculate transformed output.

    Args:
        value:
            Input integer value.

    Returns:
        int:
            Transformed value.
    '''
    result = value * 2
    return result
```

### Section Delimiters & Script Ordering

Section delimiters provide standardized visual boundaries and navigation landmarks across business modules and test suites.

* **Delimiter Format**: `# ----- `[Symbol]` [lowercase description/action]`
  * If the first word in the description is a class or code symbol, wrap it in backticks and preserve its exact casing (e.g., `# ----- `HeadManager` initialization`).
* **Delimiter Spacing**:
  * Preceded by **exactly two blank lines**.
  * Followed by **no blank lines** (the target definition immediately succeeds it).

#### Business Script Section Ordering:
When applicable, business Python modules should sequence definitions in the following order (typing aliases immediately succeed imports):

1. `# ----- typing aliases`
2. `# ----- public types`
3. `# ----- private types`
4. `# ----- public dataclasses`
5. `# ----- private dataclasses`
6. `# ----- public classes`
7. `# ----- private classes`
8. `# ----- public functions`
9. `# ----- private helpers`

### Comments Formatting

* **Standard Words**: Comments must start with a lowercase letter (e.g., `# verify shape`, `# standard imports`).
* **Code Identifiers & Class Names**: When referencing a class, function, or module-level type, preserve its original casing and enclose it in backticks:
```
# `LayerNorm` standard deviation check`
# dispatch to `execute_command`
```
* **Spacing**: Always place exactly one space between the `#` symbol and the comment text.

---

## 3. Documentation & Docstring Specifications

All docstrings must adhere to the 72-character line length limit.

### Module-Level Docstrings

Every Python module requires a top-level docstring placed immediately below the Crown Copyright header.

* Structure:
  1. A concise one-liner summary.
  2. One or more short paragraphs explaining context, architecture, and purpose.
  3. A `Public APIs:` section listing exported symbols and their brief descriptions in **flat ASCII alphabetical order** (`sorted()`), with no category sub-headers, matching `__all__` 1-to-1 when present.

```python
'''
Programmatic API entry surface.

Provides the primary orchestrator entry points for invoking training,
evaluation, preflight readiness checks, and intake pipelines.

Public APIs:
    - `run`: Run execution pipeline with resolved configuration.
    - `run_experiment`: Run full experiment lifecycle.
    - `run_preflight`: Run pre-flight diagnostic validation checks.
'''
```

### Public Functions & Methods

Public functions follow Google-style docstrings with arguments formatted on indented newlines:

* Structure:
  1. One-liner summary, followed by an optional descriptive paragraph.
  2. `Args:` block where each parameter name is followed by a colon, and its description is placed on a new indented line.
  3. `Returns:` block with the type annotation and description on a new indented line.
  4. `Raises:` block (if applicable) documenting exceptions.

```python
def configure_stage(target: str, strict: bool = False) -> StageConfig:
    '''
    Configure execution stage with target parameters.

    Resolves pipeline recipes and validates compatibility with
    registered stage manifests.

    Args:
        target:
            Identifier of target execution pipeline or stage.
        strict:
            Whether to enforce strict schema validation.

    Returns:
        StageConfig:
            Resolved and validated stage configuration instance.

    Raises:
        ValueError:
            If target identifier is unregistered or invalid.
    '''
    ...
```

### Dataclasses & Private Helpers

Dataclasses, private helper functions, and test setup helpers should use a concise one-liner docstring:

```python
@dataclasses.dataclass(frozen=True)
class GridCoordinate:
    '''Immutable spatial grid coordinate representation.'''
    x: int
    y: int


def _resolve_cache_key(prefix: str, identifier: str) -> str:
    '''Compute deterministic cache key from input metadata.'''
    ...
```

---

## 4. Typing Protocols & Static Analysis

### Typing Protocols

When defining interfaces via `typing.Protocol`:

* **Compact Layout**: Keep protocol class definitions compact with **no blank lines** within the class body.
* **No Docstrings on Stubs**: Do not place docstrings on individual method or property stubs inside the protocol.
* **Attribute Properties**: Define protocol attributes using `@property` stubs with explicit return type annotations.
* **Pylint Pragma**: Include `# pylint: disable=missing-function-docstring` near the top of the file when protocol definitions omit method docstrings.

```python
class SessionConfigShape(typing.Protocol):
    '''Interface for session construction configuration.'''
    @property
    def dataloader(self) -> data.DataLoaderConfig: ...
    @property
    def engine(self) -> engine_mod.EngineConfigShape: ...
    @property
    def orchestration(self) -> orchestration_mod.OrchestrationConfigShape: ...
```

### Pylint & Code Quality Philosophy

* **No Score Chasing / Pragma Spam**: Do not force a 10.0 Pylint score by spamming `# pylint: disable` flags across codebase files.
* **Pragmatic Disables**: Only use disable pragmas when strictly warranted (e.g., `# pylint: disable=protected-access` in unit tests accessing internal states, or `# pylint: disable=missing-function-docstring` for protocols).
* **Fix the Root Cause**: Address underlying typing, architectural, or modularization issues instead of masking valid linter diagnostics.

---

## 5. Testing Architecture & Standards

### Unit vs. Integration Boundary

The test suite enforces a strict structural boundary:

* `tests/unit/`:
  * Fast, isolated, deterministic unit tests.
  * Tests pure domain logic, mathematical transformations, configuration schemas, and isolated step contracts.
  * **Prohibited in unit tests**: Heavy filesystem I/O, multi-step pipeline dispatch, external network requests, or cross-subsystem orchestration.
* `tests/integration/`:
  * End-to-end workflows and multi-step execution pipelines.
  * Validates pipeline execution, artifact creation on disk, and runtime orchestration.
  * Must be driven by real, canonical synthetic data generated via `landseg.testing.dummy_data`.

### No "Mock Theater"

* Avoid asserting on private method calls or constructing deep mock verification chains (e.g., avoid `assert_called_once_with` on private helper methods).
* Test **observable public contracts**: return values, state changes, generated disk artifacts, and system invariants.
* Use mocks primarily at true system boundaries (e.g., hardware accelerators, external cloud storage, subprocess calls).

### Fixture Scope & Placement

* **Shared Fixtures**: Must reside strictly in hierarchy-appropriate `conftest.py` files:
  * `tests/conftest.py`: Root-level fixtures.
  * `tests/unit/conftest.py`: Unit-test-wide fixtures.
  * `tests/unit/<subsystem>/conftest.py`: Subsystem-specific shared fixtures.
  * `tests/integration/conftest.py`: Integration-wide test data fixtures.
* **No Script-Local Fixtures**: Do **NOT** define `@pytest.fixture` functions inside individual `test_*.py` files.
* **Private Setup Helpers**: Convert test-local setup or factory logic into private helper functions prefixed with an underscore (`_`), placed under `# ----- private helpers` at the bottom of the test script.

### Pathlib Contracts

* Consistently use `pathlib.Path` across all fixtures, test inputs, file operations, and path validations.
* Avoid converting paths to strings unless required by third-party external boundaries or JSON serialization payloads.

### Centralized Lazy Import Testing

* Do not add individual unit tests for `__init__.py` lazy import resolution inside subsystem test modules.
* Package-wide lazy imports are tested centrally in `tests/unit/test_lazy_imports.py`.

### Targeted Test Execution

* During iterative development or refactoring, execute only the targeted unit tests directly relevant to the affected modules.
* Running the full test suite across the entire project is reserved for milestone verifications, pre-commit checks, and CI validation.

---

## 6. Repository Hygiene & Contribution Workflow

### Virtual Environment Execution

The project requires Python 3.12+ and runs inside a dedicated virtual environment (`.venv`).

* All testing, formatting, and script execution must use the virtual environment binaries:
  * **Windows**: `.\.venv\Scripts\python.exe` and `.\.venv\Scripts\pytest`
  * **Linux / macOS**: `.venv/bin/python` and `.venv/bin/pytest`

### Scratch & Analysis Space (`sandbox/`)

To prevent repository pollution and accidental commits of intermediate outputs:

* The `sandbox/` directory is git-ignored (`/sandbox/*` in `.gitignore`).
* Place all temporary investigation scripts, exploratory notebooks, ad-hoc debug sessions, and experimental artifacts inside `sandbox/`.
* Never create untracked scratch directories or leave ad-hoc scripts in the workspace root.
* Alternatively, perform one-off inspections via ephemeral inline commands (e.g., `python -c "..."`).

### Diff Hygiene & Scoped Changes

* Keep pull requests and commits tightly scoped to the specific feature, fix, or refactoring objective.
* **Avoid Gratuitous Churn**: Do not perform sweeping reformatting, whitespace adjustments, or docstring updates on untouched legacy files.
* Maintain clean git history and preserve git blame clarity.

### Architectural Decision Records (ADRs)

New architectural designs and breaking structural changes require an ADR in `docs/ADRs/`.

* **Proposed ADRs**:
  * Filename: `docs/ADRs/ADR-XXXX-<descriptive-slug>.md`
  * Header: `**Status:** Proposed`
  * Tone: Prospective ("we will...", "will be introduced...", "will execute...").
* **Accepted ADRs**:
  * Once the proposal is fully implemented, verified, and merged:
  * Header: `**Status:** Accepted` (or `**Status:** Accepted — Implemented`)
  * Tone: Past tense ("we have...", "was introduced...").

### Git Commit Message Conventions

Commit messages must be clear, descriptive, and structured as follows:

* **Header**: A concise, imperative one-line summary (maximum 72 characters).
* **Body**: Bullet points detailing key modifications, formatted as:
  * Bullet items start with `* `.
  * Bullet text begins with a lowercase letter (unless starting with a backticked code identifier).
  * Any code symbol, class name, function, or module name must be enclosed in backticks (e.g., `* add `resolve_target_reclass` to `landseg.geopipe``).

#### Example:
```text
refactor dataset manifest assembly pipeline

* introduce `ManifestBuilder` protocol in `landseg.geopipe`
* update `resolve_target_reclass` to accept optional category mapping
* migrate manifest serialization to use `pathlib.Path` contracts
* add unit tests for schema validation in `tests/unit/geopipe`
```

### Pull Request (PR) Descriptions

* **No Local Absolute URIs**: Never include local file paths or environment URIs (e.g., `file:///...`) in PR bodies or commit logs. Use repository-relative paths (e.g., `docs/ADRs/...`, `src/landseg/...`).
* **Structured Description**: Structure PR descriptions with:
  1. **Title**: Concise summary of change.
  2. **Executive Summary**: High-level problem statement and solution.
  3. **Key Modifications**: Bulleted list of changes grouped by subsystem.
  4. **Verification & Test Suite**: Commands run and test pass confirmation.

---

## 7. Quick Reference Checklist

Before submitting code, verify adherence to this checklist:

| Checkpoint | Requirement | Reference |
| :--- | :--- | :--- |
| **Crown Copyright** | Verbatim 20-line header at lines 1–20 of every `.py` file | [Section 1](#1-crown-copyright--legal-compliance) |
| **Imports** | 3 categories, lowercase headers, no blank lines between categories, no `np`/`pd` | [Section 2](#imports-organization) |
| **Wrapping & `__all__`** | One-or-vertical (never bin-pack); flat ASCII-sorted `__all__`, no group comments | [Section 2](#multi-line-wrapping--package-namespaces) |
| **Line Length** | Code $\le$ 80 chars; docstrings and comments $\le$ 72 chars | [Section 2](#line-length-limits) |
| **Vertical Spacing** | 2 blank lines top-level; 1 between methods; 0 after docstring | [Section 2](#vertical-spacing--blank-lines) |
| **Section Delimiters** | `# ----- `[Symbol]` [action]`, 2 blank lines before, 0 after | [Section 2](#section-delimiters--script-ordering) |
| **Comments** | Start with lowercase letter or `` `Symbol` ``, 1 space after `#` | [Section 2](#comments-formatting) |
| **Docstrings** | Module docstring with ASCII-sorted `Public APIs:`; Google-style function docs | [Section 3](#3-documentation--docstring-specifications) |
| **Typing Protocols** | Compact, no blank lines, `@property` stubs, top-level pylint disable | [Section 4](#typing-protocols) |
| **Pylint** | No score chasing or pragma spam; clean architecture first | [Section 4](#pylint--code-quality-philosophy) |
| **Test Boundary** | Fast unit tests in `tests/unit/`; e2e pipelines in `tests/integration/` | [Section 5](#unit-vs-integration-boundary) |
| **Fixtures** | Shared fixtures in `conftest.py`; test helpers in `_helper` functions | [Section 5](#fixture-scope--placement) |
| **Path Handling** | Consistently use `pathlib.Path` over raw strings | [Section 5](#pathlib-contracts) |
| **Scratch Space** | All exploratory/debug scripts placed in git-ignored `sandbox/` | [Section 6](#scratch--analysis-space-sandbox) |
| **Diff Hygiene** | Tightly scoped changes; no unsolicited refactoring sweeps | [Section 6](#diff-hygiene--scoped-changes) |
| **Commit Message** | One-line header, `* ` bullets starting lowercase, backticked symbols | [Section 6](#git-commit-message-conventions) |
| **PR Description** | Repository-relative paths, no local `file:///` URIs, structured body | [Section 6](#pull-request-pr-descriptions) |
