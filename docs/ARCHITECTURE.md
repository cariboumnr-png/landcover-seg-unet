# Architecture Overview

**Last updated: 2026-06-26**

---

## Purpose

A deterministic, artifact-driven system for preparing geospatial data
and running reproducible training sessions.

This document defines **how the system fits together** — not details.

---

## Core Flow

    Raw Data → Foundation → Artifacts → Experiment → Session → Execution → Results

---

## Layers (What each owns)

### Foundation
- Pure construction (grid, domains, blocks)
- No persistence, no policy
- Fully deterministic

---

### Artifacts
- Persistence, validation, lifecycle rules
- Centralized via `artifacts.controller`
- Decides: build vs reuse vs overwrite

---

### Experiment
- Defines *what to train*
- `DataSpecs` (dataset contract)
- Model construction

---

### Session
- Defines *how to run*
- Assembles dataloaders, model frames, optimizers, and tracking instrumentation
- Orchestrated via continuous or curriculum runners
- Executes epoch policies and batch-level tasks (losses, metrics, regularizations)

---

### Configuration (CLI vs. API)
- CLI: Flat, pipeline-oriented inputs (`configs/user.yaml`) translated dynamically
- API: Notebook-first configurators with typed, programmatically safe helper methods
- Core Schemas: Kept strictly isomorphic and isolated from user-facing ergonomics

---

### Execution
- Defines *when and which pipeline runs*
- Resolves configuration + artifact lifecycles
- Delegates to factories (does not build internals)

---

## Key Rules

- No implicit recomputation — everything goes through artifacts
- No persistence in foundation
- No construction in pipelines
- Session is fully assembled before execution
- DataSpecs is the only dataset contract
- User config mappings and programmatic configurators are decoupled from core schemas

---

## Mental Model

- **Foundation builds data**
- **Artifacts make it reusable**
- **Experiment defines training**
- **Session runs it**
- **Configuration adapts it**
- **Execution orchestrates it**

---

## Guarantees

- Deterministic outputs given same inputs + config
- No silent rebuilds
- Clear separation: build-time vs runtime
- Isomorphic schemas: no validation/normalization logic leaks into core classes
