# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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

# pylint: disable=missing-class-docstring

'''
Event definitions for the training orchestration system.

This module defines a small set of immutable event types that are emitted
by the training runner and phase policies to describe *what has happened*
during a training run. Events form a linear, observable stream that can
be consumed by loggers, early-stopping policies, checkpoint managers, and
higher-level controllers.

Design notes
------------

- Events are modeled as plain, frozen dataclasses and inherit from a
  common `Event` base class. The base class serves as a semantic marker,
  establishing a clear architectural boundary: only `Event` instances are
  allowed to cross from orchestration into observation and control layers.

- Individual event classes (e.g. `PhaseStart`, `EpochEnd`) define their
  payload explicitly as typed fields rather than relying on a generic
  dictionary. This keeps event semantics self-describing, type-checkable,
  and easy to evolve as phase behavior becomes more explicit and enforced.

- Some event classes use small convenience initializers to derive common
  or generic representations (e.g. for logging or inspection) while
  keeping the call sites concise. Although overriding `__init__` in
  dataclasses is not idiomatic in general, it is used sparingly here to
  preserve clarity at emission sites without introducing mutation or
  hidden behavior.

- Events are *facts*, not callbacks: they carry no executable logic, do
  not mutate shared state, and are never modified after creation. Control
  flow is expressed through distinct event types (e.g. `StopRun`) rather
  than implicit flags embedded in payloads.

This module intentionally avoids encoding execution logic or policy
decisions. It is expected to remain stable even as the training runner
evolves toward a fully generator-based model and additional phase schemas
are introduced.
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing
# local imports
import landseg.core as core

# alias
field = dataclasses.field

# -------------------------------------------------------------------------
# Base event
# -------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Event:
    '''
    Base class for all orchestration events.

    Events are immutable facts emitted by orchestration or engine code.
    They do not perform actions and do not trigger behavior by themselves.
    '''

    name: str
    payload: dict[str, typing.Any] | None = None

# -------------------------------------------------------------------------
# Phase-level events
# -------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class PhaseStart(Event):
    phase_name: str = ''

    def __init__(self, phase_name: str) -> None:
        super().__init__(
            name='phase_start',
            payload={'phase_name': phase_name},
        )
        object.__setattr__(self, 'phase_name', phase_name)

@dataclasses.dataclass(frozen=True)
class PhaseEnd(Event):
    phase_name: str = ''

    def __init__(self, phase_name: str) -> None:
        super().__init__(
            name='phase_end',
            payload={'phase_name': phase_name},
        )
        object.__setattr__(self, 'phase_name', phase_name)

# -------------------------------------------------------------------------
# Epoch-level events
# -------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class EpochStart(Event):
    epoch_index: int = 1
    phase_name: str = ''

    def __init__(self, epoch_index: int, phase_name: str) -> None:
        super().__init__(
            name='epoch_start',
            payload={
                'epoch_index': epoch_index,
                'phase_name': phase_name,
            },
        )
        object.__setattr__(self, 'epoch_index', epoch_index)
        object.__setattr__(self, 'phase_name', phase_name)

@dataclasses.dataclass(frozen=True)
class EpochEnd(Event):
    epoch_index: int = 1
    phase_name: str = ''

    def __init__(
        self,
        epoch_index: int,
        phase_name: str,
    ) -> None:
        super().__init__(
            name='epoch_end',
            payload={
                'epoch_index': epoch_index,
                'phase_name': phase_name,
            },
        )
        object.__setattr__(self, 'epoch_index', epoch_index)
        object.__setattr__(self, 'phase_name', phase_name)

# -------------------------------------------------------------------------
# Report events
# -------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class MetricsReport(Event):
    best_so_far: float = 0.0
    best_epoch: int = -1
    is_best_epoch: bool = False
    raw_metrics: core.EpochResults | None = None

    def __init__(
        self,
        best_so_far: float,
        best_epoch: int,
        is_best_epoch: bool,
        raw_metrics: core.EpochResults,
    ) -> None:
        super().__init__(
            name='tracking_report',
            payload={
                'best_so_far': best_so_far,
                'best_epoch': best_epoch,
                'is_best_epoch': is_best_epoch,
                'raw_metrics': raw_metrics,
            },
        )
        object.__setattr__(self, 'best_so_far', best_so_far)
        object.__setattr__(self, 'best_epoch', best_epoch)
        object.__setattr__(self, 'is_best_epoch', is_best_epoch)
        object.__setattr__(self, 'raw_metrics', raw_metrics)

# -------------------------------------------------------------------------
# Control events
# -------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class StopRun(Event):
    reason: str = ''

    def __init__(self, reason: str) -> None:
        super().__init__(
            name='stop_run',
            payload={'reason': reason},
        )
        object.__setattr__(self, 'reason', reason)

@dataclasses.dataclass(frozen=True)
class CheckpointRequest(Event):
    tag: str = ''

    def __init__(self, tag: str) -> None:
        super().__init__(
            name='checkpoint_request',
            payload={'tag': tag},
        )
        object.__setattr__(self, 'tag', tag)
