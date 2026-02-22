# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods
'''Trainer config protocols.'''

from __future__ import annotations
# standard imports
import typing

# ---------------------------trainer runtime config---------------------------
@typing.runtime_checkable
class RuntimeConfigLike(typing.Protocol):
    schedule: ScheduleLike
    precision: PrecisionLike
    optim: OptimConfigLike
    monitor: MonitorLike

@typing.runtime_checkable
class ScheduleLike(typing.Protocol):
    max_epoch: int
    max_step: int | None
    logging_interval: int
    eval_interval: int | None
    checkpoint_interval: int | None
    patience_epochs: int | None
    min_delta: float | None

@typing.runtime_checkable
class MonitorLike(typing.Protocol):
    enabled: tuple[str, ...]
    metric: str
    head: str
    mode: str

@typing.runtime_checkable
class PrecisionLike(typing.Protocol):
    use_amp: bool

@typing.runtime_checkable
class OptimConfigLike(typing.Protocol):
    grad_clip_norm: float | None
