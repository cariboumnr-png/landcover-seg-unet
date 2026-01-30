# pylint: disable=missing-function-docstring, too-few-public-methods
'''
Common runtime Protocols shared across models and training.
'''

from __future__ import annotations
import typing

# Public data summary protocols
class DataSummaryHeads(typing.Protocol):
    '''Protocol for local needed by heads.'''
    @property
    def meta(self) -> _Meta:...
    @property
    def heads(self) -> _Head:...

class DataSummaryLoader(typing.Protocol):
    '''Protocol for local needed by dataloader.'''
    @property
    def data(self) -> _Data: ...
    @property
    def dom(self) -> _Dom: ...

class DataSummaryFull(DataSummaryLoader, DataSummaryHeads, typing.Protocol):
    '''Protocol for full `DataSummary`.'''

# internal pieces
class _Meta(typing.Protocol):
    '''DataSummary meta part.'''
    dataset_name: str
    ignore_index: int
    img_ch_num: int

class _Head(typing.Protocol):
    '''DataSummary head part.'''
    class_counts: dict[str, list[int]]
    logits_adjust: dict[str, list[float]]
    topology: dict[str, dict[str, typing.Any]]

class _Data(typing.Protocol):
    '''DataSummary dataloaders part.'''
    train: dict[str, str]
    val: dict[str, str]
    infer: dict[str, str] | None

class _Dom(typing.Protocol):
    '''DataSummary domain part.'''
    data: dict[str, dict[str, int | list[float]]] | None
    meta: dict[str, dict[str, str | int | list]] | None
