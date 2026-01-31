# pylint: disable=missing-function-docstring, too-few-public-methods
'''
Common runtime Protocols shared across models and training.
'''

from __future__ import annotations
import typing

# Public data summary protocols
class DataSummaryLike(typing.Protocol):
    '''Protocol for full `DataSummary`.'''
    @property
    def meta(self) -> _Meta:...
    @property
    def heads(self) -> _Head:...
    @property
    def data(self) -> _Data: ...
    @property
    def doms(self) -> _Dom: ...

# internal pieces
class _Meta(typing.Protocol):
    '''DataSummary meta part.'''
    dataset_name: str
    train_val_blk_bytes: int
    infer_blk_bytes: int
    ignore_index: int
    img_ch_num: int

class _Head(typing.Protocol):
    '''DataSummary head part.'''
    class_counts: dict[str, list[int]]
    logits_adjust: dict[str, list[float]]
    topology: dict[str, dict[str, typing.Any]]

class _Data(typing.Protocol):
    '''DataSummary dataloaders part.'''
    train: dict[str, str] | None
    val: dict[str, str] | None
    infer: dict[str, str] | None

class _Dom(typing.Protocol):
    '''DataSummary domain part.'''
    train_val_domain: dict[str, dict[str, int | list[float]]] | None
    train_val_meta: dict[str, dict[str, str | int | list]] | None
    infer_domain: dict[str, dict[str, int | list[float]]] | None
    infer_meta: dict[str, dict[str, str | int | list]] | None
