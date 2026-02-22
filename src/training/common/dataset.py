# pylint: disable=missing-function-docstring, too-few-public-methods
'''
Trainer-facing dataset specifications protocol.
'''

from __future__ import annotations
import typing

# Public data summary protocols
class DataSpecsLike(typing.Protocol):
    '''Protocol for full `DataSummary`.'''
    @property
    def meta(self) -> _Meta:...
    @property
    def heads(self) -> _Head:...
    @property
    def splits(self) -> _Splits: ...
    @property
    def domains(self) -> _Domains: ...

# --------------------------------private  type--------------------------------
# internal pieces
class _Meta(typing.Protocol):
    '''DataSummary meta part.'''
    dataset_name: str
    fit_perblk_bytes: int
    test_perblk_bytes: int
    ignore_index: int
    img_ch_num: int
    test_blks_grid: tuple[int, int]

class _Head(typing.Protocol):
    '''DataSummary head part.'''
    class_counts: dict[str, list[int]]
    logits_adjust: dict[str, list[float]]
    topology: dict[str, dict[str, typing.Any]]

class _Splits(typing.Protocol):
    '''DataSummary dataloaders part.'''
    train: dict[str, str]
    val: dict[str, str]
    test: dict[str, str] | None

class _Domains(typing.Protocol):
    '''DataSummary domain part.'''
    train: _Dom
    val: _Dom
    test: _Dom
    ids_max: int
    vec_dim: int

    class _Dom(typing.TypedDict):
        '''doc'''
        ids_domain: dict[str, int] | None
        vec_domain: dict[str, list[float]] | None
