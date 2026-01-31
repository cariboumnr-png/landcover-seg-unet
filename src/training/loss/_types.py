'''Loss related types.'''

from __future__ import annotations
import typing

class LossTypes(typing.TypedDict, total=False):
    '''Composite config for composite loss.'''
    focal: _FocalConfig
    dice: _DiceConfig

class _FocalConfig(typing.TypedDict):
    '''Parameters needed for focal loss.'''
    weight: float
    alpha: list[float] | None
    gamma: float
    reduction: str

class _DiceConfig(typing.TypedDict):
    '''Parameters needed for dice loss.'''
    weight: float
    smooth: float
