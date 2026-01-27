'''Protocol for trainer checkpointing.'''

from __future__ import annotations
# standard imports
import typing

# checkpoint metadata
class CheckpointMetaLike(typing.TypedDict):
    '''Checkpont metadata'''
    metric: float
    epoch: int
    step: int
