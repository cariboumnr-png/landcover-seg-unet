'''
Top-level namespace for `landseg.dataprep`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    # functions
    'build_schema',
    'prepare_data',
    'schema_from_a_block',
    # typing
    'BlockBuildingConfig',
    'DataprepConfigs',
    'InputConfig',
    'IOConfig',
    'OutputConfig',
    'ProcessConfig',
]

# for static check
if typing.TYPE_CHECKING:
    from .config import (
        BlockBuildingConfig,
        DataprepConfigs,
        InputConfig,
        IOConfig,
        OutputConfig,
        ProcessConfig,
    )
    from .pipeline import prepare_data
    from .schema import build_schema, schema_from_a_block

def __getattr__(name: str):

    if name in ['BlockBuildingConfig', 'DataprepConfigs', 'InputConfig',
                'IOConfig', 'OutputConfig', 'ProcessConfig',]:
        return getattr(importlib.import_module('.config', __package__), name)
    if name in ['prepare_data']:
        return getattr(importlib.import_module('.pipeline', __package__), name)
    if name in ['build_schema', 'schema_from_a_block']:
        return getattr(importlib.import_module('.schema', __package__), name)

    raise AttributeError(name)
