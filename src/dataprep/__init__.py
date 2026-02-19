# pylint: disable=too-many-return-statements
'''
Top-level namespace for dataprep.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BlockCacheBuilder',
    'DataBlock',
    'BuilderConfig',
    'DataWindows',
    'ScoreParams',
    # functions
    'count_label_class',
    'build_data_blocks',
    'map_rasters',
    'prepare_data',
    'score_blocks',
    'validate_geometry',
    'select_val_blocks',
    'split_blocks',
    'get_image_stats',
    'normalize_data_blocks',
    # typing
    'BlockMeta',
    'ImageStats',
    'GeometrySummary',
    'InputConfig',
    'ProcessConfig',
    'IOConfig',
    'DataprepConfigs',
    'BlockScore',
    'OutputConfig',
    'BlockBuildingConfig',
]

# for static check
if typing.TYPE_CHECKING:
    from .config import (InputConfig, OutputConfig, BlockBuildingConfig,
                         IOConfig, ProcessConfig, DataprepConfigs)
    from .mapper import map_rasters, validate_geometry, GeometrySummary, DataWindows
    from .blockbuilder import (DataBlock, BlockMeta, ImageStats, BlockCacheBuilder,
                               BuilderConfig, build_data_blocks)
    from .normalizer import get_image_stats, normalize_data_blocks
    from .splitter import (BlockScore, ScoreParams, score_blocks, select_val_blocks,
                           split_blocks)
    from .pipeline import prepare_data
    from .utils import count_label_class

def __getattr__(name: str):

    if name in ['InputConfig', 'OutputConfig', 'BlockBuildingConfig',
                'IOConfig', 'ProcessConfig', 'DataprepConfigs']:
        return getattr(importlib.import_module('.config', __package__), name)
    if name in ['map_rasters', 'validate_geometry', 'GeometrySummary',
                'DataWindows']:
        return getattr(importlib.import_module('.mapper', __package__), name)
    if name in ['DataBlock', 'BlockMeta', 'ImageStats', 'BlockCacheBuilder',
                'BuilderConfig', 'build_data_blocks']:
        return getattr(importlib.import_module('.blockbuilder', __package__), name)
    if name in ['get_image_stats', 'normalize_data_blocks']:
        return getattr(importlib.import_module('.normalizer', __package__), name)
    if name in ['BlockScore', 'ScoreParams', 'score_blocks', 'select_val_blocks',
                'split_blocks']:
        return getattr(importlib.import_module('.splitter', __package__), name)
    if name in ['prepare_data']:
        return getattr(importlib.import_module('.pipeline', __package__), name)
    if name in ['count_label_class']:
        return getattr(importlib.import_module('.utils', __package__), name)

    raise AttributeError(name)
