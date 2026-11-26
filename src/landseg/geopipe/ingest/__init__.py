# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

'''
Top-level namespace for `landseg.geopipe.ingest`.

Coordinates the ingestion of harmonized geospatial rasters into
tiled domain maps and canonical data blocks, providing logging,
context resolution, and pipeline execution tools via lazy module
resolution.

Public APIs:
    - `IngestionContext`: Loaded execution context with grid.
    - `IngestionLogger`: Structured logger for ingestion stages.
    - `build_ingestion_context`: Load ingestion context from report.
    - `resolve_pending_ingestion_batches`: Determine batches to ingest.
    - `run_data_ingestion`: Pipeline runner.
    - `validators`: Preflight validation namespace.
'''

# pylint: disable=import-outside-toplevel

# standard imports
from __future__ import annotations
import pathlib
import typing
# local imports
import landseg._lazy as _lazy


class _Validators:
    '''Validation namespace for ingestion preflight checks.'''

    @staticmethod
    def feature_specs(
        image_fpath: pathlib.Path,
        label_fpath: pathlib.Path | None = None,
        *,
        add_spectral: list[str] | None = None,
        add_topo: list[str] | None = None,
    ) -> None:
        '''
        Validate feature engineering specs against input rasters.

        Args:
            image_fpath:
                Path to the source image raster.
            label_fpath:
                Optional path to the source label raster.
            add_spectral:
                Optional list of spectral indices to derive.
            add_topo:
                Optional list of topographic features to derive.

        Raises:
            ValueError:
                If requested features are unsupported or required bands
                are missing.
        '''
        import landseg.geopipe.ingest.blocks.assembler.io as io
        ctx = io.RasterReadContext.from_rasters(image_fpath, label_fpath)
        ctx.validate_feature_specs(add_spectral, add_topo)

    @staticmethod
    def pool_compatibility(
        schema_fpath: pathlib.Path,
        grid_fpath: pathlib.Path | geo_core.GridLayout,
    ) -> None:
        '''
        Validate compatibility between a pool schema and grid layout.

        Args:
            schema_fpath:
                Path to the existing pool dataset schema JSON.
            grid_fpath:
                Path to the grid layout file or `GridLayout` instance.

        Raises:
            artifacts.ArtifactError:
                If the existing pool schema has an incompatible grid or
                block identity.
        '''
        import landseg.geopipe.ingest.context as context
        context.verify_pool_grid_compatibility(schema_fpath, grid_fpath)

    @staticmethod
    def raster_alignment(
        image_fpath: pathlib.Path,
        grid_fpath: pathlib.Path | geo_core.GridLayout,
        label_fpath: pathlib.Path | None = None,
    ) -> None:
        '''
        Validate raster CRS, resolution, and grid alignment.

        Args:
            image_fpath:
                Path to the source image raster.
            grid_fpath:
                Path to the grid layout file or `GridLayout` instance.
            label_fpath:
                Optional path to the source label raster.

        Raises:
            ValueError:
                If rasters are invalid, CRSs or resolutions mismatch,
                extents do not overlap, or CRS disagrees with grid.
        '''
        import landseg.geopipe.core as geo_core
        import landseg.geopipe.ingest.blocks.mapper as mapper
        import landseg.geopipe.rasters as rasters
        grid = (
            grid_fpath
            if isinstance(grid_fpath, geo_core.GridLayout)
            else geo_core.GridLayout.from_fpath(grid_fpath)
        )
        with rasters.read_rasters(image_fpath, label_fpath) as (img, lbl):
            mapper.validate_raster_alignment(grid, img, lbl)

validators = _Validators()


__all__ = [
    'IngestionContext',
    'IngestionLogger',
    'build_ingestion_context',
    'resolve_pending_ingestion_batches',
    'run_data_ingestion',
    'validators',
]


# for static check
if typing.TYPE_CHECKING:
    import landseg.geopipe.core as geo_core

    from .context import (
        IngestionContext,
        build_ingestion_context,
        resolve_pending_ingestion_batches,
    )
    from .logger import IngestionLogger
    from .pipeline import run_data_ingestion


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.context': [
        'IngestionContext',
        'build_ingestion_context',
        'resolve_pending_ingestion_batches',
    ],
    '.logger': ['IngestionLogger'],
    '.pipeline': ['run_data_ingestion'],
})
