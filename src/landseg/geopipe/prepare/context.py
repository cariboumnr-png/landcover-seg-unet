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
Execution context resolution for dataset preparation.

Provides containers and loaders to resolve upstream ingested data
catalog, schema, and canvas spatial reference metadata.

Public APIs:
    - `PreparationContext`: container holding resolved ingestion inputs.
    - `build_preparation_context`: load preparation context from files.
'''

# standard imports
import dataclasses
import pathlib
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.rasters as rasters


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class PreparationContext:
    '''Resolved upstream ingestion references for dataset preparation.'''
    catalog: dict[str, geo_core.DataCatalogEntry]
    test_catalog: dict[str, geo_core.DataCatalogEntry] | None
    schema: geo_core.DataSchema
    block_specs: tuple[int, int, int, int] # (row, col, overlap_r, overlap_c)
    train_aoi: pathlib.Path | None
    val_aoi: pathlib.Path | None
    test_aoi: pathlib.Path | None
    canvas_crs: str
    canvas_transform: rasters.Affine


# ----- public functions
def build_preparation_context(
    windows_fp: pathlib.Path,
    catalog_fp: pathlib.Path,
    schema_fp: pathlib.Path,
    config: contracts.PreparationPipelineConfig,
) -> PreparationContext:
    '''
    Build preparation context from upstream ingestion catalog and schema.

    Loads the canonical dataset schema, validates the catalog path,
    and extracts the canvas CRS and Affine transform from the image
    raster source.

    Args:
        catalog_fpath:
            path to canonical blocks catalog JSON.
        schema_fpath:
            path to dataset schema JSON.

    Returns:
        PreparationContext:
            resolved execution context containing schema and canvas
            metadata.
    '''
    # load artifacts
    Ctrl = artifacts.Controller[contracts.MappedRasterWindowsPayload]
    windows = Ctrl(windows_fp).fetch()

    Ctrl = artifacts.Controller[geo_core.DataSchema]
    schema = Ctrl(schema_fp, schema_id=geo_core.data_schema.SCHEMA_ID).fetch()

    Ctrl = artifacts.Controller[dict[str, geo_core.DataCatalogEntry]]
    catalog = Ctrl(catalog_fp).fetch()
    test_catalog = None
    if config.datasetview.test_catalog is not None:
        test_catalog = Ctrl(config.datasetview.test_catalog).fetch()

    # resolve canvas crs and transform from image source
    image_paths = schema['dataset']['data_source']['image_paths']
    image_raster_info = rasters.get_info(pathlib.Path(image_paths[0]))

    # block specs from windows
    blk_size = windows.get('tile_shape')
    blk_overlap = windows.get('tile_overlap')
    if not (
        isinstance(blk_size, list) and len(blk_size) == 2 and
        isinstance(blk_overlap, list) and len(blk_overlap) == 2
    ): # sanity
        raise ValueError(f'Invalid block windows artifact: {windows_fp}')

    splits = config.partition

    return PreparationContext(
        catalog=catalog,
        test_catalog=test_catalog,
        schema=schema,
        block_specs=(blk_size[0], blk_size[1], blk_overlap[0], blk_overlap[1]),
        train_aoi=pathlib.Path(splits.train_aoi) if splits.train_aoi else None,
        val_aoi=pathlib.Path(splits.val_aoi) if splits.val_aoi else None,
        test_aoi=pathlib.Path(splits.test_aoi) if splits.test_aoi else None,
        canvas_crs=str(image_raster_info.crs),
        canvas_transform=image_raster_info.transform,
    )
