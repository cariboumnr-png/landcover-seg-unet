# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
from __future__ import annotations
import dataclasses
# third-party imports
import rasterio
import rasterio.errors
import rasterio.transform
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core


# ----- typing aliases
CatalogCtrl = (artifacts.Controller[dict[str, geo_core.DatasetBlockMeta]])
DatasetSchemaCtrl = artifacts.Controller[geo_core.DatasetSchema]
DictCtrl = artifacts.Controller[dict]


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class PreparationContext:
    '''Resolved upstream ingestion references for dataset preparation.'''
    catalog: dict[str, geo_core.DatasetBlockMeta]
    test_catalog: dict[str, geo_core.DatasetBlockMeta] | None
    schema: geo_core.DatasetSchema
    block_specs: tuple[int, int, int, int] # (row, col, overlap_r, overlap_c)
    canvas_crs: str
    canvas_transform: rasterio.transform.Affine


# ----- public functions
def build_preparation_context(
    windows_fpath: str,
    catalog_fpath: str,
    schema_fpath: str,
    *,
    test_catalog_fpath: str | None = None,
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
    windows = DictCtrl.load_json_or_fail(windows_fpath).fetch()
    schema = DatasetSchemaCtrl.load_json_or_fail(schema_fpath).fetch()
    catalog = CatalogCtrl.load_json_or_fail(catalog_fpath).fetch()
    test_catalog = None
    if test_catalog_fpath is not None:
        test_catalog = CatalogCtrl.load_json_or_fail(test_catalog_fpath).fetch()

    # resolve canvas crs and transform from image source
    image_paths = schema['dataset']['data_source']['image_paths']
    try:
        with rasterio.open(image_paths[0]) as src:
            if src.crs is None:
                raise ValueError(f'Raster has no CRS: {image_paths[0]}')
            canvas_crs = src.crs.to_string()
            canvas_transform = src.transform
    except rasterio.errors.RasterioError as e:
        raise ValueError(f'Error reading image: {image_paths[0]}') from e

    # block specs from windows
    block_size: list[int] | None = windows.get('tile_shape')
    block_overlap: list[int] | None = windows.get('tile_overlap')
    if not (
        isinstance(block_size, list) and len(block_size) == 2 and
        isinstance(block_overlap, list) and len(block_overlap) == 2
    ): # sanity
        raise ValueError(f'Invalid block windows artifact: {windows_fpath}')

    return PreparationContext(
        catalog=catalog,
        test_catalog=test_catalog,
        schema=schema,
        block_specs=(*block_size, *block_overlap),
        canvas_crs=canvas_crs,
        canvas_transform=canvas_transform,
    )
