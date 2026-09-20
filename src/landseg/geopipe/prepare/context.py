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
DatasetSchemaCtrl = artifacts.Controller[geo_core.DatasetSchema]


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class PreparationContext:
    '''Resolved upstream ingestion references for dataset preparation.'''
    catalog_fpath: str
    schema_fpath: str
    schema: geo_core.DatasetSchema
    canvas_crs: str
    canvas_transform: rasterio.transform.Affine


# ----- public functions
def build_preparation_context(
    catalog_fpath: str,
    schema_fpath: str,
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
    # load ingested data schema
    schema = DatasetSchemaCtrl.load_json_or_fail(schema_fpath).fetch()

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

    return PreparationContext(
        catalog_fpath=catalog_fpath,
        schema_fpath=schema_fpath,
        schema=schema,
        canvas_crs=canvas_crs,
        canvas_transform=canvas_transform,
    )
