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

# pylint: disable=missing-function-docstring

'''
Catalog adapter utilities.

Provides helpers to load and filter a canonical blocks catalog and schema
to extract class counts and file paths needed for downstream sampling
and analysis.
'''

# standard imports
import dataclasses
import typing
# third-party imports
import rasterio
import rasterio.errors
import rasterio.transform
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core

# typing aliases
CatalogDictCtrl = artifacts.Controller[dict[str, geo_core.CatalogEntry]]


class _CatalogViewConfig(typing.Protocol):
    '''Typed configuration container for catalog views.'''
    @property
    def valid_pxs(self) -> dict[str, float]: ...
    @property
    def test_catalog(self) -> str | None: ...
    @property
    def non_overlapping_test_grid(self) -> bool: ...


@dataclasses.dataclass(frozen=True)
class DataBlocksView:
    '''High-level view of data blocks for partitioning.'''
    valid_blocks: dict[tuple[int, int], str]
    external_test_blocks: list[str] | None
    crs: str
    transform: rasterio.transform.Affine


def read_catalog(
    catalog_fpath: str,
    data_schema: geo_core.DataSchema,
    config: _CatalogViewConfig,
) -> DataBlocksView:
    '''
    Load and adapt canonical blocks into a structured view for
    partitioning.

    Filters blocks based on a minimum valid-pixel threshold, derives
    class counts, and optionally incorporates external holdout test
    blocks.

    Args:
        catalog_fpath: Path to canonical blocks catalog JSON.
        data_schema: Ingested dataset schema instance.
        config: Catalog view configuration.

    Returns:
        A `DataBlocksView` containing filtered metadata for
        partitioning.
    '''
    # retrieve image paths and shape from schema
    image_paths = data_schema['dataset']['data_source']['image_paths']
    image_shape = data_schema['tensor_shapes']['image']

    # resolve canvas crs and transform from image source
    try:
        with rasterio.open(image_paths[0]) as src:
            if src.crs is None:
                raise ValueError(f'Raster has no CRS: {image_paths[0]}')
            canvas_crs = src.crs.to_string()
            canvas_transform = src.transform
    except rasterio.errors.RasterioError as e:
        raise ValueError(f'Error reading image: {image_paths[0]}') from e

    # valid blocks filtered by pixel thresholds
    valid_blocks = _filter_blocks(catalog_fpath, config.valid_pxs)

    # blocks on base grid (no stride/overlap)
    row_size, col_size = image_shape['H'], image_shape['W']
    base_coords = [
        k for k, v in valid_blocks.items()
        if v['row_col'][0] % row_size == 0 and v['row_col'][1] % col_size == 0
    ]

    # parse external test data catalog if provided
    if config.test_catalog is not None:
        test_blocks = _filter_blocks(config.test_catalog, config.valid_pxs)
        if config.non_overlapping_test_grid:
            test_blocks = list(
                v['file_path'] for k, v in test_blocks.items()
                if k in base_coords
            )
        else:
            test_blocks = list(
                v['file_path'] for v in test_blocks.values()
            )
    else:
        test_blocks = None

    return DataBlocksView(
        valid_blocks={k: v['file_path'] for k, v in valid_blocks.items()},
        external_test_blocks=test_blocks,
        crs=canvas_crs,
        transform=canvas_transform,
    )


# ----- private helpers
def _filter_blocks(
    fpath: str,
    valid_px_thresholds: dict[str, float],
) -> dict[tuple[int, int], geo_core.CatalogEntry]:
    '''Parse a catalog JSON into filtered class counts and file paths.'''

    def _is_valid_block(
        valid_thresholds: dict[str, float],
        valid_ratios: dict[str, float]
    ) -> bool:
        for k, v in valid_ratios.items():
            threshold = valid_thresholds.get(k)
            if threshold and v < threshold:
                return False
        return True

    catalog_dict = CatalogDictCtrl.load_json_or_fail(fpath).fetch()
    catalog = geo_core.DataCatalog.from_dict(catalog_dict)

    valid_catalog = {
        k: v for k, v in catalog.items()
        if _is_valid_block(valid_px_thresholds, v['valid_px_ratios'])
    }
    return valid_catalog
