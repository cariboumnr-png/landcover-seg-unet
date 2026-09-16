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
Data harmonization processor for geospatial raster sources.

This module processes compiled dataset manifest entries, warping rasters
onto the canonical world grid, attaching band mappings and categorical
metadata, and producing composite stacked VRT outputs.

Public APIs:
    - `ProcessedRasters`: Container for processed raster paths.
    - `harmonize_sources`: Harmonize all compiled raster sources onto grid.
'''

# standard imports
from __future__ import annotations
import dataclasses
import os
import typing
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.harmonize.manifest as manifest
import landseg.geopipe.harmonize.rasters as rasters


# ----- public dataclasses
@dataclasses.dataclass
class ProcessedRasters:
    '''Container for processed raster paths dictionaries.'''
    provenance: dict[str, str] = dataclasses.field(default_factory=dict)
    harmonized: dict[str, str] = dataclasses.field(default_factory=dict)
    finalized: dict[str, str] = dataclasses.field(default_factory=dict)


# ----- public functions
def harmonize_sources(
    compiled_sources: dict[str, manifest.ManifestEntry],
    output_dir: str,
    world_grid: geo_core.GridLayout,
    *,
    categorical_resampling: str,
    continuous_resampling: str,
) -> typing.Generator[str, None, ProcessedRasters]:
    '''
    Harmonize all compiled raster sources onto the canonical grid.

    Args:
        compiled_sources:
            Mapping of source paths to compiled ManifestEntry objects.
        output_dir:
            Directory path where harmonized VRT files are saved.
        world_grid:
            Canonical world grid layout used for spatial alignment.
        categorical_resampling:
            Resampling algorithm name for categorical rasters.
        continuous_resampling:
            Resampling algorithm name for continuous rasters.

    Yields:
        str:
            Status messages during harmonization.

    Returns:
        ProcessedRasters:
            Container of provenance, harmonized, and finalized paths.
    '''
    features: list[str] = []
    labels: list[str] = []
    processed = ProcessedRasters()

    for path, mfst in compiled_sources.items():
        if not mfst:
            raise ValueError(f'No configuration found for raster {path}')

        is_cat = mfst['category'] in {'domains', 'domain', 'labels', 'label'}
        tagged_name = f'{mfst["category"]}_{mfst["name"]}'
        resampling = (
            categorical_resampling if is_cat else continuous_resampling
        )
        out_vrt = os.path.join(output_dir, f'{tagged_name}.vrt')

        yield (
            f'Harmonizing raster {path} -> {out_vrt} '
            f'(resampling: {resampling})'
        )

        warped = rasters.warp_to_grid(
            input_path=path,
            output_path=out_vrt,
            world_grid=world_grid,
            is_categorical=is_cat,
            resampling_method=resampling,
        )
        # band mapping is now required
        rasters.add_band_description_to_vrt(warped, mfst['band_mapping'])

        processed.provenance[tagged_name] = os.path.abspath(path)
        processed.harmonized[tagged_name] = warped

        match mfst['category']:
            case 'domains' | 'domain':
                _tag_domain_metadata(warped, mfst)
                processed.finalized[tagged_name] = warped

            case 'features' | 'feature':
                _tag_feature_metadata(warped, mfst)
                features.append(warped)

            case 'labels' | 'label':
                _tag_label_metadata(warped, mfst)
                labels.append(warped)

    processed.finalized.update(
        **(yield from rasters.stack_rasters(features, labels, output_dir))
    )
    return processed


# ----- private helpers
def _tag_domain_metadata(warped: str, mfst: manifest.ManifestEntry) -> None:
    '''Attach domain raster metadata tags to VRT file.'''
    cat_specs = mfst.get('categorical_specs')
    if not cat_specs:
        return

    if 'index_base' in cat_specs:
        rasters.add_tag_to_vrt(
            warped,
            index_base=cat_specs['index_base'],
        )


def _tag_feature_metadata(warped: str, mfst: manifest.ManifestEntry) -> None:
    '''Attach feature schemes metadata tags to VRT file.'''
    schemes = mfst.get('schemes')
    if schemes:
        rasters.add_tag_to_vrt(
            warped,
            schemes={mfst['name']: schemes},
        )


def _tag_label_metadata(warped: str, mfst: manifest.ManifestEntry) -> None:
    '''Attach categorical label metadata tags to VRT file.'''
    cat_specs = mfst.get('categorical_specs')
    if not cat_specs:
        raise ValueError('Missing categorical specs for label raster')

    schemes = mfst.get('schemes')
    rasters.add_tag_to_vrt(
        warped,
        index_base=cat_specs['index_base'],
        num_cls=cat_specs['num_cls'],
        ignore_cls=cat_specs['ignore_cls'],
        class_name=cat_specs.get('class_name', {}),
        color_map=cat_specs.get('color_map', {}),
        taxonomy=cat_specs.get('taxonomy', {}),
        schemes={mfst['name']: schemes} if schemes else {},
    )
