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

# pylint: disable=missing-function-docstring

'''
Data harmonization pipeline command implementation.
'''

# standard imports
import dataclasses
import os
import typing
# local imports
import landseg.artifacts.paths as paths
import landseg.geopipe.core as geo_core
import landseg.geopipe.harmonize.common as common
import landseg.geopipe.harmonize.manifest as manifest
import landseg.geopipe.harmonize.rasters as rasters


# ----- private types
class _HarmonizationPipelineConfig(typing.Protocol):
    @property
    def dataset_manifest(self) -> str: ...
    @property
    def resampling_continuous(self) -> str: ...
    @property
    def resampling_categorical(self) -> str: ...


# ----- private dataclasses
@dataclasses.dataclass
class _ProcessedRasters:
    '''Container for processed raster paths dictionaries.'''
    provenance: dict[str, str] = dataclasses.field(default_factory=dict)
    harmonized: dict[str, str] = dataclasses.field(default_factory=dict)
    finalized: dict[str, str] = dataclasses.field(default_factory=dict)


# ----- public functions
def data_harmonization_pipeline(
    artifacts_paths: paths.HarmonizationPaths,
    config: _HarmonizationPipelineConfig,
    world_grid: geo_core.GridLayout,
    *,
    logger: common.HarmonizationLogger
) -> None:
    '''Run data harmonization pipeline.'''
    compiled = manifest.compile_dataset_manifest(config.dataset_manifest)

    proc = _harmonize_sources(
        compiled,
        artifacts_paths.effective_root,
        world_grid,
        categorical_resampling=config.resampling_categorical,
        continuous_resampling=config.resampling_continuous,
    )

    processed: _ProcessedRasters
    while True:
        try:
            log_message = next(proc)
            logger.log('INFO', log_message)
        except StopIteration as s:
            processed = s.value
            break

    # log processed file paths
    for name, path in processed.provenance.items():
        logger.add_source_provenance(name, path)

    for name, path in processed.harmonized.items():
        logger.add_harmonized_source(name, path)

    for name, path in processed.finalized.items():
        logger.add_finalized_raster(name, path)

    # generate valid feature pixel mask if feature raster is provided
    feature_raster = processed.finalized.get('features')
    if feature_raster:
        mask_path = artifacts_paths.valid_mask_raster
        logger.log('INFO', f'Generating valid mask raster: {mask_path}')
        rasters.unify_nodata_mask(feature_raster, mask_path)
        logger.set_valid_mask_raster(mask_path)


# ----- private functions
def _harmonize_sources(
    compiled_sources: dict[str, manifest.ManifestEntry],
    output_dir: str,
    world_grid: geo_core.GridLayout,
    *,
    categorical_resampling: str,
    continuous_resampling: str,
) -> typing.Generator[str, None, _ProcessedRasters]:
    '''Harmonize all compiled raster sources onto the canonical grid.'''
    features: list[str] = []
    labels: list[str] = []
    processed = _ProcessedRasters()

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
