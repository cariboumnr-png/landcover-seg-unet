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
import landseg.artifacts as artifacts
import landseg.geopipe.contracts as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.harmonize.context as harmonize_context
import landseg.geopipe.harmonize.logger as harmonize_logger
import landseg.geopipe.harmonize.manifest as harmonize_manifest
import landseg.geopipe.harmonize.rasters as harmonize_rasters


# ----- private dataclasses
@dataclasses.dataclass
class _ProcessedRasters:
    '''Container for processed raster paths dictionaries.'''
    provenance: dict[str, str] = dataclasses.field(default_factory=dict)
    harmonized: dict[str, str] = dataclasses.field(default_factory=dict)
    finalized: dict[str, str] = dataclasses.field(default_factory=dict)


# ----- public functions
def run_data_harmonization(
    world_grid_source: str,
    artifacts_paths: artifacts.HarmonizationPaths,
    config: contracts.HarmonizationPipelineConfig,
    *,
    logger: harmonize_logger.HarmonizationLogger
) -> None:
    '''Run data harmonization pipeline.'''

    # build harmonization context
    logger.log('INFO', '[START] Building data harmonization context')
    context = harmonize_context.build_harmonization_context(
        world_grid_source,
        artifacts_paths.runs_manifest,
        config
    )
    logger.set_grid_reference(context.grid_id, context.grid_fpath)
    logger.set_identity(context.current_run_identity)

    # early exit
    if context.collided_run_uid is not None:
        logger.set_summary_status('SKIPPED')
        logger.log(
            'INFO',
            f'[COMPLETE] Harmonization run with the same inputs and configs '
            f'already done (run uid: {context.collided_run_uid}), skipped'
        )
        return
    logger.log('INFO', '[COMPLETE] Data harmonization context built')

    # set up generator - each source to harmonize
    proc = _harmonize_sources(
        context.compiled_dataset_manifest,
        artifacts_paths.effective_run_folder,
        context.grid,
        categorical_resampling=config.resampling_categorical,
        continuous_resampling=config.resampling_continuous,
    )

    # run generator
    logger.log('INFO', f'[START] Harmonizing data onto grid: {context.grid.affine_identity}')
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
        harmonize_rasters.unify_nodata_mask(feature_raster, mask_path)
        logger.set_valid_mask_raster(mask_path)


# ----- private helpers
def _harmonize_sources(
    compiled_sources: dict[str, harmonize_manifest.ManifestEntry],
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

        warped = harmonize_rasters.warp_to_grid(
            input_path=path,
            output_path=out_vrt,
            world_grid=world_grid,
            is_categorical=is_cat,
            resampling_method=resampling,
        )
        # band mapping is now required
        harmonize_rasters.add_band_description_to_vrt(warped, mfst['band_mapping'])

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
        **(yield from harmonize_rasters.stack_rasters(features, labels, output_dir))
    )
    return processed


def _tag_domain_metadata(warped: str, mfst: harmonize_manifest.ManifestEntry) -> None:
    '''Attach domain raster metadata tags to VRT file.'''
    cat_specs = mfst.get('categorical_specs')
    if not cat_specs:
        return

    if 'index_base' in cat_specs:
        harmonize_rasters.add_tag_to_vrt(
            warped,
            index_base=cat_specs['index_base'],
        )


def _tag_feature_metadata(warped: str, mfst: harmonize_manifest.ManifestEntry) -> None:
    '''Attach feature schemes metadata tags to VRT file.'''
    schemes = mfst.get('schemes')
    if schemes:
        harmonize_rasters.add_tag_to_vrt(
            warped,
            schemes={mfst['name']: schemes},
        )


def _tag_label_metadata(warped: str, mfst: harmonize_manifest.ManifestEntry) -> None:
    '''Attach categorical label metadata tags to VRT file.'''
    cat_specs = mfst.get('categorical_specs')
    if not cat_specs:
        raise ValueError('Missing categorical specs for label raster')

    schemes = mfst.get('schemes')
    harmonize_rasters.add_tag_to_vrt(
        warped,
        index_base=cat_specs['index_base'],
        num_cls=cat_specs['num_cls'],
        ignore_cls=cat_specs['ignore_cls'],
        class_name=cat_specs.get('class_name', {}),
        color_map=cat_specs.get('color_map', {}),
        taxonomy=cat_specs.get('taxonomy', {}),
        schemes={mfst['name']: schemes} if schemes else {},
    )
