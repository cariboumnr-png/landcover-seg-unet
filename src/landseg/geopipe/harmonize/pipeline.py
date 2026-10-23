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

# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts as contracts
import landseg.geopipe.harmonize.context as harmonize_context
import landseg.geopipe.harmonize.logger as harmonize_logger
import landseg.geopipe.harmonize.processor as harmonize_processor
import landseg.geopipe.rasters as rasters


# ----- public functions
def run_data_harmonization(
    context: harmonize_context.HarmonizationContext,
    artifacts_paths: artifacts.HarmonizationPaths,
    config: contracts.HarmonizationPipelineConfig,
    *,
    logger: harmonize_logger.HarmonizationLogger
):
    '''Run data harmonization pipeline.'''
    logger.set_grid_reference(context.grid_id, context.grid_fpath)
    logger.set_identity(context.current_run_identity)

    # early exit
    if context.collided_run_uid is not None:
        logger.set_status('SKIPPED')
        logger.info(
            f'[COMPLETE] Harmonization run with the same inputs and configs '
            f'already done (run uid: {context.collided_run_uid}), skipped'
        )
        return

    # set up generator and run for each source
    logger.info(f'[START] Harmonizing data onto grid: {context.grid.affine_identity}')

    processor = harmonize_processor.harmonizer(
        context.compiled_dataset_manifest,
        artifacts_paths.effective_run_folder,
        dst_crs=context.grid.crs,
        dst_transform=context.grid.transform,
        dst_height=context.grid.h,
        dst_width=context.grid.w,
        categorical_resampling=config.resampling_categorical,
        continuous_resampling=config.resampling_continuous,
    )

    processed: harmonize_processor.HarmonizedRasters
    while True:
        try:
            log_message = next(processor)
            logger.info(log_message)
        except StopIteration as s:
            processed = s.value
            break

    # generate valid feature pixel mask if feature raster is provided
    feature_raster = processed.finalized.get('features')
    if feature_raster:
        mask_path = artifacts_paths.valid_mask_raster
        logger.info(f'Generating valid mask raster: {mask_path}')
        rasters.unify_nodata_mask(feature_raster, mask_path)
        logger.set_valid_mask_raster(mask_path)

    # log processed file paths
    for name, path in processed.provenance.items():
        logger.add_source_provenance(name, path)

    for name, path in processed.harmonized.items():
        logger.add_harmonized_source(name, path)

    for name, path in processed.finalized.items():
        logger.add_finalized_raster(name, path)
