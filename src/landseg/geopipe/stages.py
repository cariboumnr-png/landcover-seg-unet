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
Public stage runner entrypoints for `landseg.geopipe`.

Owns stage logger initialization, context assembly, domain pipeline
execution, and run-ledger updates for each geospatial ETL stage so
`landseg.execution` never imports internal stage loggers, contexts, or
context builders directly.

Public APIs:
    - `run_data_harmonization_stage`: Execute data-harmonize stage.
    - `run_data_ingestion_stage`: Execute data-ingest stage.
    - `run_data_preparation_stage`: Execute data-prepare stage.
    - `run_world_grid_stage`: Execute world-grid generation stage.
'''

# standard imports
import pathlib
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts as contracts
import landseg.geopipe.grid as grid_mod
import landseg.geopipe.harmonize as harmonize_mod
import landseg.geopipe.ingest as ingest_mod
import landseg.geopipe.prepare as prepare_mod


# ----- public functions
def run_world_grid_stage(
    config: contracts.WorldGridPrepConfig,
    grid_paths: artifacts.WorldGridPaths,
) -> None:
    '''
    Execute world grid generation, logging, and report persistence.

    Args:
        config:
            World grid preparation configuration object.
        grid_paths:
            Path manager for world grid artifacts and reports.
    '''
    logger = grid_mod.GridLogger(
        name='world-grid',
        log_file=grid_paths.report,
        enable_file_log=False,
    )
    logger.init_summary(run_id='world-grid')

    try:
        logger.log_sep()
        logger.info('Building/loading canonical world grid')

        is_loaded, grid_fp, world_grid = grid_mod.prepare_world_grid(config)
        status_str = 'loaded' if is_loaded else 'created and persisted'
        logger.set_grid_report(world_grid, grid_fp)

        logger.info(f'[COMPLETE] World grid {status_str}')
        logger.info(f'Grid ID: {world_grid.gid}')
        logger.info(f'Grid artifact file path: {grid_fp}')
        logger.info(f'CRS: {world_grid.crs}')
        logger.info(f'Total Tiles: {len(world_grid)}')

    except Exception as err:
        logger.set_status('FAILED')
        logger.error(f'World grid execution failed: {err}')
        raise

    finally:
        logger.log_sep()
        logger.close()


def run_data_harmonization_stage(
    config: contracts.HarmonizationPipelineConfig,
    harm_paths: artifacts.HarmonizationPaths,
    grid_report_fpath: pathlib.Path,
) -> None:
    '''
    Execute raster harmonization, logging, and ledger update.

    Args:
        config:
            Harmonization pipeline configuration object.
        harm_paths:
            Path manager for harmonization outputs and manifests.
        grid_report_fpath:
            Path to the upstream world grid report JSON.
    '''
    logger = harmonize_mod.HarmonizationLogger(
        name='data-harmonize',
        log_file=harm_paths.report,
        enable_file_log=False,
    )
    logger.init_summary(run_id=harm_paths.run_id)

    try:
        context = harmonize_mod.build_harmonization_context(
            grid_report_fpath,
            harm_paths.runs_manifest,
            config,
        )
        logger.log_sep()
        logger.info('[START] Data harmonization')
        harmonize_mod.run_data_harmonization(
            context,
            harm_paths,
            config,
            logger=logger,
        )
        logger.info('[COMPLETE] Data harmonization')

    except Exception as err:
        logger.set_status('FAILED')
        logger.error(f'Data harmonization failed: {err}')
        raise

    finally:
        logger.update_runs_manifest(
            harm_paths.runs_manifest,
            harm_paths.effective_run_folder,
            config.dataset_manifest,
        )
        logger.log_sep()
        logger.close()


def run_data_ingestion_stage(
    config: contracts.IngestionPipelineConfig,
    harm_paths: artifacts.HarmonizationPaths,
    ingest_paths: artifacts.IngestionPaths,
    *,
    harmonization_run: int | str | None = None,
    rebuild: bool = False,
    harmonization_record: typing.Mapping[str, typing.Any] | None = None,
) -> None:
    '''
    Execute block/domain ingestion, logging, and ledger update.

    Args:
        config:
            Ingestion pipeline configuration object.
        harm_paths:
            Path manager for upstream harmonization artifacts.
        ingest_paths:
            Path manager for ingestion outputs and manifests.
        harmonization_run:
            Optional target harmonization run identifier or index.
        rebuild:
            Whether to force rebuilding existing ingestion outputs.
        harmonization_record:
            Optional pre-resolved harmonization run record override.
    '''
    logger = ingest_mod.IngestionLogger(
        name='data-ingest',
        log_file=ingest_paths.report,
        enable_file_log=False,
    )
    logger.init_summary(run_id=ingest_paths.run_id)

    try:
        hm_record = (
            typing.cast(contracts.HarmonizationRunRecord, harmonization_record)
            if harmonization_record is not None
            else _resolve_target_ingestion_record(
                harm_paths,
                ingest_paths,
                harmonization_run=harmonization_run,
                rebuild=rebuild,
            )
        )
        context = ingest_mod.build_ingestion_context(
            harm_paths,
            hm_record,
            runs_manifest_fpath=ingest_paths.runs_manifest,
            dataset_schema_fpath=ingest_paths.data_blocks.schema,
            config=config,
        )
        logger.log_sep()
        logger.info(
            f'Ingesting harmonization run [{context.harmonization_run_id}]'
            f' ({context.run_uid}) into run [{ingest_paths.run_id}]'
        )
        policy = (
            artifacts.LifecyclePolicy.REBUILD
            if rebuild
            else artifacts.LifecyclePolicy.BUILD_IF_MISSING
        )
        ingest_mod.run_data_ingestion(
            context,
            ingest_paths,
            config,
            policy=policy,
            logger=logger,
        )

    except Exception as err:
        logger.set_status('FAILED')
        logger.exception(f'Ingestion pipeline failed: {err}')
        raise

    finally:
        logger.update_runs_manifest(
            ingest_paths.runs_manifest,
            ingest_paths.effective_run_folder,
        )
        logger.log_sep()
        logger.close()


def run_data_preparation_stage(
    config: contracts.PreparationPipelineConfig,
    ingest_paths: artifacts.IngestionPaths,
    prep_paths: artifacts.PreparationPaths,
) -> None:
    '''
    Execute dataset partitioning, normalization, and schema emission.

    Args:
        config:
            Preparation pipeline configuration object.
        ingest_paths:
            Path manager for upstream ingestion artifacts.
        prep_paths:
            Path manager for dataset preparation outputs.
    '''
    logger = prepare_mod.PreparationLogger(
        name='data-prepare',
        log_file=prep_paths.report,
        enable_file_log=False,
    )
    logger.init_summary(run_id='prepare')

    try:
        context = prepare_mod.build_preparation_context(
            ingest_paths.windows,
            ingest_paths.data_blocks.catalog,
            ingest_paths.data_blocks.schema,
            config,
        )
        logger.log_sep()
        policy = (
            artifacts.LifecyclePolicy.REBUILD
            if config.rebuild
            else artifacts.LifecyclePolicy.BUILD_IF_MISSING
        )
        prepare_mod.run_data_preparation(
            context,
            prep_paths,
            config,
            policy=policy,
            logger=logger,
        )

    except Exception as err:
        logger.set_status('FAILED')
        logger.error(f'Preparation pipeline failed: {err}', exc_info=True)
        raise

    finally:
        logger.log_sep()
        logger.close()


# ----- private helpers
def _resolve_target_ingestion_record(
    harm_paths: artifacts.HarmonizationPaths,
    ingest_paths: artifacts.IngestionPaths,
    *,
    harmonization_run: int | str | None,
    rebuild: bool,
) -> contracts.HarmonizationRunRecord:
    '''Resolve target pending harmonization run record for ingestion.'''
    target = harmonization_run or 'latest'
    batches = ingest_mod.resolve_pending_ingestion_batches(
        harm_paths.runs_manifest,
        ingest_paths.runs_manifest,
        target=target,
        rebuild=rebuild,
    )
    if not batches:
        raise artifacts.ArtifactError(
            f'No pending harmonization runs found for target "{target}".'
        )
    return batches[0]
