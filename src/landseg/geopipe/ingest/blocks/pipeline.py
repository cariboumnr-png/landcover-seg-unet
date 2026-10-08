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
Canonical data-block construction pipeline.

Maps input rasters onto a pre-built world grid, materializes immutable
raw data blocks, and maintains the associated catalog and dataset
metadata. This pipeline produces experiment-agnostic artifacts intended
for reuse across downstream workflows.

Public APIs:
    - BlockPipelineRuntimeContext: Context for block pipeline.
    - run_blocks_building: Runs the canonical data block pipeline.
'''

# standard imports
import dataclasses
import time
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.ingest as ingest
import landseg.geopipe.ingest.blocks.assembler as assembler
import landseg.geopipe.ingest.blocks.manifest as manifest
import landseg.geopipe.ingest.blocks.mapper as mapper

# typing aliases
CollisionManifestCtrl = artifacts.Controller[contracts.RunCollisionManifest]


# ----- private types
class _PipelinePaths(typing.Protocol):
    '''Typed pipeline-specific paths container.'''
    @property
    def blocks(self) -> str: ...
    @property
    def catalog(self) -> str: ...
    @property
    def schema(self) -> str: ...
    def mapped_window(self, gid: str) -> str: ...


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class BlockPipelineRuntimeContext:
    '''Run context, e.g., run identity and collision handling.'''
    world_grid: geo_core.GridLayout
    block_artifact_paths: _PipelinePaths
    collisions_artifacts_fpath: str
    harmonize_run_id: str
    ingest_run_id: str


# ----- public functions
def run_blocks_building(
    inputs: assembler.BlockBuildingInputs,
    config: assembler.BlockBuildingConfig,
    context: BlockPipelineRuntimeContext,
    *,
    logger: ingest.IngestionLogger,
) -> None:
    '''
    Build canonical data blocks from rasters aligned to a world grid.

    Materializes immutable `.npz` block artifacts and maintains
    associated `catalog.json` and `schema.json` manifests. Blocks are
    built directly from raster windows without normalization or
    dataset splitting.

    Args:
        world_grid:
            World grid definition used to locate raster windows.
        artfact_paths:
            Container holding paths for blocks, catalog, and schema.
        config:
            Configuration for block building inputs and parameters.
        policy:
            Lifecycle policy governing artifact update behavior.
        logger:
            Logger instance used for structured telemetry reporting.
        collisions_fpath:
            Optional file path to persist the run collision manifest.
    '''
    start_time = time.perf_counter()

    artifact_paths = context.block_artifact_paths

    # map rasters to the provided world grid
    ras_windows = mapper.map_rasters_to_grid(
        context.world_grid,
        inputs.image_fpath,
        inputs.label_fpath,
        artifact_paths.mapped_window(context.world_grid.gid),
        policy=config.artifacts_policy,
    )

    # inspect incumbent catalog if present
    incumbent_catalog: geo_core.DatasetCatalog | None = None
    try:
        cat_dict = artifacts.Controller[dict](artifact_paths.catalog).fetch()
        if cat_dict:
            incumbent_catalog = geo_core.DatasetCatalog.from_dict(cat_dict)
    except artifacts.ArtifactError:
        pass

    # build data blocks
    building_context = assembler.BlockBuildingContext(
        image=ras_windows.image,
        label=ras_windows.label,
        block_size=ras_windows.tile_shape,
        image_band_map=assembler.read_band_map(inputs.image_fpath),
        label_specs=assembler.read_label_specs(inputs.label_fpath),
        incumbent_catalog=incumbent_catalog,
    )
    block_building_result = assembler.build_blocks(
        inputs,
        config,
        building_context,
        output_dir=artifact_paths.blocks,
    )

    # persist collision manifest
    CollisionManifestCtrl(context.collisions_artifacts_fpath).persist({
        'ingestion_run_id': context.ingest_run_id or '',
        'ingestion_run_uid': logger.run_uid,
        'harmonization_run_id': context.harmonize_run_id or '',
        'collision_policy': config.collision_policy,
        'total_collided': len(block_building_result.collided_blocks),
        'collided_blocks': block_building_result.collided_blocks,
    })

    # create/update catalog and metadata JSON
    paths = context.block_artifact_paths
    provenance = manifest.ManifestProvenance(
        grid_id=context.world_grid.gid,
        block_identity=context.world_grid.block_identity,
        source_image=inputs.image_fpath,
        source_label=inputs.label_fpath,
        harmonize_run_id=context.harmonize_run_id,
        ingest_run_id=context.ingest_run_id,
    )
    manifest_report = manifest.update_manifest(
        paths.catalog,
        paths.schema,
        paths.blocks,
        provenance,
        block_building_result,
        policy=config.artifacts_policy,
    )

    stats = block_building_result.running_stats

    logger.set_data_blocks_report({
        'image_filepath': inputs.image_fpath,
        'label_filepath': inputs.label_fpath,
        'duration_sec': time.perf_counter() - start_time,
        'stats': stats,
        'manifest': {
            'catalog_status': manifest_report['catalog_status'],
            'catalog_updated': manifest_report['catalog_updated'],
            'cataloged_blocks_count': manifest_report['cataloged_blocks_count'],
            'schema_updated': manifest_report['schema_updated'],
        },
    })

    logger.info(
        f'Intra-pool block collisions: {stats["blocks_collided"]} '
        f'(policy: {config.collision_policy}, '
        f'added: {stats["blocks_added"]}, '
        f'skipped: {stats["blocks_skipped"]}, '
        f'overwritten: {stats["blocks_overwritten"]})'
    )
