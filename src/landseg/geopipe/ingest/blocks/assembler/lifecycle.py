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
Orchestrator for building groups of blocks in parallel.

This module coordinates multi-block construction pipelines. It
intersects image and label raster windows, validates block file
integrity on disk (cleaning up corrupted files), and schedules
parallel block-generation jobs via the project's ParallelExecutor.

Public APIs:
    - `BlockBuildingInput`: Dataclass of I/O paths for block construction.
    - `BlockBuildingContext`: Dataclass of mapped read windows.
    - `BlockBuildingConfig`: Dataclass for block building configurations.
    - `BlockBuildingOutput`: Dataclass of block building results and stats.
    - `build_blocks`: Validate on-disk blocks and create missing blocks.
'''

# standard imports
from __future__ import annotations
import dataclasses
import os
import random
import typing
# third-party imports
import numpy
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.ingest.blocks.assembler.builder as builder
import landseg.geopipe.ingest.blocks.assembler.io as io
import landseg.geopipe.utils as geo_utils
import landseg.utils as utils


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class BlockBuildingInputs:
    '''I/O paths used during block construction.'''
    image_fpath: str
    label_fpath: str | None

    @property
    def has_label(self) -> bool:
        '''Return `True` is label raster is provided.'''
        return bool(self.label_fpath) and os.path.exists(self.label_fpath)


@dataclasses.dataclass(frozen=True)
class BlockBuildingConfig:
    '''Container for block building configurations.'''
    dem_pad_px: int
    ignore_index: int
    add_spectral: list[str] | None
    add_topo: list[str] | None
    artifacts_policy: artifacts.LifecyclePolicy
    collision_policy: str | contracts.CollisionPolicyType


@dataclasses.dataclass(frozen=True)
class BlockBuildingContext:
    '''Runtime facts and state for the pipeline execution.'''
    image: geo_core.RasterWindowDict
    label: geo_core.RasterWindowDict
    block_size: tuple[int, int]     # block size in row, col
    image_band_map: dict[str, int]
    label_specs: dict[str, geo_core.CategoricalSpec]
    incumbent_catalog: geo_core.DatasetCatalog | None = None


@dataclasses.dataclass(frozen=True)
class BlockBuildingOutput:
    '''Results and statistics from a multi-block building execution.'''
    coords_created: list[tuple[int, int]]
    collided_blocks: list[contracts.CollisionRecord]
    label_color_map: dict[str, list[int]] | None
    running_stats: contracts.BlocksBuildingStats


# ----- private dataclasses
@dataclasses.dataclass(frozen=True)
class _StructuralValidationResults:
    coords_todo: list[tuple[int, int]]
    removed_count: int
    on_disk_before: int
    collided_records: list[contracts.CollisionRecord]

    @property
    def collided_n(self) -> int:
        '''Return number of collided blocks.'''
        return len(self.collided_records)

    @property
    def skipped_n(self) -> int:
        '''Return number of skipped blocks.'''
        return len([
            r for r in self.collided_records
            if r['action_taken'] == 'skipped'
        ])

    @property
    def overwritten_n(self) -> int:
        '''Return number of the overwritten blocks.'''
        return len([
            r for r in self.collided_records
            if r['action_taken'] == 'overwritten'
        ])


# ----- public functions
def build_test_block(
    save_dpath: str,
    inputs: dict[str, io.RasterReadInput],
    *,
    target_head: str,
    valid_px_per: float,
    need_all_classes: bool,
) -> str | None:
    '''
    Build, normalize, and persist a single valid block for testing.

    Iterates over the available block inputs in a deterministic shuffled
    order to find the first block meeting the validity and label-coverage
    criteria. The selected block is normalized using its own image mean
    and standard deviation before being saved.

    Args:
        save_dpath:
            Directory where the test block will be written.
        inputs:
            Mapping from block name to raster inputs.
        target_head:
            Label head used when checking class coverage.
        valid_px_per:
            Minimum required proportion of valid image pixels.
        need_all_classes:
            Whether every class in the target head must be present for a
            block to be accepted.

    Returns:
        str | None:
            Path to the saved test block if found, otherwise None.
    '''
    shuffled_inputs = list(inputs.items())
    random.Random(42).shuffle(shuffled_inputs)

    selected_name = None
    selected_candidate = None

    for name, candidate_input in shuffled_inputs:
        print('Searching for a valid block...', end='\r', flush=True)

        try:
            config = builder.DataBlockConfig(candidate_input.image_dem_pad_px)
            candidate = _build_single_block(name, candidate_input, config)
            manifest = candidate.manifest

            # check valid pixel ratios based on image band
            valid_ratio = manifest['valid_ratios'].get('image', 0.0)

            # check if all classes are present
            has_all_classes = True
            if target_head in manifest['label_count']:
                has_all_classes = all(manifest['label_count'][target_head])
            else:
                has_all_classes = False

            if (
                valid_ratio >= valid_px_per and
                (has_all_classes or not need_all_classes)
            ):
                selected_name = name
                selected_candidate = candidate
                break

        except ValueError:
            continue

    if selected_candidate is None:
        return None

    candidate = selected_candidate
    name = selected_name

    # In-place image normalization for debugging block
    mean = numpy.mean(candidate.data.image)
    std = numpy.std(candidate.data.image)
    candidate.data.image = (candidate.data.image - mean) / (std or 1.0)

    os.makedirs(save_dpath, exist_ok=True)
    fpath = os.path.join(save_dpath, f'test_{name}.npz')
    candidate.save(fpath)
    return fpath


def build_blocks(
    inputs: BlockBuildingInputs,
    config: BlockBuildingConfig,
    context: BlockBuildingContext,
    *,
    output_dir: str,
) -> BlockBuildingOutput:
    '''
    Validate on-disk blocks, clear corrupt ones, and build missing.

    Args:
        inputs:
            I/O paths container for input rasters and output directory.
        context:
            Mapped read windows for image and label rasters.
        config:
            Block building configuration container.
        policy:
            Lifecycle policy determining rebuild or build-if-missing.

    Returns:
        BlockBuildingOutput:
            Execution output containing created coordinates and stats.
    '''
    os.makedirs(output_dir, exist_ok=True)

    # prepare raster read windows
    coords_to_check = _prep_block_windows(inputs, context, output_dir)

    # inspect existing blocks
    validation_results = _structural_validation(
        coords_to_check,
        context.incumbent_catalog,
        artifacts_policy=config.artifacts_policy,
        collision_policy=config.collision_policy,
    )

    # create blocks if missing
    todo = validation_results.coords_todo
    _create_missing_blocks(inputs, todo, config, context, output_dir)

    # extract label color map from specs if present and pass through
    label_color_map: dict[str, list[int]] | None = None
    if context.label_specs:
        for spec in context.label_specs.values():
            if 'color_map' in spec and spec['color_map']:
                label_color_map = spec['color_map']
                break

    running_stats: contracts.BlocksBuildingStats = {
        'blocks_candidate': len(coords_to_check),
        'blocks_on_disk_before': validation_results.on_disk_before,
        'blocks_collided': validation_results.collided_n,
        'blocks_skipped': validation_results.skipped_n,
        'blocks_overwritten': validation_results.overwritten_n,
        'blocks_added': len(coords_to_check) - validation_results.collided_n,
        'damaged_blocks_removed': validation_results.removed_count
    }

    return BlockBuildingOutput(
        coords_created=validation_results.coords_todo,
        collided_blocks=validation_results.collided_records,
        label_color_map=label_color_map,
        running_stats=running_stats,
    )


# ----- private helpers
def _prep_block_windows(
    inputs: BlockBuildingInputs,
    context: BlockBuildingContext,
    output_dir: str,
) -> dict[tuple[int, int], str]:
    '''Find coordinates matching block size and compute window counts.'''
    if inputs.has_label:
        common_coords = set(context.image.keys()) & set(context.label.keys())
    else:
        common_coords = set(context.image.keys())

    valid_coords = set(common_coords) # copy
    for coord in common_coords:
        iw = context.image[coord]
        lw = context.label[coord] if inputs.has_label else None
        if (iw.height, iw.width) != context.block_size or (
            lw is not None and (lw.height, lw.width) != context.block_size
        ):
            valid_coords.remove(coord)

    return {
        c: os.path.join(output_dir, f'{geo_utils.xy_name(c)}.npz')
        for c in valid_coords
    }


def _structural_validation(
    input_coords: dict[tuple[int, int], str],
    incumbent_catalog: geo_core.DatasetCatalog | None = None,
    *,
    artifacts_policy: artifacts.LifecyclePolicy,
    collision_policy: str | contracts.CollisionPolicyType = 'skip',
) -> _StructuralValidationResults:
    '''Verify block file integrity and resolve intra-pool collisions.'''
    results: list[dict[tuple[int, int], bool]] = utils.ParallelExecutor().run(
        [(io.check_npz_integrity, (c, p), {}) for c, p in input_coords.items()],
        ' - Checking existing data blocks (.npz files)'
    )

    collided_coords: list[tuple[int, int]] = []
    coords_todo: list[tuple[int, int]] = []
    removed_count = 0
    on_disk_before = 0

    for block_integity in results:
        c, intact = next(iter(block_integity.items()))
        if intact:
            on_disk_before += 1
            collided_coords.append(c)
        else:
            try:
                os.remove(input_coords[c])
                removed_count += 1
            except FileNotFoundError:
                pass
            coords_todo.append(c)

    # evaluate collisions against collision policy
    if collided_coords and collision_policy == 'error':
        raise artifacts.ArtifactError(
            f'Intra-pool block collision detected for '
            f'{len(collided_coords)} blocks under collision policy '
            f'"error": {[geo_utils.xy_name(c) for c in collided_coords[:5]]}'
        )

    match artifacts_policy:
        case artifacts.LifecyclePolicy.REBUILD:
            action: typing.Literal['overwritten', 'skipped'] = 'overwritten'
            coords_todo = list(input_coords)

        case artifacts.LifecyclePolicy.BUILD_IF_MISSING:
            if collision_policy == 'overwrite':
                action = 'overwritten'
                coords_todo.extend(collided_coords)
            else:
                action = 'skipped'

        case _: raise ValueError(f'Unsupported policy: {artifacts_policy}')

    # build collision records
    collided: list[contracts.CollisionRecord] = []
    for c in sorted(collided_coords):
        meta = incumbent_catalog.get(c, {}) if incumbent_catalog else {}
        collided.append({
            'block_name': geo_utils.xy_name(c),
            'grid_coord': [c[1], c[0]],
            'incumbent_ingest_run': meta.get('ingest_run_id'),
            'incumbent_harmonize_run':  meta.get('harmonize_run_id'),
            'action_taken': action,
        })

    return _StructuralValidationResults(
        coords_todo,
        removed_count,
        on_disk_before,
        collided,
    )


def _create_missing_blocks(
    inputs: BlockBuildingInputs,
    coords_todo: list[tuple[int, int]],
    config: BlockBuildingConfig,
    context: BlockBuildingContext,
    output_dir: str
) -> None:
    '''Build all missing coordinates in parallel.'''
    creation_jobs = []
    for c in coords_todo:

        name = geo_utils.xy_name(c)
        block_inputs = io.RasterReadInput(
            image_fpath=inputs.image_fpath,
            image_window=context.image[c],
            image_band_map=context.image_band_map,
            image_dem_pad_px=config.dem_pad_px,
            label_fpath=inputs.label_fpath,
            label_window=context.label[c] if inputs.has_label else None,
            label_specs=context.label_specs
        )

        build_config = builder.DataBlockConfig(
            image_dem_pad_px=config.dem_pad_px,
            label_ignore_index=config.ignore_index,
            add_spectral=config.add_spectral,
            add_topo=config.add_topo
        )

        job = (
            _build_single_block,
            (name, block_inputs, build_config),
            {'save_fpath': os.path.join(output_dir, f'{name}.npz')}
        )
        creation_jobs.append(job)

    if creation_jobs:
        utils.ParallelExecutor().run(creation_jobs, desc=' - Creating blocks')


def _build_single_block(
    name: str,
    raster_inputs: io.RasterReadInput,
    db_config: builder.DataBlockConfig,
    *,
    save_fpath: str | None = None,
) -> geo_core.DataBlock:
    '''
    Create a DataBlock from input rasters for the window context.

    Args:
        name:
            Unique identifier for the block.
        inputs:
            Raster inputs and metadata required to construct the block.
        config:
            Block building configurations.
        save_fpath:
            Optional output path where the block will be saved.

    Returns:
        geo_core.DataBlock:
            A populated and validated block instance.
    '''
    read_outputs = io.read_block_raster_data(raster_inputs)

    db_inputs = builder.DataBlockInputs(
        block_name=name,
        image_array=read_outputs.image_array,
        image_padded_dem=read_outputs.image_padded_dem,
        label_array=read_outputs.label_array,
    )

    db_context = builder.DataBlockContext(
        image_band_map=raster_inputs.image_band_map,
        image_nodata=read_outputs.image_nodata,
        label_specs=raster_inputs.label_specs,
        label_nodata=read_outputs.label_nodata,
    )

    block = builder.build_data_block(db_inputs, db_config, db_context)

    if save_fpath:
        block.save(save_fpath)
    return block
