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
    - BlockBuildingInputs: Source raster paths and window mappings.
    - `BlockBuildingConfig`: Dataclass for block building configurations.
    - `BlockBuildingOutput`: Dataclass of block building results and stats.
    - `build_blocks`: Validate on-disk blocks and create missing blocks.
'''

# standard imports
import dataclasses
import pathlib
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.ingest.blocks.assembler.builder as builder
import landseg.geopipe.ingest.blocks.assembler.io as io
import landseg.geopipe.rasters as rasters
import landseg.utils as utils


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class BlockBuildingInputs:
    '''I/O paths used during block construction.'''
    image_fpath: pathlib.Path
    image_windows: geo_core.RasterWindowDict
    label_fpath: pathlib.Path | None
    label_windows: geo_core.RasterWindowDict | None
    expected_block_size: tuple[int, int] # not a config
    image_band_map: dict[str, int] | None = None
    label_specs: dict[str, geo_core.CategoricalSpec] | None = None


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
class BlockBuildingOutput:
    '''Results and statistics from a multi-block building execution.'''
    coords_created: list[tuple[int, int]]
    collided_blocks: list[contracts.CollisionRecord]
    label_color_map: dict[str, tuple[int, int, int]] | None
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
def build_blocks(
    inputs: BlockBuildingInputs,
    config: BlockBuildingConfig,
    *,
    output_dir: pathlib.Path,
    incumbent_catalog: geo_core.DataCatalog | None = None,
) -> BlockBuildingOutput:
    '''
    Validate on-disk blocks, clear corrupt ones, and build missing.

    Args:
        inputs:
            Source raster paths, window mappings, and expected block size.
        config:
            Block building configuration container.
        output_dir:
            Target directory where `.npz` blocks will be saved.
        incumbent_catalog:
            Optional incumbent catalog to resolve block collisions.

    Returns:
        BlockBuildingOutput:
            Execution output containing created coordinates and stats.
    '''
    output_dir.mkdir(parents=True, exist_ok=True)

    # prepare raster read windows
    coords_to_check = _prep_block_windows(inputs, output_dir)

    # inspect existing blocks
    validation_results = _structural_validation(
        coords_to_check,
        incumbent_catalog,
        artifacts_policy=config.artifacts_policy,
        collision_policy=config.collision_policy,
    )

    # immutable context from input rasters
    raster_read_context = io.RasterReadContext.from_rasters(
        inputs.image_fpath,
        inputs.label_fpath,
        image_band_map=inputs.image_band_map,
        label_specs=inputs.label_specs,
    )
    raster_read_context.validate_feature_specs(
        config.add_spectral,
        config.add_topo
    )

    # build blocks if missing
    build_jobs = []
    for c in validation_results.coords_todo:

        name = geo_core.xy_name(c)
        img_window = inputs.image_windows[c]
        lbl_window = inputs.label_windows[c] if inputs.label_windows else None
        job = (
            _build_a_block,
            (name, img_window, lbl_window, raster_read_context),
            {
                'ignore_index': config.ignore_index,
                'add_spectral': config.add_spectral,
                'add_topo': config.add_topo,
                'dem_pad_px': config.dem_pad_px,
                'save_fpath': pathlib.Path(output_dir) / f'{name}.npz',
            }
        )
        build_jobs.append(job)

    if build_jobs:
        utils.ParallelExecutor().run(build_jobs, desc=' - Creating blocks')

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
        label_color_map=raster_read_context.label_color_map, # pass through
        running_stats=running_stats,
    )


# ----- private helpers
def _prep_block_windows(
    inputs: BlockBuildingInputs,
    output_dir: pathlib.Path,
) -> dict[tuple[int, int], pathlib.Path]:
    '''Find coordinates matching block size and compute window counts.'''
    if inputs.label_windows:
        common_coords = (
            set(inputs.image_windows.keys()) &
            set(inputs.label_windows.keys())
        )
    else:
        common_coords = set(inputs.image_windows.keys())

    valid_coords = set(common_coords) # copy
    for coord in common_coords:
        iw = inputs.image_windows[coord]
        lw = inputs.label_windows[coord] if inputs.label_windows else None
        if (iw.height, iw.width) != inputs.expected_block_size or (
            lw is not None
            and (lw.height, lw.width) != inputs.expected_block_size
        ):
            valid_coords.remove(coord)

    return {c: output_dir / f'{geo_core.xy_name(c)}.npz' for c in valid_coords}


def _structural_validation(
    input_coords: dict[tuple[int, int], pathlib.Path],
    incumbent_catalog: geo_core.DataCatalog | None = None,
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
                pathlib.Path(input_coords[c]).unlink()
                removed_count += 1
            except FileNotFoundError:
                pass
            coords_todo.append(c)

    # evaluate collisions against collision policy
    if collided_coords and collision_policy == 'error':
        raise artifacts.ArtifactError(
            f'Intra-pool block collision detected for '
            f'{len(collided_coords)} blocks under collision policy '
            f'"error": {[geo_core.xy_name(c) for c in collided_coords[:5]]}'
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
            'block_name': geo_core.xy_name(c),
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


def _build_a_block( # pylint: disable=R0913
    name: str,
    image_window: rasters.Window,
    label_window: rasters.Window | None,
    context: io.RasterReadContext,
    *,
    ignore_index: int = 255,
    add_spectral: list[str] | None = None,
    add_topo: list[str] | None = None,
    dem_pad_px: int = 8,
    save_fpath: pathlib.Path | None = None,
) -> geo_core.DataBlock:
    '''
    Create a DataBlock from input rasters for the window context.

    Args:
        name:
            Unique identifier for the block.
        image_window:
            Window bounding box for the image raster.
        label_window:
            Window bounding box for the label raster, if present.
        context:
            Raster read context containing paths and metadata.
        ignore_index:
            Target ignore index for categorical labels.
        add_spectral:
            Optional sequence of spectral indices to compute.
        add_topo:
            Optional sequence of topographic metrics to compute.
        dem_pad_px:
            Padding in pixels for DEM neighborhood reflection.
        save_fpath:
            Optional output path where the block will be saved.

    Returns:
        geo_core.DataBlock:
            A populated and validated block instance.
    '''
    raster_data = io.read_block_raster_data(
        image_window,
        label_window,
        context,
        dem_pad=dem_pad_px
    )

    block = builder.build_data_block(
        name,
        raster_data,
        ignore_index=ignore_index,
        add_spectral=add_spectral,
        add_topo=add_topo,
    )

    if save_fpath:
        block.save(save_fpath)

    return block
