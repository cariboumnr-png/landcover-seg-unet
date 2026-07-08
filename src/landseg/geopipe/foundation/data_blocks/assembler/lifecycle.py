# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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
    - BlockBuilderConfig: Config parameters for block builder setup.
    - BlockBuilderResult: Dataclass result wrapping builder execution outputs.
    - build_blocks: Validates and creates data blocks in parallel.
'''

# standard imports
import dataclasses
import os
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.common.alias as alias
import landseg.geopipe.foundation.data_blocks.assembler as assembler
import landseg.geopipe.utils as geo_utils
import landseg.utils as utils


@dataclasses.dataclass(frozen=True)
class BlockBuildingInput:
    '''I/O paths used during block construction.'''
    output_root: str            # path to output artifacts
    image_fpath: str            # path to input image data (.tiff)
    label_fpath: str | None     # path to input label data (.tiff)
    config_fpath: str           # path to input metadata (.json)

    @property
    def has_label(self) -> bool:
        '''Return `True` is label raster is provided.'''
        return bool(self.label_fpath) and os.path.exists(self.label_fpath)


@dataclasses.dataclass(frozen=True)
class BlockBuildingContext:
    '''Mapped read windows for the pipeline execution.'''
    image: alias.RasterWindowDict
    label: alias.RasterWindowDict


@dataclasses.dataclass(frozen=True)
class BlockBuildingConfig:
    '''Container for block building configurations'''
    ignore_index: int               # global ignore label index
    dem_pad_px: int                 # image DEM channel padding in pixels
    block_size: tuple[int, int]     # block size in row, col
    image_band_map: dict[str, int]
    label_specs: dict[str, geo_core.LabelSpecs]
    add_spectral: list[str] | None = None
    add_topo: bool = False


@dataclasses.dataclass(frozen=True)
class BlockBuildingOutput:
    '''Results and statistics from a multi-block building execution.'''
    coords_created: list[tuple[int, int]]
    stats: dict[str, int]
    label_color_map: dict[str, list[int]] | None


def build_blocks(
    inputs: BlockBuildingInput,
    context: BlockBuildingContext,
    config: BlockBuildingConfig,
) -> BlockBuildingOutput:
    '''
    Validate on-disk blocks, clear corrupt ones, and build missing.

    Args:
        inputs:
        context:
        config: The block builder configuration container.

    Returns:
        BlockBuilderResult: Struct holding created coords and
            execution stats.
    '''

    # load dataset configuration JSON
    ctrl = artifacts.Controller[dict].load_json_or_fail(inputs.config_fpath)
    ctrl.hash(overwrite=False)  # Hash once
    dataset_config = ctrl.fetch()
    assert dataset_config # typing only

    blks_dir = inputs.output_root
    os.makedirs(blks_dir, exist_ok=True)

    # prepare raster read windows
    valid_coords, shared_count, expected_shape_count = _prepare_block_windows(
        inputs, context, config
    )

    # inspect existing blocks
    coords_todo, removed_count, on_disk_before = _structural_validation(
        blks_dir, valid_coords
    )

    # create blocks if missing
    _create_missing_blocks(inputs, coords_todo, context, config)

    # simple runtime stats
    stats = {
        'shared_raster_windows': shared_count,
        'expected_shape_windows': expected_shape_count,
        'blocks_on_disk_before': on_disk_before,
        'blocks_to_process': len(coords_todo),
        'damaged_blocks_removed': removed_count,
        'blocks_created': len(coords_todo)
    }

    return BlockBuildingOutput(
        coords_created=coords_todo,
        stats=stats,
        label_color_map=dataset_config.get('label_color_map') # pass-through
    )


# --------------------------------------------------------------------------- #
# Private Helper Functions                                                    #
# --------------------------------------------------------------------------- #

def _prepare_block_windows(
    inputs: BlockBuildingInput,
    context: BlockBuildingContext,
    config: BlockBuildingConfig
) -> tuple[set[tuple[int, int]], int, int]:
    '''Find coordinates matching block size and compute window counts.'''
    if inputs.has_label:
        common_coords = set(context.image.keys()) & set(context.label.keys())
    else:
        common_coords = set(context.image.keys())

    shared_count = len(common_coords)

    valid_coords = set(common_coords)
    for coord in common_coords:
        iw = context.image[coord]
        lw = context.label[coord] if inputs.has_label else None
        if (iw.height, iw.width) != config.block_size or (
            lw is not None and (lw.height, lw.width) != config.block_size
        ):
            valid_coords.remove(coord)

    expected_shape_count = len(valid_coords)
    return valid_coords, shared_count, expected_shape_count


def _structural_validation(
    blks_dir: str,
    valid_coords: set[tuple[int, int]]
) -> tuple[list[tuple[int, int]], int, int]:
    '''Verify existing block file integrity; remove damaged ones.'''
    blks_to_check = {
        c: os.path.join(blks_dir, f'{geo_utils.xy_name(c)}.npz')
        for c in valid_coords
    }

    jobs = [
        (assembler.check_npz_integrity, (c, fp), {})
        for c, fp in blks_to_check.items()
    ]

    results = utils.ParallelExecutor().run(jobs, ' - Checking datablocks')
    parsed = {k: v for rr in results for k, v in rr.items()}

    coords_todo = []
    removed_count = 0
    on_disk_before = 0

    for c, valid in parsed.items():
        if not valid:
            coords_todo.append(c)
            try:
                os.remove(blks_to_check[c])
                removed_count += 1
            except FileNotFoundError:
                pass
        else:
            on_disk_before += 1

    return coords_todo, removed_count, on_disk_before


def _create_missing_blocks(
    inputs: BlockBuildingInput,
    coords_todo: list[tuple[int, int]],
    windows: BlockBuildingContext,
    config: BlockBuildingConfig
) -> None:
    '''Build all missing coordinates in parallel.'''
    creation_jobs = []
    for c in coords_todo:
        # positionals
        name = geo_utils.xy_name(c)
        block_inputs = assembler.RasterReadInput(
            image_fpath=inputs.image_fpath,
            image_window=windows.image[c],
            image_band_map=config.image_band_map,
            image_dem_pad_px=config.dem_pad_px,
            label_fpath=inputs.label_fpath,
            label_window=windows.label[c] if inputs.has_label else None,
            label_specs=config.label_specs
        )
        # keyword args
        kwargs = {
            'ignore_index': True,
            'add_spectral': config.add_spectral,
            'add_topo': config.add_topo,
            'save_fpath': os.path.join(inputs.output_root, f'{name}.npz'),
        }

        # add job
        job = (assembler.build_single_block, (name, block_inputs), kwargs,)
        creation_jobs.append(job)

    if creation_jobs:
        utils.ParallelExecutor().run(creation_jobs, desc='Creating datablocks')
