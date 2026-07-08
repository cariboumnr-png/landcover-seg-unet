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
import landseg.geopipe.foundation.common.alias as alias
import landseg.geopipe.foundation.data_blocks.assembler as assembler
import landseg.geopipe.utils as geo_utils
import landseg.utils as utils


@dataclasses.dataclass
class BlockBuilderConfig:
    '''
    I/O paths and configuration parameters used during block construction.
    '''
    output_root: str            # path to output artifacts
    config_fpath: str           # path to input metadata (.json)
    ignore_index: int           # global ignore label index
    block_size: tuple[int, int]  # block size in row, col
    build_config: assembler.BlockBuildConfig


@dataclasses.dataclass(frozen=True)
class _PipelineWindows:
    '''
    Mapped read windows for the pipeline execution.
    '''
    image: alias.RasterWindowDict
    label: alias.RasterWindowDict


@dataclasses.dataclass(frozen=True)
class BlockBuilderResult:
    '''Results and statistics from a multi-block building execution.'''
    coords_created: list[tuple[int, int]]
    stats: dict[str, int]
    label_color_map: dict[str, list[int]] | None


def build_blocks(
    config: BlockBuilderConfig,
    image_windows: alias.RasterWindowDict,
    label_windows: alias.RasterWindowDict,
) -> BlockBuilderResult:
    '''
    Validate on-disk blocks, clear corrupt ones, and build missing.

    Args:
        config: The block builder configuration container.
        image_windows: Mapped read windows for the image raster.
        label_windows: Mapped read windows for the label raster.

    Returns:
        BlockBuilderResult: Struct holding created coords and
            execution stats.
    '''
    windows = _PipelineWindows(image=image_windows, label=label_windows)

    valid_coords, shared_count, expected_shape_count = _prepare_block_windows(
        config, windows
    )

    # Load metadata source JSON
    ctrl = artifacts.Controller.load_json_or_fail(config.config_fpath)
    ctrl.hash(overwrite=False)  # Hash once
    meta_src = ctrl.fetch()

    blks_dir = config.output_root
    os.makedirs(blks_dir, exist_ok=True)

    coords_todo, removed_count, on_disk_before = _structural_validation(
        blks_dir, valid_coords
    )

    build_cfg = dataclasses.replace(
        config.build_config,
        label_specs=meta_src.get('label_specs')
    )

    _create_missing_blocks(
        config,
        build_cfg,
        coords_todo,
        windows
    )

    stats = {
        'shared_raster_windows': shared_count,
        'expected_shape_windows': expected_shape_count,
        'blocks_on_disk_before': on_disk_before,
        'blocks_to_process': len(coords_todo),
        'damaged_blocks_removed': removed_count,
        'blocks_created': len(coords_todo)
    }

    label_color_map = meta_src.get('label_color_map')

    return BlockBuilderResult(
        coords_created=coords_todo,
        stats=stats,
        label_color_map=label_color_map
    )


# --------------------------------------------------------------------------- #
# Private Helper Functions                                                    #
# --------------------------------------------------------------------------- #

def _prepare_block_windows(
    config: BlockBuilderConfig,
    windows: _PipelineWindows
) -> tuple[set[tuple[int, int]], int, int]:
    '''Find coordinates matching block size and compute window counts.'''
    has_label = (
        bool(config.build_config.lbl_path)
        and os.path.exists(config.build_config.lbl_path)
    )
    if has_label:
        common_coords = set(windows.image.keys()) & set(windows.label.keys())
    else:
        common_coords = set(windows.image.keys())

    shared_count = len(common_coords)

    valid_coords = set(common_coords)
    for coord in common_coords:
        iw = windows.image[coord]
        lw = windows.label[coord] if has_label else None
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
    config: BlockBuilderConfig,
    build_cfg: assembler.BlockBuildConfig,
    coords_todo: list[tuple[int, int]],
    windows: _PipelineWindows
) -> None:
    '''Build all missing coordinates in parallel.'''
    ctrl = artifacts.Controller.load_json_or_fail(config.config_fpath)
    ctrl.hash(overwrite=False)  # Hash once
    meta_src = ctrl.fetch()

    base_meta = {
        'image_band_map': meta_src['image_band_map'],
        'ignore_index': config.ignore_index
    }

    has_label = (
        bool(build_cfg.lbl_path) and os.path.exists(build_cfg.lbl_path)
    )

    creation_jobs = []
    for c in coords_todo:
        name = geo_utils.xy_name(c)
        save_fpath = os.path.join(config.output_root, f'{name}.npz')

        ctx = assembler.BlockCreationContext(
            name=name,
            img_window=windows.image[c],
            lbl_window=windows.label[c] if has_label else None
        )

        save_args = {
            'save': True,
            'save_fpath': save_fpath
        }
        creation_jobs.append(
            (assembler.build_single_block, (base_meta, ctx, build_cfg), save_args)
        )

    if coords_todo:
        utils.ParallelExecutor().run(
            creation_jobs, desc='Creating datablocks'
        )
