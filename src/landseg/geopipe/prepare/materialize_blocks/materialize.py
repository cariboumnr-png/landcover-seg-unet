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
Block materialization and normalization utilities.

Applies global image normalization to raw data blocks using statistics
computed from training data. Produces normalized block artifacts with
reclassified multi-head label stacks and maintains file mappings.

Public APIs:
    - materialize_blocks: normalize blocks and write artifacts to disk.
'''

# standard imports
import os
import typing
# third-party imports
import numpy
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.prepare.common.alias as alias
import landseg.geopipe.prepare.data_context as data_context
import landseg.utils as utils


# ----- public functions
def materialize_blocks(
    input_blocks: set[str],
    stats: dict[str, geo_core.ImageBandStats],
    context: data_context.DatasetContext,
    output_dir: str,
    *,
    rebuild: bool = False,
) -> tuple[dict[str, str], int]:
    '''
    Normalize a collection of raw data blocks using global image stats.

    Computes which blocks need processing, purges stale artifacts from
    target directory, applies per-band normalization, builds multi-head
    label stacks, and saves compressed numpy archives.

    Args:
        input_blocks:
            set of file paths to raw block artifacts.
        stats:
            per-band global image statistics from training data.
        context:
            dataset context containing features and target hierarchy.
        output_dir:
            directory where normalized block files are written.
        rebuild:
            if True, reprocesses all blocks regardless of existence.

    Returns:
        tuple[dict[str, str], int]:
            mapping of block names to file paths, and count of purged
            files.
    '''
    names: list[str] = []
    work: list[str] = []
    for b in input_blocks:
        name = os.path.basename(b)
        names.append(name)
        if rebuild:
            work.append(b)
        else:
            if not os.path.exists(f'{output_dir}/{name}'):
                work.append(b)

    # purge blocks not belong
    purged = _purge(names, output_dir)

    # normalize blocks
    os.makedirs(output_dir, exist_ok=True)
    jobs = [
        (
            _materialize_one_block,
            (b, stats, output_dir, context),
            {}
        )
        for b in work
    ]
    if jobs:
        utils.ParallelExecutor(desc=' - Normalize data blocks').run(jobs)

    # return current file paths
    indexed_files: dict[str, str] = {}
    for fpath in os.listdir(output_dir):
        if fpath.endswith('.npz'):
            name, _ = os.path.splitext(os.path.basename(fpath))
            fp = os.path.abspath(f'{output_dir}/{fpath}') # use absolute fpath
            indexed_files[name] = fp
    return indexed_files, purged


# ----- private helpers
def _purge(
    filenames_to_keep: list[str],
    target_dir: str,
) -> int:
    '''Remove files in target directory not present in expected list.'''
    if not os.path.exists(target_dir) or not os.listdir(target_dir):
        return 0

    removed = 0
    for name in os.listdir(target_dir):
        path = os.path.join(target_dir, name)
        if os.path.isfile(path) and name not in filenames_to_keep:
            os.remove(path)
            removed += 1
    return removed


def _materialize_one_block(
    block_fpath: str,
    img_stats: dict[str, geo_core.ImageBandStats],
    target_dpath: str,
    context: data_context.DatasetContext,
):
    '''Normalize a single data block and write it to disk.'''
    # read block
    block = geo_core.DataBlock.load(block_fpath)
    data = block.data

    # image channel selection
    ch_idx = context.features.indices
    img_arr = _normalize_image(data.image[ch_idx], data.valid_mask, img_stats)

    # label layers reclassification
    base_head_names = list(context.targets.resolved_reclass.keys())
    reclass = {
        k: v.groups
        for k, v in context.targets.resolved_reclass.items()
        if v is not None
    }
    ignore_idx = block.manifest['label_ignore_index']
    lbl_arr = _reclassify_labels(
        data.label, base_head_names, reclass, ignore_idx
    )

    # write blocks to files
    filename = os.path.basename(block_fpath)
    save_fpath = os.path.join(target_dpath, filename)
    to_write = {'image': img_arr, 'label': lbl_arr}
    numpy.savez_compressed(save_fpath, **to_write)


def _normalize_image(
    raw_image_arr: alias.Float32Array,
    valid_mask: alias.MaskArray,
    global_stats: dict[str, geo_core.ImageBandStats],
) -> alias.Float32Array:
    '''Apply per-band normalization using global stats.'''
    # assertion
    assert raw_image_arr.ndim == 3
    assert len(global_stats) == len(raw_image_arr)
    assert valid_mask.shape == raw_image_arr.shape[-2:]

    # init data attribute, inherit dtype float32
    image_normalized = numpy.empty_like(raw_image_arr)

    # normalize each band
    for i, (band, stats) in enumerate(global_stats.items()):
        # sanity check - dict keys from band_0
        assert band.lstrip('band_') == str(i)
        # get global stats from input
        g_mean = stats['current_mean']
        g_std = stats['std'] if stats['std'] != 0 else 1
        # get image band and replace invalid pixels with global mean
        img_band = raw_image_arr[i]
        img_band = numpy.where(valid_mask, img_band, g_mean)
        # normalize band
        image_normalized[i] = (img_band - g_mean) / g_std

    # return
    return image_normalized


def _reclassify_labels(
    raw_labels: numpy.ndarray | typing.Sequence[numpy.ndarray],
    label_layer_names: typing.Sequence[str],
    target_reclass: typing.Mapping[str, dict[int, tuple[int, ...]] | None],
    ignore_index: int,
) -> numpy.ndarray:
    '''Build multi-head label stack applying active target reclasses.'''
    if isinstance(raw_labels, numpy.ndarray):
        if raw_labels.ndim == 3:
            label_list = [raw_labels[i] for i in range(raw_labels.shape[0])]
        elif raw_labels.ndim == 2:
            label_list = [raw_labels]
        else:
            raise ValueError(
                f'Expected 2D or 3D label array, got shape {raw_labels.shape}'
            )
    else:
        label_list = list(raw_labels)

    stack: list[numpy.ndarray] = []

    for i, arr in enumerate(label_list):
        name = (
            label_layer_names[i]
            if i < len(label_layer_names)
            else f'label_{i}'
        )
        reclass_cfg = target_reclass.get(name)
        if not reclass_cfg:
            stack.append(arr)
            continue

        # 1. base layer
        stack.append(arr)

        # 2. grouping layer (parent)
        group_layer = numpy.full_like(arr, ignore_index, dtype=arr.dtype)
        for group_id, classes in reclass_cfg.items():
            source_pixels = [c + 1 for c in classes]
            mask = numpy.isin(arr, source_pixels)
            group_layer[mask] = int(group_id)
        stack.append(group_layer)

        # 3. child slices (sub-groups)
        for group_id, classes in reclass_cfg.items():
            child_arr = numpy.full_like(arr, ignore_index, dtype=arr.dtype)
            for k, cls_id in enumerate(classes):
                child_arr[arr == (cls_id + 1)] = int(k)
            stack.append(child_arr)

    return numpy.stack(stack, axis=0)
