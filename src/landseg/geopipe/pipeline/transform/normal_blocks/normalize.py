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

'''doc'''

# standard imports
import os
# third-party imports
import numpy
# local imports
import landseg.geopipe.core as core
import landseg.geopipe.pipeline.common as common
import landseg.geopipe.pipeline.common.alias as alias
import landseg.utils as utils

def normalize_blocks(
    input_blocks: set[str],
    stats: dict[str, common.ImageBandStats],
    output_dir: str,
    *,
    rebuild: bool = False
) -> list[str]:
    '''doc'''

    #
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
    if purged:
        print(f'{purged} unwanted block files removed')

    # normalize blocks
    os.makedirs(output_dir, exist_ok=True)
    jobs = [(_normalize_one_block, (b, stats, output_dir), {}) for b in work]
    if jobs:
        utils.multip.ParallelExecutor().run(jobs)

    # return current file paths
    return [f'{output_dir}/{f}' for f in os.listdir(output_dir) if f.endswith('.npz')]

def _normalize_one_block(
    block_fpath: str,
    global_stats: dict[str, common.ImageBandStats],
    target_dpath: str
):
    '''doc'''

    # read block
    data = core.DataBlock.load(block_fpath).data

    # prep dict of arrays to write
    to_write = {
        'image': _normalize_image(data.image, data.valid_mask, global_stats),
        'label': data.label_stack
    }

    # use the same file name
    filename = os.path.basename(block_fpath)
    save_fpath = os.path.join(target_dpath, filename)
    numpy.savez_compressed(save_fpath, **to_write)

def _normalize_image(
    raw_image_arr: alias.Float32Array,
    valid_mask: alias.MaskArray,
    global_stats: dict[str, common.ImageBandStats],
) -> alias.Float32Array:
    '''doc'''

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

def _purge(
    filenames_to_keep: list[str],
    target_dir: str
) -> int:
    '''
    Remove files in target_dir that are not listed in input_blocks.

    All paths in input_blocks are expected to be within target_dir.
    Return the count of removed files.
    '''

    if not os.path.exists(target_dir) or not os.listdir(target_dir):
        return 0

    removed = 0
    for name in os.listdir(target_dir):
        path = os.path.join(target_dir, name)
        if os.path.isfile(path) and name not in filenames_to_keep:
            os.remove(path)
            removed += 1
    return removed
