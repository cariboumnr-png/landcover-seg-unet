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
Image and label statistics aggregation utilities for data blocks.

Computes global per-band means and standard deviations using Welford's
online algorithm and aggregates per-head label class counts across
selected block subsets.

Public APIs:
    - aggregate_image_stats: aggregate global per-band image statistics.
    - count_label: aggregate label class counts across block subsets.
'''

# standard imports
import math
# local imports
import landseg.geopipe.contracts.preparation as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.utils as geo_utils


# ----- public functions
def count_label(
    raw_class_counts: dict[tuple[int, int], dict[str, list[int]]],
    selected_block_id: list[str],
) -> dict[str, list[int]]:
    '''
    Aggregate label class counts across a list of block files.

    Args:
        raw_class_counts:
            mapping of block coordinate to per-head class count lists.
        selected_block_id:
            list of block filename strings to aggregate.

    Returns:
        dict[str, list[int]]:
            aggregated per-head class pixel counts.
    '''
    # parse selected block id
    parsed_id: set[tuple[int, int]] = set()
    for id_str in selected_block_id:
        try:
            coord = geo_utils.name_xy(id_str)
            parsed_id.add(coord)
        except (IndexError, ValueError) as e:
            raise ValueError(f'Invalid block coord string: {id_str}') from e

    # iterate current training blocks to get label class counts
    lbl_stats: dict[str, list[int]] = {}
    for c, blk_counts in raw_class_counts.items():
        if c not in parsed_id:
            continue
        for head, cls_counts in blk_counts.items():
            if head in lbl_stats:
                lbl_stats[head] = [
                    a + b for a, b in zip(lbl_stats[head], cls_counts)
                ]
            else:
                lbl_stats[head] = [int(x) for x in cls_counts]
    return lbl_stats


def aggregate_image_stats(
    input_blocks: set[str],
    channel_indices: list[int] | None = None,
) -> dict[str, contracts.ImageBandStats]:
    '''
    Aggregate per-band image statistics across the input blocks.

    Args:
        input_blocks:
            set of file paths to block artifacts to scan.
        channel_indices:
            optional list of 0-based channel indices to select for
            aggregation.

    Returns:
        dict[str, ImageBandStats]:
            mapping of band keys to aggregated count, mean, and std.
    '''
    if not input_blocks:
        raise ValueError(
            'input_blocks cannot be empty for image stats aggregation'
        )

    # get image channel count from the first block
    sample = geo_core.DataBlock.load(next(iter(input_blocks))).data
    num_bands = (
        len(channel_indices)
        if channel_indices is not None
        else sample.image.shape[0]
    )

    # define a return dict
    stats_dict: dict[str, contracts.ImageBandStats] = {
        f'band_{_}': {
            'total_count': 0,
            'current_mean': 0.0,
            'accum_m2': 0.0,
            'std': 0.0
        } for _ in range(0, num_bands)
    }

    # iterate through provided block files
    for fpath in input_blocks:
        manifest_stats = geo_core.DataBlock.load(fpath).manifest['image_stats']
        if channel_indices is not None:
            for new_idx, orig_idx in enumerate(channel_indices):
                orig_key = f'band_{orig_idx}'
                if orig_key in manifest_stats:
                    stats_dict[f'band_{new_idx}'] = _welfords_online(
                        manifest_stats[orig_key], stats_dict[f'band_{new_idx}']
                    )
        else:
            for key, value_dict in manifest_stats.items():
                if key in stats_dict:
                    stats_dict[key] = _welfords_online(
                        value_dict, stats_dict[key]
                    )

    # deviation to std
    for v in stats_dict.values():
        if v['total_count'] > 1:
            v['std'] = math.sqrt((v['accum_m2'] / (v['total_count'] - 1)))
        else:
            v['std'] = 1.0

    # return
    return stats_dict


# ----- private helpers
def _welfords_online(
    input_stats: dict[str, int | float],
    current_results: contracts.ImageBandStats
) -> contracts.ImageBandStats:
    '''Combine per-block stats using Welford's online algorithm.'''
    # block stats from stats dict
    # NOTE: see key and value type conventions at geopipe.core.block
    nb = input_stats['count']
    mb = input_stats['mean']
    m2b = input_stats['m2']
    # type guards
    assert isinstance(nb, int)
    assert isinstance(mb, float)
    assert isinstance(m2b, float)
    # current stats from the processed blocks
    nt = current_results['total_count']
    mt = current_results['current_mean']
    dt = current_results['accum_m2']
    # Welford's online algorithm
    delta = mb - mt
    mt += delta * nb / (nt + nb)
    dt += m2b + (delta ** 2) * nt * nb / (nt + nb)
    nt += nb
    # assign back and return
    current_results['total_count'] = nt
    current_results['current_mean'] = mt
    current_results['accum_m2'] = dt
    return current_results
