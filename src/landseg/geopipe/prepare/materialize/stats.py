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
Image and label statistics aggregation utilities for data blocks.

Computes global per-band means and standard deviations using Welford's
online algorithm and aggregates per-head label class counts across
selected block subsets.

Public APIs:
    - `aggregate_image_stats`: aggregate per-band image statistics.
    - `count_label`: aggregate label class counts across block subsets.
'''

# standard imports
import math
import pathlib
# local imports
import landseg.geopipe.contracts.preparation as contracts
import landseg.geopipe.core as geo_core
import landseg.utils.validator as val


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
            coord = geo_core.name_xy(id_str)
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
    input_blocks: set[pathlib.Path],
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
    available_channels = sample.image.shape[0]

    # validate and resolve channel indices
    target_indices = _resolve_channel_indices(
        channel_indices, available_channels
    )

    # define a return dict
    stats_dict: dict[str, contracts.ImageBandStats] = {
        f'band_{idx}': {
            'total_count': 0,
            'current_mean': 0.0,
            'accum_m2': 0.0,
            'std': 0.0,
        }
        for idx in range(len(target_indices))
    }

    # iterate through provided block files
    for fpath in input_blocks:
        block = geo_core.DataBlock.load(fpath)
        if 'image_stats' not in block.manifest:
            raise ValueError(
                f'Block "{fpath}" is missing "image_stats" in manifest'
            )
        manifest_stats = val.require_mapping(block.manifest['image_stats'])
        for new_idx, orig_idx in enumerate(target_indices):
            orig_key = f'band_{orig_idx}'
            if orig_key not in manifest_stats:
                raise ValueError(
                    f'Missing channel statistics for "{orig_key}" in '
                    f'block "{fpath}"'
                )
            new_key = f'band_{new_idx}'
            stats_dict[new_key] = _welfords_online(
                manifest_stats[orig_key], stats_dict[new_key]
            )

    # compute sample standard deviation
    for v in stats_dict.values():
        if v['total_count'] > 1:
            v['std'] = math.sqrt(
                max(0.0, v['accum_m2']) / (v['total_count'] - 1)
            )
        else:
            v['std'] = 1.0

    return stats_dict


# ----- private helpers
def _resolve_channel_indices(
    channel_indices: list[int] | None,
    available_channels: int,
) -> list[int]:
    '''Validate and resolve selected channel indices.'''
    if channel_indices is None:
        return list(range(available_channels))
    indices = list(val.require_sequence(channel_indices))
    if not indices:
        raise ValueError('channel_indices cannot be empty when provided')
    for idx in indices:
        val.require_numeric_range(idx, (0, available_channels - 1), int)
    return indices


def _validate_summary(input_stats: dict[str, int | float]) -> None:
    '''Validate per-block summary statistics before merging.'''
    stats_map = val.require_mapping(input_stats)
    for key in ('count', 'mean', 'm2'):
        if key not in stats_map:
            raise ValueError(f'Missing required stats key: "{key}"')
    val.require_numeric_min(stats_map['count'], 0, int)
    val.require_finite(stats_map['mean'])
    val.require_numeric_min(stats_map['m2'], 0.0, finite=True)


def _welfords_online(
    input_stats: dict[str, int | float],
    current_results: contracts.ImageBandStats,
) -> contracts.ImageBandStats:
    '''Combine per-block stats using Welford's online algorithm.'''
    _validate_summary(input_stats)
    nt = val.require_numeric_min(current_results['total_count'], 0, int)
    mt = val.require_finite(current_results['current_mean'])
    dt = val.require_numeric_min(current_results['accum_m2'], 0.0, finite=True)
    nb = int(input_stats['count'])
    if nb == 0:
        return current_results
    mb = float(input_stats['mean'])
    m2b = float(input_stats['m2'])
    delta = mb - mt
    mt += delta * nb / (nt + nb)
    dt += m2b + (delta ** 2) * nt * nb / (nt + nb)
    nt += nb
    current_results['total_count'] = nt
    current_results['current_mean'] = mt
    current_results['accum_m2'] = dt
    return current_results
