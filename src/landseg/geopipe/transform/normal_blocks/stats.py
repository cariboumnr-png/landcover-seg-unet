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
Image statistics aggregation utilities for cached data blocks. Computes
global per-band means and standard deviations using Welford's online
algorithm and validates per-block stats when needed.

Public APIs:
    - get_image_stats: Aggregate global image statistics across blocks,
      validating and (if needed) repairing per-block stats before use.
'''

# standard imports
import math
# local imports
import landseg.geopipe.core as core

# -------------------------------Public Function-------------------------------
def aggregate_image_stats(
    input_blocks: set[str]
) -> dict[str, core.ImageBandStats]:
    '''
    Aggregate per-band image statistics across the input blocks.

    Args:
        input_blocks: List of file paths to block artifacts to scan.
        stats_fpath: Output JSON path for the aggregated statistics.
        logger: Logger for progress messages and diagnostics.
        recompute: If True, force recompute stats from blocks.

    Returns:
        dict: A mapping of band keys to statistics, including
            "total_count", "current_mean", "accum_m2", and "std".
    '''

    # get image channel count from the first block
    sample = core.DataBlock.load(next(iter(input_blocks))).data
    num_bands = sample.image.shape[0]

    # define a return dict
    stats_dict: dict[str, core.ImageBandStats] = {
        f'band_{_}': {
            'total_count': 0,
            'current_mean': 0.0,
            'accum_m2': 0.0,
            'std': 0.0
        } for _ in range(0, num_bands)
    }

    # iterate through provided block files
    for fpath in input_blocks:
        # prep
        stats = core.DataBlock.load(fpath).meta['image_stats']
        # return dict and stats dict have the same keys
        for key, value_dict in stats.items():
            stats_dict[key] = _welfords_online(value_dict, stats_dict[key])

    # deviation to std
    for v in stats_dict.values():
        v['std'] = math.sqrt((v['accum_m2'] / (v['total_count']-1)))

    # return
    return stats_dict

def _welfords_online(
    input_stats: dict[str, int | float],
    current_results: core.ImageBandStats
) -> core.ImageBandStats:
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
