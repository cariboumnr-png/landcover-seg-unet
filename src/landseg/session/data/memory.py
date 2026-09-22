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
RAM-aware preload and caching strategy for dataset splits.

Infers whether each dataset split (train/val/test) should be fully
preloaded into RAM or streamed with a block-level LRU cache, based
on available system memory and estimated per-block byte size.

Public APIs:
    - `MemoryFlags`: preload and cache configuration per split.
    - `get_memory_strategy`: infer `MemoryFlags` from `DataSpecs`
      and available system memory.
'''


# standard imports
import dataclasses
# third-party imports
import psutil
# local imports
import landseg.core as core


# ----- public dataclasses
@dataclasses.dataclass
class MemoryFlags:
    '''Preload and caching configuration flags across data splits.'''
    preload_train: bool
    cache_train: int
    preload_val: bool
    cache_val: int
    preload_test: bool
    cache_test: int


# ----- public functions
def get_memory_strategy(
    data_specs: core.DataSpecs,
    available_bytes: int | None = None
) -> MemoryFlags:
    '''Infer dataset preload and caching strategy flags based on RAM.'''
    fbytes = data_specs.meta.blk_bytes
    if not fbytes: # can be set to 0 in `DataSpecs` to disable strategy
        return MemoryFlags(
            preload_train=True,
            cache_train=0,
            preload_val=True,
            cache_val=0,
            preload_test=True,
            cache_test=0
        ) # all preload

    # get dataset sizes
    train_bytes = len(data_specs.splits.train or {}) * fbytes
    val_bytes = len(data_specs.splits.val or {}) * fbytes

    # decision on preload and cache size
    mem = (
        available_bytes
        if available_bytes is not None
        else psutil.virtual_memory().available
    )
    _val = _train = False
    _val_n = train_n = 0

    # first priority: preload validation blocks into memory
    if val_bytes <= 0.6 * mem:
        _val = True
        # second priority: preload training blocks if possible
        if train_bytes <= 0.6 * (mem - val_bytes):
            _train = True
            train_n = round(0.1 * (mem - val_bytes) / fbytes)
    else:
        _val_n = round(0.3 * mem / fbytes)
        train_n = round(0.2 * mem / fbytes)

    # return flags container
    return MemoryFlags(
        preload_train=_train,
        cache_train=train_n,
        preload_val=_val,
        cache_val=_val_n,
        preload_test=True,
        cache_test=0
    )
