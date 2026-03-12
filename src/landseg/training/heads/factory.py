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

'''Trainer facing. Generate headspecs from data and config.'''

# third-party imports
import numpy
# local imports
import landseg.core as core
import landseg.training.common as common
import landseg.training.heads as heads

class HeadSpecs:
    '''
    Typed wrapper around a mapping of heads to `Spec` objects.

    This class provides:
    - key-based access to individual `Spec` instances.
    - a stable, strongly-typed container for passing head specs through
    the codebase.

    It is *not* a full `dict` replacement. To work with the underlying
    mapping directly, use method: `as_dict()`.
    '''

    def __init__(self, specs: dict[str, heads.Spec]):
        self._specs = specs

    def __getitem__(self, key: str) -> common.SpecLike:
        return self._specs[key]

    def __len__(self) -> int:
        return len(self._specs)

    def as_dict(self) -> dict[str, common.SpecLike]:
        '''Return a shallow copy of the mapping as `dict[str, Spec]`.'''
        return dict(self._specs)

# Public API
def build_headspecs(
    data: core.DataSpecs,
    alpha_fn: str,
    *,
    en_beta: float | None = None
) -> HeadSpecs:
    '''
    Generate concreate head specs classes indexed by head name.

    Args:
        data: Object matching the shape of `.dataset.DataSpecs`.
        config: `ConfigType` with loss alpha calculation parameters.

    Return:
        HeadSpecs: Per-head specification at runtime.
    '''

    # alpha compute functions
    alpha_fn_registry = {
        'effective_n': _count_to_effective_num,
        'inverse': _count_to_inv_weights
    }

    # regester alpha function and kwargs
    if alpha_fn == 'effective_n':
        assert en_beta
        fn_kwargs = {'b': en_beta}
    elif alpha_fn == 'inverse':
        fn_kwargs = {}
    else:
        raise ValueError(f'Invalid alpha calc fn: {alpha_fn}')

    # iterate heads in data and create headspec for each
    headspecs_dict: dict[str, heads.Spec] = {}
    for name, counts in data.heads.class_counts.items():
        headspec = heads.Spec(
            name=name,
            count=counts,
            loss_alpha=alpha_fn_registry[alpha_fn](list(counts), **fn_kwargs),
            parent_head=data.heads.head_parent[name],
            parent_cls=data.heads.head_parent_cls[name],
            weight=1.0,
            exclude_cls=()
        )
        headspecs_dict[name] = headspec
    # return
    return HeadSpecs(headspecs_dict)

def _count_to_inv_weights(count: list[int]) -> list[float]:
    '''Convert count to inversed weights normalized to sum of 1.'''

    inv = [1 / c  if c != 0 else 0.0 for c in count]
    inv_sum = sum(inv)
    assert inv_sum != 0
    return [float(x / inv_sum) for x in inv]

def _count_to_effective_num(
    counts: list[int],
    *,
    b: float
) -> list[float]:
    '''Convert count to EN weights Cui et al. 2019'''

    counts_arr = numpy.array(counts)
    weights = numpy.zeros_like(counts_arr)
    # assign weight only to non-zero classes
    n_zeros = counts_arr > 0
    weights[n_zeros] = (1.0 - b) / (1.0 - numpy.power(b, counts_arr[n_zeros]))
    s = weights.sum()
    if s > 0:
        weights = weights / s * len(counts_arr)
    return weights.tolist()
