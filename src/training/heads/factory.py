'''Trainer facing. Generate headspecs from data and config.'''

# third-party imports
import numpy
# local imports
import alias
import training.common
import training.heads
import utils

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

    def __init__(self, specs: dict[str, training.heads.Spec]):
        self._specs = specs

    def __getitem__(self, key: str) -> training.common.SpecLike:
        return self._specs[key]

    def __len__(self) -> int:
        return len(self._specs)

    def as_dict(self) -> dict[str, training.common.SpecLike]:
        '''Return a shallow copy of the mapping as `dict[str, Spec]`.'''
        return dict(self._specs)

# Public API
def build_headspecs(
        data: training.common.DataSpecsLike,
        config: alias.ConfigType
    ) -> HeadSpecs:
    '''Generate concreate head specs classes indexed by head name.'''

    # config accesor
    cfg = utils.ConfigAccess(config)

    # register alpha compute function
    alpha_fn = cfg.get_option('alpha_fn')
    alpha_fn_registry = {
        'effective_n': _count_to_effective_num,
        'inverse': _count_to_inv_weights
    }
    if alpha_fn not in alpha_fn_registry:
        raise ValueError(f'Invalid alpha calc fn: {alpha_fn}')

    # get Effective Number weights beta
    en_beta = cfg.get_option('en_beta')
    fn_kwargs = {'b': en_beta}

    # iterate heads in data and create headspec for each
    headspecs_dict: dict[str, training.heads.Spec] = {}
    for name, counts in data.heads.class_counts.items():
        headspec =  training.heads.Spec(
            name=name,
            count=counts,
            loss_alpha=alpha_fn_registry[alpha_fn](list(counts), **fn_kwargs),
            parent_head=data.heads.topology[name]['parent'],
            parent_cls=data.heads.topology[name]['parent_cls'],
            weight=1.0,
            exclude_cls=()
        )
        headspecs_dict[name] = headspec

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
    weights = (1.0 - b) / (1.0 - numpy.power(b, counts_arr))
    normalized = weights / weights.sum() * len(counts_arr)
    return normalized.tolist()
