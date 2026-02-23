'''Module entry: build head loss components for trainer.'''

# local imports
import landseg.training.common as common
import landseg.training.loss as loss

class HeadLosses:
    '''
    Typed wrapper around a mapping of heads to `CompositeLoss` objects.

    This class provides:
    - key-based access to individual `CompositeLoss` instances.
    - a stable, strongly-typed container for passing head specs through
    the codebase.

    It is *not* a full `dict` replacement. To work with the underlying
    mapping directly, use method: `as_dict()`.
    '''

    def __init__(self, specs: dict[str, loss.CompositeLoss]):
        self._specs = specs

    def __getitem__(self, key: str) -> common.CompositeLossLike:
        return self._specs[key]

    def __len__(self) -> int:
        return len(self._specs)

    def as_dict(self) -> dict[str, common.CompositeLossLike]:
        '''Return a shallow copy of the mapping as `dict[str, Spec]`.'''
        return dict(self._specs)

# Public API
def build_headlosses(
        headspecs: common.HeadSpecsLike,
        config: dict[str, dict],
        ignore_index: int
    ) -> HeadLosses:
    '''
    Generate indexed concrete `CompositeLoss` instances by given names.

    Note head names and loss config dicts need to be aligned.
    '''

    loss_dict: dict[str, loss.CompositeLoss] = {}
    # iterate through names
    per_head_alphas = {
        h.name: h.loss_alpha for h in headspecs.as_dict().values()
    }
    for name in per_head_alphas.keys():
        # if focal is used edit its alphas:
        if config.get('focal') is not None:
            config['focal']['alpha'] = per_head_alphas[name]
        # init loss compute module for each head
        loss_cls = loss.CompositeLoss(config, ignore_index)
        loss_dict[name] = loss_cls
    return HeadLosses(loss_dict)

#
sample_config = {
    'focal': {
        'weight': 0.7,
        'alpha': None,
        'gamma': 2.0,
        'reduction': 'mean',
    },
    'dice': {
        'weight': 0.3,
        'smooth': 1.0,
    }
}
sample_class = loss.CompositeLoss(sample_config, 255)
