'''
Training phase
'''

# standard imports
from __future__ import annotations
import dataclasses
# local imports
import landseg.alias as alias
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class Phase:
    '''doc'''
    name: str
    num_epochs: int
    heads: _HeadsConifg
    la_scheme: _LogitAdjustScheme
    lr_scale: float = 1.0
    finished: bool = False

    def __str__(self) -> str:
        return '\n'.join([
            f'- Phase Name:\t{self.name}',
            f'- Max Epochs:\t{self.num_epochs}',
            str(self.heads),
            str(self.la_scheme),
            f'- LR Scale:\t{self.lr_scale}'
        ])

@dataclasses.dataclass
class _HeadsConifg:
    '''Phase level heads-related config.'''
    active_heads: list[str]
    frozen_heads: list[str] | None
    excluded_cls: dict[str, tuple[int, ...]] | None

    def __str__(self) -> str:
        indent: int=2
        s = ' ' * indent
        return f'\n{s}'.join([
            '- Heads Specs',
            f'- Active Heads:\t{self.active_heads}',
            f'- Frozen Heads:\t{self.frozen_heads}',
            f'- Excld. Class:\t{self.excluded_cls}',
        ])

@dataclasses.dataclass
class _LogitAdjustScheme:
    '''Logit adjustment scheme.'''
    logit_adjust_alpha: float
    enable_train_logit_adjustment: bool
    enable_val_logit_adjustment: bool
    enable_test_logit_adjustment: bool

    def __str__(self) -> str:
        indent: int=2
        s = ' ' * indent
        return f'\n{s}'.join([
            '- Logit Adjustment',
            f'- Global Alpha:\t{self.logit_adjust_alpha:.2f}',
            f'- Training Stage:\t{self.enable_train_logit_adjustment}',
            f'- Validation Stage:\t{self.enable_val_logit_adjustment}',
            f'- Inference Stage:\t{self.enable_test_logit_adjustment}',
        ])

# -------------------------------Public Function-------------------------------
def generate_phases(phase_config: alias.ConfigType) -> list[Phase]:
    '''doc'''

    # config accesor
    phase_cfg = utils.ConfigAccess(phase_config)
    phases: list[Phase] = []
    # iterate through phases in config (1-based)
    for p in phase_cfg.get_option('phases'):
        cfg = utils.ConfigAccess(p)
        phases.append(
            Phase(
                name=cfg.get_option('name'),
                num_epochs=cfg.get_option('num_epochs'),
                heads=_HeadsConifg(
                    cfg.get_option('heads', 'active_heads'),
                    cfg.get_option('heads', 'frozen_heads', default=None),
                    cfg.get_option('heads', 'masked_classes', default=None)
                ),
                la_scheme=_LogitAdjustScheme(
                    cfg.get_option('logit_adjust', 'alpha', default=1.0),
                    cfg.get_option('logit_adjust', 'train', default=False),
                    cfg.get_option('logit_adjust', 'val', default=False),
                    cfg.get_option('logit_adjust', 'test', default=False)
                ),
                lr_scale=cfg.get_option('lr_scale')
            )
        )

    # return
    return phases
