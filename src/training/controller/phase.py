'''
Training phase
'''

# standard imports
import dataclasses
import typing
# third-party imports
import omegaconf

@dataclasses.dataclass
class Phase:
    '''doc'''
    name: str
    num_epochs: int
    active_heads: list[str]
    frozen_heads: list[str] | None = None
    excluded_cls: dict[str, tuple[int, ...]] | None = None
    lr_scale: float = 1.0

    def __str__(self) -> str:
        indent: int=2
        s = ' ' * indent
        return f'\n{s}'.join([
            f'- Phase Name:\t{self.name}',
            f'- Max Epochs:\t{self.num_epochs}',
            f'- Active Heads:\t{self.active_heads}',
            f'- Frozen Heads:\t{self.frozen_heads}',
            f'- Excld. Class:\t{self.excluded_cls}',
            f'- LR Scale:\t{self.lr_scale}'
        ])

def generate_phases(config: omegaconf.DictConfig) -> list[Phase]:
    '''doc'''

    phases: list[Phase] = []
    cfg_phases = typing.cast(
        typ=dict[str, typing.Any],
        val=omegaconf.OmegaConf.to_container(config.phases, resolve=True)
    )
    # iterate through phases in config (1-based)
    for p in cfg_phases.values():
        phases.append(
            Phase(
                name=p['name'],
                num_epochs=p['num_epochs'],
                active_heads=p['active_heads'],
                frozen_heads=p['frozen_heads'],
                excluded_cls=p['masked_classes'],
                lr_scale=p['lr_scale']
            )
        )

    # return
    return phases
