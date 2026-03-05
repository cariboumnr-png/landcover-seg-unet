'''Multihead model config classes'''

from __future__ import annotations
# standard imports
import dataclasses
# local imports
import landseg.models.backbones as backbones

# -------------------------model general configuration-------------------------
@dataclasses.dataclass
class ModelConfig:
    '''General config'''
    body: backbones.Backbone
    in_ch: int
    base_ch: int
    logit_adjust: dict[str, list[float]]
    heads_w_counts: dict[str, list[int]]
    conditioning: CondConfig
    clamp_range: tuple[float, float]

# ----------------------model conditioning configuration----------------------
@dataclasses.dataclass
class CondConfig:
    '''Wrapper for conditioning configuration.'''

    mode: str               # mode
    domain_ids_num: int     # id categories
    domain_vec_dim: int     # vector dims
    concat: ConcatConfig    # Concat
    film: FilmConfig        # FiLM

    def __post_init__(self):
        assert self.mode in ['none', 'concat', 'film', 'hybrid']

@dataclasses.dataclass
class ConcatConfig:
    '''doc'''
    out_dim: int
    use_ids: bool
    use_vec: bool
    use_mlp: bool

@dataclasses.dataclass
class FilmConfig:
    '''doc'''
    embed_dim: int
    use_ids: bool
    use_vec: bool
    hidden: int

# -----------------------model head state configuration-----------------------
@dataclasses.dataclass
class HeadsState:
    '''Multihead status.'''
    active: list[str] | None = dataclasses.field(init=False)
    frozen: list[str] | None = dataclasses.field(init=False)
