'''Multihead model config classes'''

from __future__ import annotations
# standard imports
import dataclasses

# -------------------------model general configuration-------------------------
@dataclasses.dataclass
class ModelConfig:
    '''General config'''
    in_ch: int
    base_ch: int
    heads_w_counts: dict[str, list[int]]
    logit_adjust: dict[str, list[float]]
    clamp_range: tuple[float, float]
    conditioning: CondConfig

# ----------------------model conditioning configuration----------------------
@dataclasses.dataclass
class CondConfig:
    '''Wrapper for conditioning configuration.'''
    # mode
    mode: str
    # id categories and vector dims for adapter embeddings
    domain_ids_num: int
    domain_vec_dim: int
    # Concat
    concat: ConcatConfig
    # FiLM
    film: FilmConfig

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
