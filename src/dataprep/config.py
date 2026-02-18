'''doc'''

# standard imports
import typing

class InputConfig(typing.TypedDict):
    '''Bimodal input rasters and config file paths.'''
    fit_input_img: str
    fit_input_lbl: str | None
    test_input_img: str | None
    input_config: str

class ArtifactConfig(typing.TypedDict):
    '''Bimodal block building related artifacts.'''
    fit_blks_dir: str
    fit_all_blks: str
    fit_valid_blks: str
    test_blks_dir: str
    test_all_blks: str

class PixelThresholdConfig(typing.TypedDict):
    '''Block validation pixel ratio thresholds.'''
    fit_blk_thres: float
    test_blk_thres: float

class ScoringConfig(typing.TypedDict):
    '''Block label representation scoring config.'''
    score_head: str
    score_alpha: float
    score_beta: float
    score_epsilon: float
    score_reward: tuple[int, ...]

class CacheConfig(
    PixelThresholdConfig,
    ScoringConfig,
    typing.TypedDict
):
    '''Cache processing configs.'''

class BlockBuilderConfigs(
    InputConfig,
    ArtifactConfig,
    typing.TypedDict
):
    '''Bimodal I/O paths for configuring a block builder.'''

class DataprepConfigs(
    InputConfig,
    ArtifactConfig,
    PixelThresholdConfig,
    ScoringConfig,
    typing.TypedDict,
):
    '''Composite config for data preparation.'''
