'''doc'''

# standard imports
import typing

class InputConfig(typing.TypedDict):
    '''Bimodal input rasters and config file paths.'''
    fit_input_img: str
    fit_input_lbl: str
    test_input_img: str | None
    input_config: str

class OutputConfig(typing.TypedDict):
    '''Paths to artifacts generated during data preparation.'''
    fit_windows: str
    fit_blks_dir: str
    fit_all_blks: str
    fit_valid_blks: str
    fit_img_stats: str
    lbl_count_global: str
    blk_scores: str
    train_blks: str
    val_blks: str
    lbl_count_train: str
    test_windows: str
    test_blks_dir: str
    test_all_blks: str
    test_valid_blks: str
    test_img_stats: str

class PixelThresholdConfig(typing.TypedDict):
    '''Block validation pixel ratio thresholds.'''
    blk_thres_fit: float
    blk_thres_test: float

class ScoringConfig(typing.TypedDict):
    '''Block label representation scoring config.'''
    score_head: str
    score_alpha: float
    score_beta: float
    score_epsilon: float
    score_reward: tuple[int, ...]

class IOConfig(
    InputConfig,
    OutputConfig,
    typing.TypedDict
):
    '''Bimodal I/O paths for configuring a block builder.'''

class BlockBuildingConfig(
    InputConfig,
    OutputConfig,
    PixelThresholdConfig,
    typing.TypedDict
):
    '''Block building related config'''

class ProcessConfig(
    OutputConfig,
    ScoringConfig,
    typing.TypedDict
):
    '''Post block building process configs.'''

class DataprepConfigs(
    InputConfig,
    OutputConfig,
    PixelThresholdConfig,
    ScoringConfig,
    typing.TypedDict,
):
    '''Composite config for data preparation.'''
