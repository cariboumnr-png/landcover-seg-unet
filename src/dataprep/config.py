'''doc'''

# standard imports
import typing

class InputConfig(typing.TypedDict):
    '''Bimodal input rasters and config file paths.'''
    train_img: str
    train_lbl: str | None
    infer_img: str | None
    input_config: str

class ArtifactConfig(typing.TypedDict):
    '''Bimodal block building related artifacts.'''
    train_blks_dir: str
    train_all_blks: str
    train_valid_blks: str
    infer_blks_dir: str
    infer_all_blks: str

class RuntimeConfig(typing.TypedDict):
    '''doc'''
    train_px_thres: float
    infer_px_thres: float

class BlockBuilderConfigs(
    InputConfig,
    ArtifactConfig,
    typing.TypedDict
):
    '''Bimodal I/O paths for configuring a block builder.'''

class DataprepConfigs(
    InputConfig,
    ArtifactConfig,
    RuntimeConfig,
    typing.TypedDict,
):
    '''Composite config for data preparation.'''
