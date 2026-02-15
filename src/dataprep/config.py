'''doc'''

# standard imports
import typing

class InputConfig(typing.TypedDict):
    '''doc'''
    train_img: str
    train_lbl: str | None
    infer_img: str | None
    input_config: str

class ArtifactConfig(typing.TypedDict):
    '''doc'''
    train_blks_dir: str
    train_all_blks: str
    train_valid_blks: str
    infer_blks_dir: str
    infer_all_blks: str

class RuntimeConfig(typing.TypedDict):
    '''doc'''
    train_px_thres: float
    infer_px_thres: float

class DataprepConfigs(
    InputConfig,
    ArtifactConfig,
    RuntimeConfig,
    typing.TypedDict,
):
    '''doc'''
