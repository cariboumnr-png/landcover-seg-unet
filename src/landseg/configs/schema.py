# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
'''
This module mirrors the Hydra/YAML config tree using Python dataclasses,
suitable for OmegaConf structured configs.
'''

# standard imports
from __future__ import annotations
import dataclasses
import os
import re
import typing
# third-party imports
import omegaconf

# --------------------------------INPUT CONFIGS--------------------------------
# ----- extent
@dataclasses.dataclass
class Extent:
    filename: str = ''
    filepath: str ='${exp_root}/input/extent_reference/${inputs.extent.inputs.filename}'
    origin: tuple[float, float] = (0.0, 0.0)
    pixel_size: tuple[float, float] = (0.0, 0.0)
    grid_extent: tuple[float, float] | None = None
    grid_shape: tuple[int, int] | None = None

@dataclasses.dataclass
class InputExtentCfg:
    crs: str = omegaconf.MISSING
    mode: str = 'ref'  # 'ref' | 'aoi' | 'tiles'
    inputs: Extent = dataclasses.field(default_factory=Extent)

    def __post_init__(self):
        if not _is_resolved(self.crs) or not _is_resolved(self.mode):
            return
        if not bool(re.fullmatch(r'epsg:\d+', self.crs, re.I)):
            raise ValueError('Invalid CRS identifier. Must be [EPSG:....]')
        if self.mode not in {'ref', 'aoi', 'tiles'}:
            raise ValueError(f'Invalid mode: {self.mode}')

    def validate(self) -> None:
        if not _is_resolved(self.mode):
            return
        if self.mode == 'ref':
            if not os.path.exists(self.inputs.filepath):
                raise FileNotFoundError('Mode=ref but ref raster not provided')
        elif self.mode == 'aoi':
            if not all(self.inputs.pixel_size):
                raise ValueError('Mode=aoi|tiles but pixel size has zero(s)')
            if not all(self.inputs.grid_extent or ()):
                raise ValueError('Mode=aoi but grid extent has zero(s)')
        elif self.mode == 'tiles':
            if not all(self.inputs.pixel_size):
                raise ValueError('Mode=aoi|tiles but pixel size has zero(s)')
            if not all(self.inputs.grid_shape or ()):
                raise ValueError('Mode=tiles but grid shape has zero(s)')

# ----- domain
@dataclasses.dataclass
class DomainFile:
    filename: str = omegaconf.MISSING
    index_base: int = omegaconf.MISSING

@dataclasses.dataclass
class InputDomainCfg:
    input_dirpath: str = '${exp_root}/input/domain_knowledge'
    files: list[DomainFile] = dataclasses.field(default_factory=lambda: [])

    def validate(self):
        for file in self.files:
            fpath = os.path.join(self.input_dirpath, file.filename)
            if file.filename and not os.path.exists(fpath):
                raise FileNotFoundError(f'Invalid domain raster: {fpath}')

# ----- data
@dataclasses.dataclass
class FileNames:
    dev_image: str = omegaconf.MISSING
    dev_label: str = omegaconf.MISSING
    test_image: str | None = omegaconf.MISSING
    test_label: str | None = omegaconf.MISSING
    config: str = omegaconf.MISSING

@dataclasses.dataclass
class Dirs:
    dev: str = '${exp_root}/input/${inputs.data.name}/dev'
    test: str = '${exp_root}/input/${inputs.data.name}/test'
    config: str = '${exp_root}/input/${inputs.data.name}/config'

@dataclasses.dataclass
class FilePaths:
    dev_image: str = '${inputs.data.dirs.dev}/${inputs.data.filenames.dev_image}'
    dev_label: str = '${inputs.data.dirs.dev}/${inputs.data.filenames.dev_label}'
    test_image: str = '${inputs.data.dirs.test}/${inputs.data.filenames.test_image}'
    test_label: str = '${inputs.data.dirs.test}/${inputs.data.filenames.test_label}'
    config: str = '${inputs.data.dirs.config}/${inputs.data.filenames.config}'

@dataclasses.dataclass
class InputDataCfg:
    name: str = omegaconf.MISSING
    input_dirpath: str = '${exp_root}/input/${inputs.data.name}'
    filenames: FileNames = dataclasses.field(default_factory=FileNames)
    dirs: Dirs = dataclasses.field(default_factory=Dirs)
    filepaths: FilePaths = dataclasses.field(default_factory=FilePaths)

    def __post_init__(self):
        # defer validation until config is composed and resolved
        if not _is_resolved(self.name):
            return
        if not self.name:
            raise ValueError('Input data name not provided')

    def validate(self) -> None:
        def _must_exist(path: str | None, label: str) -> None:
            if path and not os.path.exists(path):
                raise FileNotFoundError(f'invalid {label}: {path}')
        # checks
        _must_exist(self.filepaths.dev_image, 'fit image')
        _must_exist(self.filepaths.dev_label, 'fit label')
        _must_exist(self.filepaths.test_image, 'test image')
        _must_exist(self.filepaths.test_label, 'test label')
        _must_exist(self.filepaths.config, 'config json')

# ----- INPUTS
@dataclasses.dataclass
class Inputs:
    extent: InputExtentCfg = dataclasses.field(default_factory=InputExtentCfg)
    domain: InputDomainCfg = dataclasses.field(default_factory=InputDomainCfg)
    data: InputDataCfg = dataclasses.field(default_factory=InputDataCfg)

    def validate(self) -> None:
        self.extent.validate()
        self.domain.validate()
        self.data.validate()

# ------------------------------DATAPREP  CONFIGS------------------------------
# ----- grid
@dataclasses.dataclass
class TileSize:
    row: int = 256
    col: int = 256

@dataclasses.dataclass
class TileOverlap:
    row: int = 0
    col: int = 0

@dataclasses.dataclass
class PrepGridCfg:
    id: str = (
        'grid_row_${prep.grid.tile_size.row}_${prep.grid.tile_overlap.row}_'
        'col_${prep.grid.tile_size.col}_${prep.grid.tile_overlap.col}'
    )
    output_dirpath: str = '${exp_root}/artifacts/world_grids'
    version: str = 'v1'
    tile_size: TileSize = dataclasses.field(default_factory=TileSize)
    tile_overlap: TileOverlap = dataclasses.field(default_factory=TileOverlap)

# ----- domain
@dataclasses.dataclass
class PrepDomainCfg:
    output_dirpath: str = '${exp_root}/artifacts/domain'
    as_ids: str | None = None
    as_vec: str | None = None
    valid_threshold: float = 0.7
    target_variance: float = 0.9

# ----- data
@dataclasses.dataclass
class ArtifactsDirs:
  # test blocks stats for normalization
    foundation: str = '${exp_root}/artifacts/foundation'
    transform: str = '${exp_root}/artifacts/transform'

@dataclasses.dataclass
class General:
  # test blocks stats for normalization
    ignore_index: int = 255
    image_dem_pad: int = 8

@dataclasses.dataclass
class Thresholds:
    blk_thres_fit: float = 0.75
    blk_thres_test: float = 0.1

@dataclasses.dataclass
class Partition:
    val_ratio: float = 0.1
    test_ratio: float = 0.0
    buffer_step: int = 1

@dataclasses.dataclass
class Scoring:
    reward: dict[int, float] = dataclasses.field(default_factory=dict)
    alpha: float = 0.6
    beta: float = 0.8

@dataclasses.dataclass
class Hydration:
    max_skew_rate: float = 10.0

@dataclasses.dataclass
class PrepDataCfg:
    artifacts: ArtifactsDirs = dataclasses.field(default_factory=ArtifactsDirs)
    general: General = dataclasses.field(default_factory=General)
    threshold: Thresholds = dataclasses.field(default_factory=Thresholds)
    partition: Partition = dataclasses.field(default_factory=Partition)
    scoring: Scoring = dataclasses.field(default_factory=Scoring)
    hydration: Hydration = dataclasses.field(default_factory=Hydration)

# ----- PREP
@dataclasses.dataclass
class Prep:
    grid: PrepGridCfg = dataclasses.field(default_factory=PrepGridCfg)
    domain: PrepDomainCfg = dataclasses.field(default_factory=PrepDomainCfg)
    data: PrepDataCfg = dataclasses.field(default_factory=PrepDataCfg)

# ---------------------------------MODELS CONFIGS------------------------------
@dataclasses.dataclass
class ConvParams:
    norm: str = 'gn'
    gn_groups: int = 8
    p_drop: float = 0.0

@dataclasses.dataclass
class UnetBody:
    base_ch: int = 32
    conv_params: dict[str, typing.Any] = dataclasses.field(
        default_factory=lambda: {
            'downs': dataclasses.asdict(ConvParams()),
            'ups': dataclasses.asdict(ConvParams())
        }
    )

@dataclasses.dataclass
class UnetPPBody:
    base_ch: int = 32
    conv_params: dict[str, typing.Any] = dataclasses.field(
        default_factory=lambda: {
            'downs': dataclasses.asdict(ConvParams()),
            'nodes': dataclasses.asdict(ConvParams())
        }
    )

@dataclasses.dataclass
class Concat:
    out_dim: int = 4
    use_ids: bool = True
    use_vec: bool = True
    use_mlp: bool = True

@dataclasses.dataclass
class FiLM:
    embed_dim: int = 4
    use_ids: bool = True
    use_vec: bool = True
    hidden: int = 128

@dataclasses.dataclass
class CondCfg:
    mode: str | None = None  #  'concat' | 'film' | 'hybrid'
    concat: Concat = dataclasses.field(default_factory=Concat)
    film: FiLM = dataclasses.field(default_factory=FiLM)

@dataclasses.dataclass
class ModelFlags:
    enable_logit_adjust: bool = True
    enable_clamp: bool = True

# ----- MODELS
@dataclasses.dataclass
class ModelsCfg:
    use_body: str = 'unet'
    body_registry: dict[str, typing.Any] = dataclasses.field(
        default_factory=lambda: {
            'unet': UnetBody(),
            'unetpp': UnetPPBody(),
        }
    )
    conditioning: CondCfg = dataclasses.field(default_factory=CondCfg)
    clamp_range: tuple[float, float] = (1e-4, 1e4)
    flags: ModelFlags = dataclasses.field(default_factory=ModelFlags)

    def __post_init__(self):
        mode = self.conditioning.mode
        if mode and mode not in ['hybrid', 'concat', 'film']:
            raise ValueError(f'Invalid conditionning mode: {mode}')

    def validate(self) -> None:
        # cross-check clamp range ordering
        lo, hi = self.clamp_range
        if lo <= 0 or hi <= 0 or lo >= hi:
            raise ValueError('invalid clamp_range ordering or non-positive')

# -------------------------------TRAINER CONFIGS-------------------------------
@dataclasses.dataclass
class LoaderConfig:
    patch_size: int = 128
    batch_size: int = 16

@dataclasses.dataclass
class FocalLossConfig:
    weight: float = 0.5
    gamma: float = 2.0
    reduction: str = 'mean'

@dataclasses.dataclass
class DiceLossConfig:
    weight: float = 0.5
    smooth: float = 1.0

@dataclasses.dataclass
class LossTypesConfig:
    focal: FocalLossConfig = dataclasses.field(default_factory=FocalLossConfig)
    dice: DiceLossConfig = dataclasses.field(default_factory=DiceLossConfig)

@dataclasses.dataclass
class LossConfig:
    alpha_fn: str = 'effective_n'
    en_beta: float = 0.999
    types: LossTypesConfig = dataclasses.field(default_factory=LossTypesConfig)

@dataclasses.dataclass
class OptimConfig:
    opt_cls: str = 'AdamW'
    lr: float = 1e-4
    weight_decay: float = 1e-3
    sched_cls: str | None = 'CosAnneal'
    sched_args: dict[str, typing.Any] = dataclasses.field(default_factory=lambda: {'T_max': 50})

@dataclasses.dataclass
class RuntimeSchedule:
    max_epoch: int = 50
    max_step: int = 1_000_000
    log_every: int = 50
    val_every: int = 1
    ckpt_every: int = 5
    patience: int = 10
    min_delta: float = 0.0005

@dataclasses.dataclass
class RuntimeMonitor:
    metric_name: str = 'iou'
    track_head_name: str = 'layer1'
    track_mode: str = 'max'

@dataclasses.dataclass
class RuntimePrecision:
    use_amp: bool = True

@dataclasses.dataclass
class RuntimeOptimization:
    grad_clip_norm: float | None = 1.0

@dataclasses.dataclass
class RuntimeData:
    domain_ids_name: str | None = None
    domain_vec_name: str | None = None

@dataclasses.dataclass
class RuntimeConfig:
    data: RuntimeData = dataclasses.field(default_factory=RuntimeData)
    schedule: RuntimeSchedule = dataclasses.field(default_factory=RuntimeSchedule)
    monitor: RuntimeMonitor = dataclasses.field(default_factory=RuntimeMonitor)
    precision: RuntimePrecision = dataclasses.field(default_factory=RuntimePrecision)
    optimization: RuntimeOptimization = dataclasses.field(default_factory=RuntimeOptimization)

# ----- TRAINER
@dataclasses.dataclass
class TrainerCfg:
    loader: LoaderConfig = dataclasses.field(default_factory=LoaderConfig)
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        # Example: scheduler-specific requirements
        if self.optim.sched_cls == 'CosAnneal':
            if 'T_max' not in self.optim.sched_args:
                raise ValueError('missing T_max for CosAnneal')

# ---------------------------------RUNNER  CONFIGS-----------------------------
@dataclasses.dataclass
class LogitAdjustConfig:
    train: bool = False
    val: bool = False
    test: bool = False
    alpha: float = 1.0

@dataclasses.dataclass
class PhaseHeads:
    active_heads: list[str] = dataclasses.field(default_factory=lambda: ['layer1'])
    frozen_heads: list[str] | None = None
    masked_classes: dict[str, list[int]] | None = None

@dataclasses.dataclass
class PhaseConfig:
    name: str = 'coarse_head'
    num_epochs: int = 50
    heads: PhaseHeads = dataclasses.field(default_factory=PhaseHeads)
    logit_adjust: LogitAdjustConfig = dataclasses.field(default_factory=LogitAdjustConfig)
    lr_scale: float = 1.0

# ----- RUNNER
@dataclasses.dataclass
class RunnerCfg:
    ckpt_dpath: str = '${exp_root}/checkpoints'
    preview_dpath: str = '${exp_root}/previews'
    phases: list[PhaseConfig] = dataclasses.field(default_factory=lambda: [PhaseConfig()])

# --------------------------------ROOT  CONFIGS--------------------------------
@dataclasses.dataclass
class RootConfig:
    '''Root structured config for landseg.'''

    # root dir for an experiment run
    exp_root: str = './experiment'
    # dev override paths
    dev_settings_path: str | None = None
    # raw input data and configs
    inputs: Inputs = dataclasses.field(default_factory=Inputs)
    # data preparation
    prep: Prep = dataclasses.field(default_factory=Prep)
    # model settings
    models: ModelsCfg = dataclasses.field(default_factory=ModelsCfg)
    # trainer settings
    trainer: TrainerCfg = dataclasses.field(default_factory=TrainerCfg)
    # controller settings
    runner: RunnerCfg = dataclasses.field(default_factory=RunnerCfg)
    # pipeline overrides
    pipeline: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

    def validate_all(self) -> None:
        # delegated to subtrees.
        self.inputs.validate()
        self.models.validate()
        self.trainer.validate()
        # future checks to be added below (e.g., controller phases)

# ------------------------------private  function------------------------------
def _is_resolved(value: typing.Any) -> bool:
    '''Return True if not omegaconf.MISSING and not an interpolation.'''
    if value is omegaconf.MISSING:
        return False
    # OmegaConf marks interpolations as strings like '${...}' pre-resolution
    if isinstance(value, str) and value.strip().startswith('${'):
        return False
    return True
