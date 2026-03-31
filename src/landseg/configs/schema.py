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

# -------------------------------DATA FOUNDATION-------------------------------
# ----- grid
@dataclasses.dataclass
class Extent:
    default_input_dpath: str = '${exp_root}/input/extent_reference'
    filename: str = ''
    filepath: str = ''
    origin: tuple[float, float] = (0.0, 0.0)
    pixel_size: tuple[float, float] = (0.0, 0.0)
    grid_extent: tuple[float, float] | None = None
    grid_shape: tuple[int, int] | None = None

@dataclasses.dataclass
class TileSize:
    row: int = 256
    col: int = 256

@dataclasses.dataclass
class TileOverlap:
    row: int = 0
    col: int = 0

@dataclasses.dataclass
class Grid:
    mode: str = 'ref'
    crs: str = omegaconf.MISSING
    extent: Extent = dataclasses.field(default_factory=Extent)
    tile_size: TileSize = dataclasses.field(default_factory=TileSize)
    tile_overlap: TileOverlap = dataclasses.field(default_factory=TileOverlap)

    def __post_init__(self):
        if not self.extent.filepath:
            self.extent.filepath = os.path.join(
                self.extent.default_input_dpath, self.extent.filename
            )
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
            if not os.path.exists(self.extent.filepath):
                raise FileNotFoundError('Mode=ref but ref raster not provided')
        elif self.mode == 'aoi':
            if not all(self.extent.pixel_size):
                raise ValueError('Mode=aoi|tiles but pixel size has zero(s)')
            if not all(self.extent.grid_extent or ()):
                raise ValueError('Mode=aoi but grid extent has zero(s)')
        elif self.mode == 'tiles':
            if not all(self.extent.pixel_size):
                raise ValueError('Mode=aoi|tiles but pixel size has zero(s)')
            if not all(self.extent.grid_shape or ()):
                raise ValueError('Mode=tiles but grid shape has zero(s)')

# ----- domain
@dataclasses.dataclass
class DomainFile:
    name: str = omegaconf.MISSING
    path: str = ''
    index_base: int = omegaconf.MISSING

@dataclasses.dataclass
class Domains:
    default_input_dpath: str = '${exp_root}/input/domain_knowledge'
    files: list[DomainFile] = dataclasses.field(default_factory=lambda: [])
    valid_threshold: float = 0.7
    target_variance: float = 0.9

    def validate(self):
        for file in self.files:
            if not file.path:
                file.path = os.path.join(self.default_input_dpath, file.name)
            if not os.path.exists(file.path):
                raise FileNotFoundError(f'Invalid domain raster: {file.path}')

# ----- data
@dataclasses.dataclass
class FileNames:
    dev_image: str = omegaconf.MISSING
    dev_label: str = omegaconf.MISSING
    test_image: str = omegaconf.MISSING
    test_label: str = omegaconf.MISSING
    config: str = omegaconf.MISSING

@dataclasses.dataclass
class FilePaths:
    dev_image: str = ''
    dev_label: str = ''
    test_image: str = ''
    test_label: str = ''
    config: str = ''

@dataclasses.dataclass
class General:
    ignore_index: int = 255
    image_dem_pad: int = 8

@dataclasses.dataclass
class DataBlocks:
    name: str = omegaconf.MISSING
    default_input_dpath: str = '${exp_root}/input/${inputs.data.name}'
    filenames: FileNames = dataclasses.field(default_factory=FileNames)
    filepaths: FilePaths = dataclasses.field(default_factory=FilePaths)
    general: General = dataclasses.field(default_factory=General)

    def __post_init__(self):
        # compose file paths
        root = self.default_input_dpath
        paths = self.filepaths
        names = self.filenames
        # dev image
        if not self.filepaths.dev_image:
            paths.dev_image = os.path.join(root, 'dev', names.dev_image)
        # dev label
        if not self.filepaths.dev_label:
            paths.dev_label = os.path.join(root, 'dev', names.dev_label)
        # test image (optional)
        if not self.filepaths.test_image:
            paths.test_image = os.path.join(root, 'test', names.test_image)
        # test label (optional)
        if not self.filepaths.dev_label:
            paths.dev_label = os.path.join(root, 'test', names.dev_label)
        # config JSON
        if not self.filepaths.config:
            paths.config = os.path.join(root, names.config)
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
        _must_exist(self.filepaths.dev_image, 'dev image')
        _must_exist(self.filepaths.dev_label, 'dev label')
        _must_exist(self.filepaths.config, 'config json')

# ----- composite
@dataclasses.dataclass
class DataFoundation:
    grid: Grid = dataclasses.field(default_factory=Grid)
    domains: Domains = dataclasses.field(default_factory=Domains)
    datablocks: DataBlocks = dataclasses.field(default_factory=DataBlocks)
    output_dpath: str = '${exp_root}/artifacts/foundation'

    def validate(self) -> None:
        self.grid.validate()
        self.domains.validate()
        self.datablocks.validate()

# -------------------------------DATA  TRANSFORM-------------------------------
@dataclasses.dataclass
class Thresholds:
    blk_thres_dev: float = 0.75
    blk_thres_test: float = 0.1

@dataclasses.dataclass
class Partition:
    val_ratio: float = 0.1
    test_ratio: float = 0.0
    buffer_step: int = 1

@dataclasses.dataclass
class Scoring:
    reward: dict[int, float] = dataclasses.field(default_factory=dict)
    alpha: float = 1.0
    beta: float = 0.0

@dataclasses.dataclass
class Hydration:
    max_skew_rate: float = 10.0

@dataclasses.dataclass
class DataTransform:
    threshold: Thresholds = dataclasses.field(default_factory=Thresholds)
    partition: Partition = dataclasses.field(default_factory=Partition)
    scoring: Scoring = dataclasses.field(default_factory=Scoring)
    hydration: Hydration = dataclasses.field(default_factory=Hydration)
    output_dpath: str = '${exp_root}/artifacts/transform'

    def validate(self):
        pass

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
    foundation: DataFoundation = dataclasses.field(default_factory=DataFoundation)
    # data preparation
    transform: DataTransform = dataclasses.field(default_factory=DataTransform)
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
        self.foundation.validate()
        self.transform.validate()
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
