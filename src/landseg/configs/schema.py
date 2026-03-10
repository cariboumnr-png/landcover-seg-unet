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
    dirpath: str = '${exp_root}/input/extent_reference'
    filename: str = ''
    origin: tuple[float, float] = (0.0, 0.0)
    pixel_size: tuple[float, float] = (0.0, 0.0)
    grid_extent: tuple[float, float] = (0.0, 0.0)
    grid_shape: tuple[int, int] = (0, 0)

@dataclasses.dataclass
class InputExtentCfg:
    crs: str = omegaconf.MISSING
    mode: str = 'ref'  # 'ref' | 'aoi' | 'tiles'
    inputs: Extent = dataclasses.field(default_factory=Extent)

    def __post_init__(self):
        # defer validation until config is composed and resolved
        if not _is_resolved(self.crs) or not _is_resolved(self.mode):
            return
        # validation checks
        if not bool(re.fullmatch(r'epsg:\d+', self.crs, re.I)):
            raise ValueError('Invalid CRS identifier. Must be [EPSG:....]')
        if self.mode not in {'ref', 'aoi', 'tiles'}:
            raise ValueError(f'Invalid mode: {self.mode}')
        if self.mode == 'ref':
            ref = os.path.join(self.inputs.dirpath, self.inputs.filename)
            if not os.path.exists(ref):
                raise FileNotFoundError('Mode=ref but ref raster not provided')
        elif self.mode == 'aoi':
            if not all(self.inputs.pixel_size):
                raise ValueError('Mode=aoi|tiles but pixel size has zero(s)')
            if not all(self.inputs.grid_extent):
                raise ValueError('Mode=aoi but grid extent has zero(s)')
        elif self.mode == 'tiles':
            if not all(self.inputs.pixel_size):
                raise ValueError('Mode=aoi|tiles but pixel size has zero(s)')
            if not all(self.inputs.grid_shape):
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

    def __post_init__(self):
        for file in self.files:
            fpath = os.path.join(self.input_dirpath, file.filename)
            if file.filename and not os.path.exists(fpath):
                raise FileNotFoundError(f'Invalid domain raster: {fpath}')

# ----- data
@dataclasses.dataclass
class FileNames:
    fit_image: str = omegaconf.MISSING
    fit_label: str = omegaconf.MISSING
    test_image: str | None = omegaconf.MISSING
    test_label: str | None = omegaconf.MISSING
    config: str = omegaconf.MISSING

@dataclasses.dataclass
class Dirs:
    fit: str = '${exp_root}/input/${inputs.data.name}/fit'
    test: str = '${exp_root}/input/${inputs.data.name}/test'
    config: str = '${exp_root}/input/${inputs.data.name}/config'

@dataclasses.dataclass
class FilePaths:
    fit_image: str = '${inputs.data.dirs.fit}/${inputs.data.filenames.fit_image}'
    fit_label: str = '${inputs.data.dirs.fit}/${inputs.data.filenames.fit_label}'
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

        fps = self.filepaths
        if not self.name:
            raise ValueError('Input data name not provided')
        if not os.path.exists(fps.fit_image):
            raise FileNotFoundError(f'Invalid fit image: {fps.fit_image}')
        if not os.path.exists(fps.fit_label):
            raise FileNotFoundError(f'Invalid fit label: {fps.fit_label}')
        if fps.test_image and not os.path.exists(fps.test_image):
            raise FileNotFoundError(f'Invalid test image: {fps.fit_image}')
        if fps.test_label and not os.path.exists(fps.test_label):
            raise FileNotFoundError(f'Invalid test label: {fps.fit_label}')
        if not os.path.exists(fps.config):
            raise FileNotFoundError(f'Invalid config json: {fps.config}')

# ----- INPUTS
@dataclasses.dataclass
class Inputs:
    extent: InputExtentCfg = dataclasses.field(default_factory=InputExtentCfg)
    domain: InputDomainCfg = dataclasses.field(default_factory=InputDomainCfg)
    data: InputDataCfg = dataclasses.field(default_factory=InputDataCfg)

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
class FitBlocks:
    raster_windows: str = '${prep.data.output_dirpath}/fit/fit_raster_windows.pkl'
    blocks_dir: str = '${prep.data.output_dirpath}/fit/blocks'
    all_blocks: str = '${prep.data.output_dirpath}/fit/all_blocks.json'
    valid_blocks: str = '${prep.data.output_dirpath}/fit/valid_blocks.json'

@dataclasses.dataclass
class FitPostBlocks:
    image_stats: str = '${prep.data.output_dirpath}/fit/image_stats.json'
    label_count_global: str = '${prep.data.output_dirpath}/fit/count_global.json'
    block_scores: str = '${prep.data.output_dirpath}/fit/score.json'
    train_blocks_split: str = '${prep.data.output_dirpath}/fit/train_blocks_split.json'
    val_blocks_split: str = '${prep.data.output_dirpath}/fit/val_blocks_split.json'
    label_count_train: str = '${prep.data.output_dirpath}/fit/count_train.json'

@dataclasses.dataclass
class TestBlocks:
    raster_windows: str = '${prep.data.output_dirpath}/test/test_raster_windows.pkl'
    blocks_dir: str = '${prep.data.output_dirpath}/test/blocks'
    all_blocks: str = '${prep.data.output_dirpath}/test/all_blocks.json'
    valid_blocks: str = '${prep.data.output_dirpath}/test/valid_blocks.json'

@dataclasses.dataclass
class TestPostBlocks:
  # test blocks stats for normalization
    image_stats: str = '${prep.data.output_dirpath}/test/image_stats.json'

@dataclasses.dataclass
class Thresholds:
    blk_thres_fit: float = 0.75
    blk_thres_test: float = 0.1

@dataclasses.dataclass
class Scoring:
    head: str = 'layer1'
    alpha: float = 0.6
    beta: float = 0.8
    epsilon: float = 1e-12
    reward: tuple[int, ...] = ()

@dataclasses.dataclass
class PrepDataCfg:
    output_dirpath: str = '${exp_root}/artifacts/data_cache/${inputs.data.name}'
    fit_blocks: FitBlocks = dataclasses.field(default_factory=FitBlocks)
    fit_post_blocks: FitPostBlocks = dataclasses.field(default_factory=FitPostBlocks)
    test_blocks: TestBlocks = dataclasses.field(default_factory=TestBlocks)
    test_post_blocks: TestPostBlocks = dataclasses.field(default_factory=TestPostBlocks)
    threshold: Thresholds = dataclasses.field(default_factory=Thresholds)
    scoring: Scoring = dataclasses.field(default_factory=Scoring)

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

# -------------------------------TRAINER CONFIGS-------------------------------
@dataclasses.dataclass
class LoaderConfig:
    patch_dim_denom: int = 2
    batch_size: int = 16

@dataclasses.dataclass
class FocalLossConfig:
    weight: float = 0.5
    alpha: list[float] | None = None
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
    model_body: str = 'unetpp'
    loader: LoaderConfig = dataclasses.field(default_factory=LoaderConfig)
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)

# -------------------------------CONTROLLER  CONFIGS---------------------------
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

# ----- CONTROLLER
@dataclasses.dataclass
class ControllerCfg:
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
    controller: ControllerCfg = dataclasses.field(default_factory=ControllerCfg)
    # optional profile overrides
    profile: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

# ------------------------------private  function------------------------------
def _is_resolved(value: typing.Any) -> bool:
    '''Return True if not omegaconf.MISSING and not an interpolation.'''
    if value is omegaconf.MISSING:
        return False
    # OmegaConf marks interpolations as strings like '${...}' pre-resolution
    if isinstance(value, str) and value.strip().startswith('${'):
        return False
    return True
