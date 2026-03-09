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
'''doc'''

# standard imports
from __future__ import annotations
import dataclasses
import typing

# NOTE: This module mirrors the Hydra/YAML config tree using
# Python dataclasses, suitable for OmegaConf structured configs.
# It is intentionally verbose to provide a single source of truth
# for configuration shape, defaults, and types.
#
# Usage at the CLI boundary (conceptual):
#   schema = OmegaConf.structured(RootConfig)
#   merged = OmegaConf.merge(schema, hydra_cfg)
#   OmegaConf.resolve(merged)
#   cfg: RootConfig = OmegaConf.to_object(merged)
#
# Field defaults below reflect the repository YAMLs; string defaults
# may contain Hydra/OmegaConf interpolations (e.g., "${exp_root}").


# ----- input_extent
@dataclasses.dataclass
class Extent:
    # ref mode
    dirpath: str = '${exp_root}/input/extent_ref'
    filename: str = 'on_3161_30m.tif'
    # aoi / tiles mode
    origin: tuple[float, float] = (0.0, 0.0)
    pixel_size: tuple[float, float] = (0.0, 0.0)
    grid_extent: tuple[float, float] = (0.0, 0.0)
    grid_shape: tuple[int, int] = (0, 0)

@dataclasses.dataclass
class InputExtentCfg:
    crs: str = 'EPSG:3161'
    mode: str = 'ref'  # 'ref' | 'aoi' | 'tiles'
    inputs: Extent = dataclasses.field(default_factory=Extent)

# ----- dataprep_grid
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
    id: str = '''
    grid_row_${grid.tile_size.row}_${grid.tile_overlap.row}_
    col_${grid.tile_size.col}_${grid.tile_overlap.col}
    '''
    output_dirpath: str = '${exp_root}/artifacts/world_grids'
    version: str = 'v1'
    tile_size: TileSize = dataclasses.field(default_factory=TileSize)
    tile_overlap: TileOverlap = dataclasses.field(default_factory=TileOverlap)

# ----- input_domain
@dataclasses.dataclass
class DomainFile:
    name: str = ''
    index_base: int = 1

@dataclasses.dataclass
class InputDomainCfg:
    input_dirpath: str = '${exp_root}/input/domain'
    files: list[DomainFile] = dataclasses.field(default_factory=lambda: [
        DomainFile(name='ecodist.tif', index_base=1),
        DomainFile(name='geology.tif', index_base=1)
    ])

# ----- dataprep_domain
@dataclasses.dataclass
class PrepDomainCfg:
    output_dirpath: str = '${exp_root}/artifacts/domain'
    as_ids: str = 'ecodist'
    as_vec: str = 'geology'
    valid_threshold: float = 0.7
    target_variance: float = 0.9

# ----- input_data
@dataclasses.dataclass
class FileNames:
    fit_image: str = 'image.tif'
    fit_label: str = 'label.tif'
    test_image: str | None = 'image.tif'
    test_label: str | None = 'label.tif'
    config: str = 'config.json'

@dataclasses.dataclass
class Dirs:
    fit: str = '${exp_root}/input/${input_data.name}/fit'
    test: str = '${exp_root}/input/${input_data.name}/test'
    config: str = '${exp_root}/input/${input_data.name}/config'

@dataclasses.dataclass
class FilePaths:
    fit_image: str = '${input_data.dirs.fit}/${input_data.file_names.fit_image}'
    fit_label: str = '${input_data.dirs.fit}/${input_data.file_names.fit_label}'
    test_image: str = '${input_data.dirs.test}/${input_data.file_names.test_image}'
    test_label: str = '${input_data.dirs.test}/${input_data.file_names.test_label}'
    config: str = '${input_data.dirs.config}/${input_data.file_names.config}'

@dataclasses.dataclass
class InputDataCfg:
    name: str = 'demo_data'
    input_dirpath: str = '${exp_root}/input/${input_data.name}'
    file_names: FileNames = dataclasses.field(default_factory=FileNames)
    dirs: Dirs = dataclasses.field(default_factory=Dirs)
    file_paths: FilePaths = dataclasses.field(default_factory=FilePaths)

# ----- prep_data
@dataclasses.dataclass
class FitBlocks:
    fit_raster_windows: str = '${prep_data.output_dirpath}/fit/fit_raster_windows.pkl'
    fit_blocks_dir: str = '${prep_data.output_dirpath}/fit/blocks'
    fit_all_blocks: str = '${prep_data.output_dirpath}/fit/all_blocks.json'
    fit_valid_blocks: str = '${prep_data.output_dirpath}/fit/valid_blocks.json'

@dataclasses.dataclass
class FitPostBlocks:
    fit_image_stats: str = '${prep_data.output_dirpath}/fit/image_stats.json'
    label_count_global: str = '${prep_data.output_dirpath}/fit/count_global.json'
    block_scores: str = '${prep_data.output_dirpath}/fit/score.json'
    train_blocks_split: str = '${prep_data.output_dirpath}/fit/train_blocks_split.json'
    val_blocks_split: str = '${prep_data.output_dirpath}/fit/val_blocks_split.json'
    label_count_train: str = '${prep_data.output_dirpath}/fit/count_train.json'

@dataclasses.dataclass
class TestBlocks:
    test_raster_windows: str = '${prep_data.output_dirpath}/test/test_raster_windows.pkl'
    test_blocks_dir: str = '${prep_data.output_dirpath}/test/blocks'
    test_all_blocks: str = '${prep_data.output_dirpath}/test/all_blocks.json'
    test_valid_blocks: str = '${prep_data.output_dirpath}/test/valid_blocks.json'

@dataclasses.dataclass
class TestPostBlocks:
  # test blocks stats for normalization
    test_image_stats: str = '${prep_data.output_dirpath}/test/image_stats.json'

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
    output_dirpath: str = '${exp_root}/artifacts/data_cache/${input_data.name}'
    fit_blocks: FitBlocks = dataclasses.field(default_factory=FitBlocks)
    fit_post_blocks: FitPostBlocks = dataclasses.field(default_factory=FitPostBlocks)
    test_blocks: TestBlocks = dataclasses.field(default_factory=TestBlocks)
    test_post_blocks: TestPostBlocks = dataclasses.field(default_factory=TestPostBlocks)
    threshold: Thresholds = dataclasses.field(default_factory=Thresholds)
    scoring: Scoring = dataclasses.field(default_factory=Scoring)

# ------------------------------- experiment ---------------------------
@dataclasses.dataclass
class LogitAdjustConfig:
    train: bool = False
    val: bool = False
    test: bool = False
    alpha: float = 1.0

@dataclasses.dataclass
class PhaseHeads:
    active_heads: tuple[str, ...] = ('layer1',)
    frozen_heads: tuple[str, ...] | None = None
    masked_classes: dict[str, list[int]] | None = None

@dataclasses.dataclass
class PhaseConfig:
    name: str = 'coarse_head'
    num_epochs: int = 50
    heads: PhaseHeads = dataclasses.field(default_factory=PhaseHeads)
    logit_adjust: LogitAdjustConfig = dataclasses.field(default_factory=LogitAdjustConfig)
    lr_scale: float = 1.0

@dataclasses.dataclass
class ExperimentCfg:
    ckpt_dpath: str = '${exp_root}/checkpoints'
    preview_dpath: str = '${exp_root}/previews'
    phases: list[PhaseConfig] = dataclasses.field(default_factory=lambda: [PhaseConfig()])

# --------------------------------- models ------------------------------
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
            'downs': ConvParams,
            'ups': ConvParams
        }
    )

@dataclasses.dataclass
class UnetPPBody:
    base_ch: int = 32
    conv_params: dict[str, typing.Any] = dataclasses.field(
        default_factory=lambda: {
            'downs': ConvParams,
            'nodes': ConvParams
        }
    )

@dataclasses.dataclass
class ConditioningConcat:
    out_dim: int = 4
    use_ids: bool = True
    use_vec: bool = True
    use_mlp: bool = True

@dataclasses.dataclass
class ConditioningFiLM:
    embed_dim: int = 4
    use_ids: bool = True
    use_vec: bool = True
    hidden: int = 128

@dataclasses.dataclass
class ConditioningConfig:
    mode: str = 'hybrid'  # 'none' | 'concat' | 'film' | 'hybrid'
    concat: ConditioningConcat = dataclasses.field(default_factory=ConditioningConcat)
    film: ConditioningFiLM = dataclasses.field(default_factory=ConditioningFiLM)

@dataclasses.dataclass
class ModelFlags:
    enable_logit_adjust: bool = True
    enable_clamp: bool = True

@dataclasses.dataclass
class ModelsCfg:
    body: dict[str, typing.Any] = dataclasses.field(
        default_factory=lambda: {
            'unet': UnetBody(),
            'unetpp': UnetPPBody(),
        }
    )
    conditioning: ConditioningConfig = dataclasses.field(default_factory=ConditioningConfig)
    clamp_range: tuple[float, float] = (1e-4, 1e4)
    flags: ModelFlags = dataclasses.field(default_factory=ModelFlags)

# --------------------------------- trainer -----------------------------
@dataclasses.dataclass
class LoaderConfig:
    block_size: int = 256
    patch_size: int = 128
    batch_size: int = 16
    stream_cache: int = 16

@dataclasses.dataclass
class FocalLossConfig:
    weight: float = 0.5
    alpha: float | None = None
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
    sched_cls: str = 'CosAnneal'
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
    domain_ids_name: str = 'ecodist'
    domain_vec_name: str = 'geology'

@dataclasses.dataclass
class RuntimeConfig:
    data: RuntimeData = dataclasses.field(default_factory=RuntimeData)
    schedule: RuntimeSchedule = dataclasses.field(default_factory=RuntimeSchedule)
    monitor: RuntimeMonitor = dataclasses.field(default_factory=RuntimeMonitor)
    precision: RuntimePrecision = dataclasses.field(default_factory=RuntimePrecision)
    optimization: RuntimeOptimization = dataclasses.field(default_factory=RuntimeOptimization)

@dataclasses.dataclass
class TrainerCfg:
    model_body: str = 'unetpp'
    loader: LoaderConfig = dataclasses.field(default_factory=LoaderConfig)
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)

# ----- profile
@dataclasses.dataclass
class ProfileConfig:
    name: str = "default"

# --------------------------------- root --------------------------------
@dataclasses.dataclass
class RootConfig:
    """Root structured config for landseg."""

    # composition defaults are Hydra-side; keep explicit sections here
    exp_root: str = './experiment'
    dev_settings_path: str | None = None
    #
    input_extent: InputExtentCfg = dataclasses.field(default_factory=InputExtentCfg)
    input_domain: InputDomainCfg = dataclasses.field(default_factory=InputDomainCfg)
    input_data: InputDataCfg = dataclasses.field(default_factory=InputDataCfg)
    #
    prep_grid: PrepGridCfg = dataclasses.field(default_factory=PrepGridCfg)
    prep_domain: PrepDomainCfg = dataclasses.field(default_factory=PrepDomainCfg)
    prep_data: PrepDataCfg = dataclasses.field(default_factory=PrepDataCfg)
    #
    models: ModelsCfg = dataclasses.field(default_factory=ModelsCfg)
    trainer: TrainerCfg = dataclasses.field(default_factory=TrainerCfg)
    #
    experiment: ExperimentCfg = dataclasses.field(default_factory=ExperimentCfg)
    profile: ProfileConfig = dataclasses.field(default_factory=ProfileConfig)

# This module intentionally does not import OmegaConf to avoid a hard
# runtime dependency here. A helper function can be created in the CLI
# module to compose/merge/resolve against this schema.
