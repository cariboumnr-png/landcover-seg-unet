# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

'''Integration tests for complete end-to-end experiment lifecycle.'''

# standard imports
import typing
# third-party imports
import omegaconf
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.execution.workflows.e2e_experiment as e2e_experiment


# ----- `execute_e2e_experiment` lifecycle integration tests


def test_full_ml_experiment_and_evaluate_lifecycle(
    tmp_path,
    dummy_data_paths,
):
    '''
    Given: Canonical dummy raw raster datasets and manifest.
    When: `execute_e2e_experiment` runs end-to-end, followed by evaluation.
    Then: Produce all ETL artifacts, train UNet model, and evaluate checkpoint.
    '''
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)

    # 1. execution context
    exp_root = tmp_path / 'exp'
    cfg_schema.execution.exp_root = str(exp_root)

    # 2. world grid configuration
    grid_cfg = cfg_schema.data.world_grid
    grid_cfg.mode = 'ref'
    grid_cfg.output_dpath = str(tmp_path / 'world_grids')
    grid_cfg.ref_fpath = dummy_data_paths.extent
    grid_cfg.tile_size = (256, 256)
    grid_cfg.tile_stride = (128, 128)

    # 3. harmonization configuration
    harm_cfg = cfg_schema.data.harmonization
    harm_cfg.dataset_manifest = dummy_data_paths.manifest
    harm_cfg.output_dpath = str(tmp_path / 'harmonized')

    # 4. ingestion configuration
    ingest_cfg = cfg_schema.data.ingestion
    ingest_cfg.output_dpath = str(tmp_path / 'ingested_data')
    ingest_cfg.rebuild = True

    # 5. preparation configuration
    prep_cfg = cfg_schema.data.preparation
    prep_cfg.output_dpath = str(tmp_path / 'prepared_data')
    prep_cfg.rebuild = True
    prep_cfg.datasetview.valid_pxs = {'image': 0.05}
    prep_cfg.datasetview.focal_target = None
    prep_cfg.partition.val_ratio = 0.2
    prep_cfg.partition.test_ratio = 0.1
    prep_cfg.partition.buffer_step = 1
    prep_cfg.scoring.reward = {0: 1.0}
    prep_cfg.scoring.alpha = 1.0
    prep_cfg.scoring.beta = 0.5
    prep_cfg.hydration.max_skew_rate = 1.5

    # 6. session configuration (optimized for fast CPU execution)
    cfg_schema.models.model_body_registry.unet.base_ch = 4
    cfg_schema.session.dataloader.patch_size = 16
    cfg_schema.session.dataloader.batch_size = 2
    cfg_schema.session.engine.engine_exec.use_amp = False

    phase_0 = cfg_schema.session.orchestration.curriculum.single.phases[0]
    phase_0.num_epochs = 1
    phase_0.start_epoch = 1

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    # execute full experiment workflow
    e2e_experiment.execute_e2e_experiment(config)

    # verify ETL outputs
    world_grid_fpath = (
        tmp_path / 'world_grids' / 'grid_row_256_128_col_256_128.json'
    )
    assert world_grid_fpath.exists()

    harm_run_1 = tmp_path / 'harmonized' / 'run_0001'
    assert (harm_run_1 / 'report.json').exists()
    assert (harm_run_1 / 'valid_pixel_mask.vrt').exists()

    ingest_root = tmp_path / 'ingested_data'
    assert (ingest_root / 'data_blocks' / 'catalog.json').exists()
    assert (ingest_root / 'data_blocks' / 'schema.json').exists()

    prep_root = tmp_path / 'prepared_data'
    assert (prep_root / 'block_splits_source.json').exists()
    assert (prep_root / 'block_splits_prepared.json').exists()
    assert (prep_root / 'image_stats.json').exists()

    # verify model training outputs
    train_report_fpath = exp_root / 'results' / 'run_0001' / 'report.json'
    assert train_report_fpath.exists()
    train_report = artifacts.Controller[dict](train_report_fpath).fetch()
    assert train_report
    assert train_report['status'] == 'SUCCESS'

    ckpt_dir = exp_root / 'results' / 'run_0001' / 'checkpoints'
    assert ckpt_dir.exists()
    ckpt_files = list(ckpt_dir.glob('*.pt'))
    assert len(ckpt_files) >= 1

    # execute model evaluation on the trained checkpoint
    eval_cfg = config.session.evaluation
    eval_cfg.checkpoint = ckpt_files[0]
    eval_cfg.split = 'val'

    eval_metric = pipelines.ModelEvaluation(config).run()
    assert eval_metric is not None

    eval_report_fpath = exp_root / 'results' / 'run_0002' / 'report.json'
    assert eval_report_fpath.exists()
    eval_report = artifacts.Controller[dict](eval_report_fpath).fetch()
    assert eval_report
    assert eval_report['status'] == 'SUCCESS'
