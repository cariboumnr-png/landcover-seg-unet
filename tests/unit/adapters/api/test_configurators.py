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

'''Unit tests for API configurator classes.'''

# local imports
import landseg.adapters.api.configurators as configurators


# ----- `WorldGridConfigurator` tests
def test_world_grid_configurator(tmp_path):
    '''
    Given: Parameters for world grid canonical lifecycle.
    When: Chaining methods on `WorldGridConfigurator`.
    Then: Correctly populate underlying `RootConfig` and validate.
    '''
    ref_tif = tmp_path / 'ref.tif'
    ref_tif.write_text('ref')

    cfg_builder = configurators.WorldGridConfigurator(
        experiment_root=str(tmp_path),
    )
    cfg_builder.set_grid(
        tile_size=256,
        tile_stride=128,
        crs='EPSG:3161',
        reference_raster=str(ref_tif)
    ).set_output_dpath(
        output_dpath=str(tmp_path / 'world_grids')
    )

    root = cfg_builder.running_root_config
    assert root.command == 'world-grid'
    assert root.data.world_grid.manual_specs.crs_string == 'EPSG:3161'
    assert root.data.world_grid.ref_fpath == str(ref_tif)
    assert root.data.world_grid.tile_size == (256, 256)
    assert root.data.world_grid.tile_stride == (128, 128)
    assert root.data.world_grid.output_dpath == str(tmp_path / 'world_grids')


# ----- `DataHarmonizationConfigurator` tests
def test_data_harmonization_configurator(tmp_path):
    '''
    Given: Parameters for data harmonization ETL.
    When: Chaining methods on `DataHarmonizationConfigurator`.
    Then: Correctly populate underlying `RootConfig` and validate.
    '''
    manifest_json = tmp_path / 'manifest.json'
    manifest_json.write_text('[]')

    cfg_builder = configurators.DataHarmonizationConfigurator(
        experiment_root=str(tmp_path),
    )
    cfg_builder.set_grid(
        tile_size=512,
        tile_stride=64,
    ).set_dataset_manifest(
        dataset_manifest=str(manifest_json),
    ).set_resampling(
        continuous='bilinear',
        categorical='nearest'
    ).set_output_dpath(
        output_dpath=str(tmp_path / 'harmonized')
    ).set_grid_reference(
        output_dpath=str(tmp_path / 'world_grids')
    )

    root = cfg_builder.running_root_config
    assert root.command == 'data-harmonize'
    assert root.data.world_grid.tile_size == (512, 512)
    assert root.data.world_grid.tile_stride == (64, 64)
    assert root.data.world_grid.output_dpath == str(tmp_path / 'world_grids')
    assert root.data.harmonization.dataset_manifest == str(manifest_json)
    assert root.data.harmonization.resampling_continuous == 'bilinear'
    assert root.data.harmonization.resampling_categorical == 'nearest'
    assert root.data.harmonization.output_dpath == str(tmp_path / 'harmonized')


# ----- `DataIngestionConfigurator` tests
def test_data_ingestion_configurator(tmp_path):
    '''
    Given: Parameters for data ingestion pipeline.
    When: Chaining methods on `DataIngestionConfigurator`.
    Then: Correctly populate rebuild, harmonization run, and
        feature engineering.
    '''
    cfg_builder = configurators.DataIngestionConfigurator(
        experiment_root=str(tmp_path),
    )
    cfg_builder.set_rebuild(
        rebuild=True
    ).set_harmonization_run(
        target=1
    ).set_feature_engineering(
        add_topo=['slope', 'tpi'],
        add_spectral=['ndvi', 'ndmi'],
    ).set_collision_policy('overwrite')

    root = cfg_builder.running_root_config
    assert root.command == 'data-ingest'
    assert root.data.ingestion.rebuild is True
    assert root.data.ingestion.harmonization_run == 1
    assert root.data.ingestion.datablocks.add_topo == ['slope', 'tpi']
    assert root.data.ingestion.datablocks.add_spectral == ['ndvi', 'ndmi']
    assert root.data.ingestion.datablocks.collision_policy == 'overwrite'



# ----- `DataPreparationConfigurator` tests
def test_data_preparation_configurator(tmp_path):
    '''
    Given: Parameters for data preparation pipeline.
    When: Chaining methods on `DataPreparationConfigurator`.
    Then: Correctly populate partition and scheme settings.
    '''
    cfg_builder = configurators.DataPreparationConfigurator(
        experiment_root=str(tmp_path),
    )
    cfg_builder.set_features(
        features={'sentinel2': 'rgb_nir'}
    ).set_targets(
        targets={'landcover': 'binary'}
    ).set_partition(
        validation_blocks_ratio=0.15,
        test_holdout_blocks_ratio=0.05
    ).set_oversampling(
        target_head='class_head',
        reward_classes={1: 2.0}
    ).set_rebuild(
        rebuild=True
    )

    root = cfg_builder.running_root_config
    assert root.command == 'data-prepare'
    assert root.data.preparation.datasetview.features == {'sentinel2': 'rgb_nir'}
    assert root.data.preparation.datasetview.targets == {'landcover': 'binary'}
    assert root.data.preparation.partition.val_ratio == 0.15
    assert root.data.preparation.partition.test_ratio == 0.05
    assert root.data.preparation.datasetview.focal_target == 'class_head'
    assert root.data.preparation.scoring.reward == {1: 2.0}
    assert root.data.preparation.rebuild is True


# ----- `TrainingSessionConfigurator` tests
def test_training_session_configurator(tmp_path):
    '''
    Given: Parameters for model training session.
    When: Chaining methods on `TrainingSessionConfigurator`.
    Then: Correctly populate model, optimizer, data loader, and tasks.
    '''
    cfg_builder = configurators.TrainingSessionConfigurator(
        experiment_root=str(tmp_path),
    )

    cfg_builder.set_model(
        body='unet',
        bottleneck='conv',
        base_channel=64
    ).set_optimization(
        optimizer='AdamW',
        learning_rate=1e-3,
        weight_decay=1e-4,
        scheduler='CosAnneal'
    ).set_data_loading(
        batch_size=16,
        patch_size=256
    ).set_domain_source(
        category_domain='sample_domain_1',
        continuous_domain=None
    ).set_tasks(
        logit_adjust_alpha=0.5,
        exclude_classes={'class_head': [255]},
        loss_weights={'ce': 1.0},
        head_weights={'class_head': 1.0}
    ).set_runtime(
        max_epochs=10,
        active_heads=['class_head'],
        patience_epoch=5,
        track_heads={'class_head': 1.0}
    )

    root = cfg_builder.running_root_config
    assert root.command == 'model-train'
    assert root.models.model_body == 'unet'
    assert root.models.bottleneck == 'conv'
    assert root.models.model_body_registry['unet'].base_ch == 64
    assert root.session.engine_optim.opt_cls == 'AdamW'
    assert root.session.engine_optim.lr == 1e-3
    assert root.session.dataloader.batch_size == 16
    assert root.session.dataloader.patch_size == 256
    assert root.data.specification.domain_ids_name == 'sample_domain_1'


# ----- `BaseConfigurator.preflight` tests
def test_configurator_preflight(mocker, tmp_path):
    '''
    Given: A configurator instance.
    When: Calling `.preflight()` on the configurator.
    Then: Delegate to `api.run_preflight` with current configuration.
    '''
    mock_run_preflight = mocker.patch(
        'landseg.adapters.api.api.run_preflight',
        return_value='report'
    )
    cfg_builder = configurators.WorldGridConfigurator(
        experiment_root=str(tmp_path),
    )
    res = cfg_builder.preflight(strict=True)

    assert res == 'report'
    mock_run_preflight.assert_called_once()
    assert mock_run_preflight.call_args.args[1] == 'world-grid'
    assert mock_run_preflight.call_args.kwargs['strict'] is True


# ----- Canonical Decoupled & E2E Resolution Tests
def test_canonical_decoupled_configurators(tmp_path):
    '''
    Given: Explicit output and upstream directory paths (without experiment_root).
    When: Instantiating configurators in canonical standalone recipe mode.
    Then: Correctly wire output and upstream paths on the underlying RootConfig.
    '''
    grid_out = tmp_path / 'grids'
    harm_out = tmp_path / 'harmonized'
    ingest_out = tmp_path / 'ingested'
    prep_out = tmp_path / 'prepared'
    results_out = tmp_path / 'results'

    # 1. WorldGridConfigurator
    wg = configurators.WorldGridConfigurator(output_dpath=grid_out)
    assert wg.running_root_config.data.world_grid.output_dpath == str(grid_out)

    # 2. DataHarmonizationConfigurator
    dh = configurators.DataHarmonizationConfigurator(
        output_dpath=harm_out,
        grid_dpath=grid_out,
    )
    assert dh.running_root_config.data.harmonization.output_dpath == str(harm_out)
    assert dh.running_root_config.data.world_grid.output_dpath == str(grid_out)

    # 3. DataIngestionConfigurator
    di = configurators.DataIngestionConfigurator(
        output_dpath=ingest_out,
        harmonization_dpath=harm_out,
    )
    assert di.running_root_config.data.ingestion.output_dpath == str(ingest_out)
    assert di.running_root_config.data.harmonization.output_dpath == str(harm_out)

    # 4. DataPreparationConfigurator
    dp = configurators.DataPreparationConfigurator(
        output_dpath=prep_out,
        ingestion_dpath=ingest_out,
    )
    assert dp.running_root_config.data.preparation.output_dpath == str(prep_out)
    assert dp.running_root_config.data.ingestion.output_dpath == str(ingest_out)

    # 5. TrainingSessionConfigurator
    ts = configurators.TrainingSessionConfigurator(
        output_dpath=results_out,
        preparation_dpath=prep_out,
    )
    ts.set_runtime(
        max_epochs=1,
        active_heads=None,
        patience_epoch=None,
        track_heads=None,
    )
    assert ts.running_root_config.session.output_dpath == str(results_out)
    assert ts.running_root_config.data.preparation.output_dpath == str(prep_out)


def test_study_sweep_configurator_canonical(tmp_path):
    '''
    Given: Canonical parameters for study sweep without experiment_root.
    When: Initializing StudySweepConfigurator with output_dpath and optuna_storage.
    Then: Correctly sets sweep storage, session output, and upstream preparation paths.
    '''
    results_out = tmp_path / 'sweep_results'
    prep_out = tmp_path / 'prepared_data'
    prep_out.mkdir(parents=True, exist_ok=True)
    storage = 'sqlite:///' + str(tmp_path / 'custom_optuna.db').replace('\\', '/')

    sweep_cfg = configurators.StudySweepConfigurator(
        output_dpath=results_out,
        preparation_dpath=prep_out,
        optuna_storage=storage,
    )
    sweep_cfg.set_runtime(
        max_epochs=1,
        active_heads=None,
        patience_epoch=None,
        track_heads=None,
    )
    root = sweep_cfg.running_root_config
    assert root.command == 'study-sweep'
    assert root.session.output_dpath == str(results_out)
    assert root.data.preparation.output_dpath == str(prep_out)
    assert root.study.sweep.storage == storage


def test_configurator_set_output_and_upstream_dpaths(tmp_path):
    '''
    Given: An initialized configurator.
    When: Calling set_output_dpath and set_upstream_dpath fluent methods.
    Then: Updates the respective target and upstream paths on the underlying config.
    '''
    ts = configurators.TrainingSessionConfigurator(experiment_root=tmp_path)
    new_results = tmp_path / 'custom_results'
    new_prep = tmp_path / 'custom_prep'

    ts.set_runtime(
        max_epochs=1,
        active_heads=None,
        patience_epoch=None,
        track_heads=None,
    )
    ts.set_output_dpath(new_results).set_upstream_dpath(new_prep)
    root = ts.running_root_config

    assert root.session.output_dpath == str(new_results)
    assert root.data.preparation.output_dpath == str(new_prep)


def test_e2e_unified_experiment_root_resolution(tmp_path):
    '''
    Given: An experiment_root path.
    When: Initializing configurators with experiment_root (e2e notebook mode).
    Then: Cascades default artifact paths under <exp_root>/artifacts/... and /results.
    '''
    root = tmp_path / 'unified_exp'

    wg = configurators.WorldGridConfigurator(root)
    assert wg.running_root_config.data.world_grid.output_dpath == str(
        root / 'artifacts' / 'world_grids'
    )

    dh = configurators.DataHarmonizationConfigurator(root)
    assert dh.running_root_config.data.harmonization.output_dpath == str(
        root / 'artifacts' / 'harmonized_data'
    )
    assert dh.running_root_config.data.world_grid.output_dpath == str(
        root / 'artifacts' / 'world_grids'
    )

    ts = configurators.TrainingSessionConfigurator(root)
    ts.set_runtime(
        max_epochs=1,
        active_heads=None,
        patience_epoch=None,
        track_heads=None,
    )
    assert ts.running_root_config.session.output_dpath == str(root / 'results')
    assert ts.running_root_config.data.preparation.output_dpath == str(
        root / 'artifacts' / 'prepared_data'
    )

