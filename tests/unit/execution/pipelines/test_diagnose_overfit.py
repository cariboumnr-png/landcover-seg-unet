# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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

# pylint: disable=protected-access

'''
Unit tests for diagnose overfit pipeline (diagnose_overfit.py).
'''

# standard imports
import json
import os
import shutil
import typing
# third-party imports
import omegaconf
# local imports
import landseg.configs as configs
import landseg.core as core
import landseg.execution.pipelines.diagnose_overfit as overfit_pipeline
import landseg.session as session


# ----- `_prepare_dataspecs` helper
def test_prepare_dataspecs(tmp_path, dataspecs):
    '''
    Given: A RootConfig and an existing block in output directory.
    When: `_prepare_dataspecs` is called.
    Then: Return a valid DataSpecs instance with correct shape and meta.
    '''
    block_fpath = dataspecs.splits.train['train_block']
    dd = str(tmp_path / 'results' / 'overfit_test')
    os.makedirs(dd, exist_ok=True)
    shutil.copy(block_fpath, os.path.join(dd, 'test_block.npz'))

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = str(tmp_path)
    cfg = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(schema))

    logger = session.SessionLogger('test_overfit', log_file=f'{dd}/test.json')
    specs = overfit_pipeline._prepare_dataspecs(dd, cfg, logger)

    assert isinstance(specs, core.DataSpecs)
    assert specs.mode == 'default'
    assert specs.meta.image_specs.num_channels == 4
    assert specs.meta.image_specs.height_width == 256
    assert 'head_1' in specs.heads.class_counts
    assert specs.domains.ids_num == 0
    assert specs.domains.vec_dim == 0


# ----- `overfit` pipeline mock test
def test_overfit_pipeline_with_existing_block(tmp_path, dataspecs, monkeypatch):
    '''
    Given: A RootConfig and an existing block in output directory.
    When: `overfit` pipeline executes.
    Then: Successfully load existing block, set up model and run overfit.
    '''
    monkeypatch.setattr(overfit_pipeline.c, 'OVERFIT_MAX_EPOCH', 2)

    block_fpath = dataspecs.splits.train['train_block']
    exp_root = str(tmp_path / 'exp')
    overfit_dpath = f'{exp_root}/results/overfit_test'
    os.makedirs(overfit_dpath, exist_ok=True)
    shutil.copy(block_fpath, os.path.join(overfit_dpath, 'test_block.npz'))

    # compose config with OmegaConf
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = exp_root

    # override session & models for fast overfit test
    schema.session.orchestration.monitor.track_heads = {'head_1': 1.0}
    schema.session.dataloader.patch_size = 16
    schema.models.model_body_registry.unet.base_ch = 8

    cfg = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(schema))

    overfit_pipeline.overfit(cfg)

    summary_fpath = f'{overfit_dpath}/log/overfit_summary.json'
    assert os.path.exists(summary_fpath)
    with open(summary_fpath, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)

    assert summary_data['status'] == 'FAILED'
    assert summary_data['results']['overfit_reached'] is False
    assert summary_data['results']['final_epoch'] == 2
