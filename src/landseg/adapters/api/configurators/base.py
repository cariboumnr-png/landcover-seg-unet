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

'''
Configurator base class
'''

# standard imports
import pathlib
import typing
# local imports
import landseg.adapters.api.api as api
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.utils.validator as val

class BaseConfigurator:
    '''Base configurator for execution commands.'''

    def __init__(
        self,
        command: str,
        experiment_root: str | pathlib.Path | None = None,
        *,
        output_dpath: str | pathlib.Path | None = None,
    ):
        '''
        Initialize the configurator.

        Supports both:
        1. Canonical decoupled mode: Explicitly passing destination output and
           upstream input directories. No experiment root is inferred or
           required, so stages may live on unrelated storage volumes.
        2. Auxiliary unified mode: Passing an optional `experiment_root` that
           auto-populates all stage inputs and outputs under
           `<exp_root>/artifacts/...` and `<exp_root>/results`.
        '''
        # init a default RootConfig instance
        self.cfg = configs.RootConfig()
        self.cfg.command = command

        if experiment_root is None:
            self.output = self.require_dir(output_dpath, writable=True)
        else:
            resolved_root = self.require_dir(experiment_root, writable=True)
            self.cfg.execution.exp_root = resolved_root

            _paths = artifacts.ArtifactPaths(root=pathlib.Path(resolved_root))
            dt_cfg = self.cfg.data
            dt_cfg.world_grid.output_dpath = str(_paths.world_grid.root)
            dt_cfg.harmonization.output_dpath = str(_paths.data_harmonization.root)
            dt_cfg.ingestion.output_dpath = str(_paths.data_ingestion.root)
            dt_cfg.preparation.output_dpath = str(_paths.data_preparation.root)
            self.cfg.session.output_dpath = str(_paths.session.root)


    def set_output_dpath(self, output_dpath: str | pathlib.Path) -> typing.Self:
        '''Set destination output directory path for the active command.'''
        dpath_str = str(output_dpath)
        self.output = dpath_str
        match self.cfg.command:
            case 'world-grid':
                self.cfg.data.world_grid.output_dpath = dpath_str
            case 'data-harmonize':
                self.cfg.data.harmonization.output_dpath = dpath_str
            case 'data-ingest':
                self.cfg.data.ingestion.output_dpath = dpath_str
            case 'data-prepare':
                self.cfg.data.preparation.output_dpath = dpath_str
            case (
                'model-train' |
                'study-sweep' |
                'diagnose-overfit' |
                'model-evaluate'
            ):
                self.cfg.session.output_dpath = dpath_str
        return self

    def set_upstream_dpath(
        self, upstream_dpath: str | pathlib.Path
    ) -> typing.Self:
        '''Set direct upstream input directory path for the active command.'''
        dpath_str = str(upstream_dpath)
        match self.cfg.command:
            case 'data-harmonize':
                self.cfg.data.world_grid.output_dpath = dpath_str
            case 'data-ingest':
                self.cfg.data.harmonization.output_dpath = dpath_str
            case 'data-prepare':
                self.cfg.data.ingestion.output_dpath = dpath_str
            case (
                'model-train' |
                'study-sweep' |
                'diagnose-overfit' |
                'model-evaluate'
            ):
                self.cfg.data.preparation.output_dpath = dpath_str
        return self

    @property
    def running_root_config(self) -> configs.RootConfig:
        '''Validate respective section of the `RootConfig` and return.'''
        match self.cfg.command:
            case 'world-grid':
                self.cfg.data.world_grid.validate()
            case 'data-harmonize':
                self.cfg.data.harmonization.validate()
            case 'data-ingest':
                self.cfg.data.ingestion.validate()
            case 'data-prepare':
                self.cfg.data.preparation.validate()
            case 'model-evaluate':
                self.cfg.models.validate()
                self.cfg.session.validate()
            case (
                'model-train' |
                'diagnose-overfit' |
                'study-sweep' |
                'study-analysis'
            ):
                self.cfg.models.validate()
                self.cfg.session.validate()
        return self.cfg

    def preflight(self, strict: bool = False) -> typing.Any:
        '''Run pre-flight diagnostic validation for configured command.'''
        return api.run_preflight(self.cfg, self.cfg.command, strict=strict)

    # ----- shared methods for configuring runtime sessions
    def set_data_loading(
        self,
        batch_size: int,
        patch_size: int
    ) -> typing.Self:
        '''Set data sizes.'''
        self.cfg.session.dataloader.batch_size = batch_size
        self.cfg.session.dataloader.patch_size = patch_size
        return self

    def set_domain_source(
        self,
        category_domain: str | None,
        continuous_domain: str | None,
    ) -> typing.Self:
        '''Set data source'''
        self.cfg.data.specification.domain_ids_name = category_domain
        self.cfg.data.specification.domain_vec_name = continuous_domain
        return self

    def set_tasks(
        self,
        logit_adjust_alpha: float,
        exclude_classes: dict[str, list[int]] | None,
        loss_weights: dict[str, float],
        head_weights: dict[str, float] | None = None,
    ) -> typing.Self:
        '''Set engine tasks.'''
        self.cfg.session.engine_exec.logit_adjust_alpha = logit_adjust_alpha
        self.cfg.session.engine_tasks.excluded_cls = exclude_classes
        self.cfg.session.engine_tasks.head_weights = head_weights
        losses = self.cfg.session.engine_tasks.loss_configs
        losses.focal.weight = loss_weights.get('focal', 0.5)
        losses.dice.weight = loss_weights.get('dice', 0.5)
        losses.spectral.weight = loss_weights.get('spectral', 1e-3)
        losses.tv.weight = loss_weights.get('total_var', 1e-4)
        losses.ecological.weight = loss_weights.get('ecological', 0.0)
        return self

    def set_runtime(
        self,
        max_epochs: int,
        active_heads: list[str] | None,
        patience_epoch: int | None,
        track_heads: dict[str, float] | None,
    ) -> typing.Self:
        '''Set training runtime behaviour.'''
        orchestration = self.cfg.session.orchestration
        orchestration.curriculum.single.phases[0].num_epochs = max_epochs
        orchestration.curriculum.single.phases[0].active_heads = active_heads
        orchestration.monitor.patience = patience_epoch
        orchestration.monitor.track_heads = track_heads
        return self

    @staticmethod
    def require_dir(
        path: str | pathlib.Path | None,
        *,
        writable: bool = False,
    ) -> str:
        '''Require path to a directory (optionally writable).'''
        return str(val.require_dir(path, writable=writable))
