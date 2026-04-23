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

# pylint: disable=missing-function-docstring

'''
Curriculum-based training runner for multi-phase model execution.

The Runner orchestrates sequential training phases over a shared trainer
engine. Each phase can configure active heads, logit adjustments, and
training schedules, and supports checkpointing, resumption, validation,
and early stopping.

Phase progress is persisted to disk so experiments can be resumed safely
across interruptions.
'''

# standard imports
import dataclasses
import os
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.session.engine as engine
import landseg.session.orchestration.training as training
import landseg.utils as utils

@dataclasses.dataclass
class RunnerConfig:
    '''Runner configuration.'''
    paths: artifacts.ResultsPaths
    mode: typing.Literal['epochs', 'phases'] = 'epochs'
    verbose: bool = True
    heads: training.HeadsConfigLike | None = None
    max_epochs: int | None = None
    patience_epoch: int | None = None
    phases: typing.Sequence[training.TrainingPhaseLike] | None = None

# --------------------------------Public  Class--------------------------------
class TrainingRunner:
    '''
    Orchestrates multi-phase curriculum training for a model engine.

    The Runner executes a sequence of training phases, where each phase
    can configure model heads, training behavior, and logit adjustments.
    It supports checkpoint resumption, validation scheduling, inference
    previews, and persistent phase tracking across runs.
    '''
    def __init__(
        self,
        trainer: engine.MultiHeadTrainer,
        evaluator: engine.MultiHeadEvaluator,
        config: RunnerConfig,
        *,
        logger: utils.Logger,
        **kwargs
    ):
        '''
        Initialize the curriculum runner.

        Behavior:
            - Initializes checkpoint and preview directories.
            - Restores phase completion state from disk if available.
            - Sets the first active phase.

        Args:
            engine:
                Multi-head trainer instance used for all phases.
            phases:
                Ordered list of training phases defining curriculum flow.
            exp_dir:
                Experiment root directory for checkpoints and previews.
            logger:
                Logger instance used for phase-level logging.
        '''

        # parse arguments
        self.trainer = trainer
        self.evaluator = evaluator
        self.config = config
        self.logger = logger.get_child('phase') # a child from base logger

        # set la status
        self.trainer.model.set_logit_adjust_alpha(
            kwargs.get('logit_adjust_alpha', 1.0)
        )
        self.trainer.config_logit_adjust(
            enable_train_logit_adjust=kwargs.get('train_logit_adjust', True)
        )
        self.evaluator.model.set_logit_adjust_alpha(
            kwargs.get('logit_adjust_alpha', 1.0)
        )
        self.evaluator.config_logit_adjust(
            enable_val_logit_adjust=kwargs.get('val_logit_adjustment', True),
            enable_test_logit_adjust=kwargs.get('test_logit_adjustment', False)
        )

        # track the final checkpoint
        self._end_checkpoint: str | None = None

    @property
    def final_checkpoint(self) -> str | None:
        '''Return the final checkpoint of a training session.'''
        return self._end_checkpoint

    def fit(self) -> None:
        '''
        Execute full curriculum training across all phases.

        Iterates through phases in order, skipping completed ones,
        resuming from checkpoints when available, and saving progress
        after each phase.

        Behavior:
            - Loads checkpoint state per phase if present.
            - Runs training loop for each active phase.
            - Performs validation and optional inference previews.
            - Persists phase completion status after each phase.
            - Stops early when all phases are complete.
        '''

        # mode matching
        match self.config.mode:
            # train by epochs
            case 'epochs': self._train_epochs()
            # train by phases
            case 'phases': self._train_phases()

    def _train_epochs(self) -> tuple[dict, dict]:
        '''doc'''

        assert self.config.max_epochs, 'Max epoch not provided'

        # init train and validation logs
        t_logs = {}
        v_logs = {}

        # try load progress
        self.logger.log('INFO', 'Try to load from previous checkpoint')
        start = self._load_progress('checkpoint')
        self.logger.log('INFO', f'Start/resume from epoch {start}')

        # set head state
        assert self.config.heads, 'Head config not provided'
        self.trainer.set_head_state(
            self.config.heads.active_heads,
            self.config.heads.frozen_heads,
            self.config.heads.excluded_cls
        )

        # start training
        n_epochs = self.config.max_epochs
        self.logger.log('INFO', f'Train with max {n_epochs} epochs')
        for epoch in range(start, n_epochs + 1):
            self.logger.log('INFO', f'__Epoch: {epoch}/{n_epochs}__')
            # early stop check
            # - patience can be None = no early stop
            # - warm up epochs: 5
            patience = self.config.patience_epoch
            patience_counter = self.trainer.state.metrics.patience_n
            if patience and patience_counter >= patience and epoch >= 5:
                self.logger.log('INFO', 'Patience limit reached, stopping')
                break
            # refresh logs per epoch
            t_logs, v_logs = self._run_one_epoch(
                epoch=epoch,
                active_heads=self.trainer.state.heads.active_heads,
                checkpoint_name_tag='checkpoint' # generic name tag
            )

        # return training and validation logs
        self.logger.log('INFO', '__Experiment Complete__')
        return t_logs, v_logs

    def _train_phases(self) -> tuple[dict, dict]:
        '''Run training loop for a single phase and return logs.'''

        # init train and validation logs
        t_logs = {}
        v_logs = {}

        assert self.config.phases, 'Phases not provided'
        for phase in self.config.phases:
            # log phase start with optional print
            self.logger.log('INFO', f'__Phase [{phase.name}] started__')
            if self.config.verbose:
                self._print_phase(phase)

            # try load progress
            self.logger.log('INFO', 'Try to load from previous checkpoint')
            start = self._load_progress(phase.name)
            if start == phase.num_epochs + 1:
                self.logger.log('INFO', f'Phase {phase.name} finished, skip')
                continue
            self.logger.log('INFO', f'Start/resume from epoch {start}')

            # set head state per phase
            self.trainer.set_head_state(
                phase.heads.active_heads,
                phase.heads.frozen_heads,
                phase.heads.excluded_cls
            )

            # run current phase
            for epoch in range(start, phase.num_epochs + 1):
                self.logger.log('INFO', f'__Epoch: {epoch}/{phase.num_epochs}__')
                # refresh logs per epoch
                t_logs, v_logs = self._run_one_epoch(
                    epoch=epoch,
                    active_heads=self.trainer.state.heads.active_heads,
                    checkpoint_name_tag=phase.name
                )

            #  reset trainer state and continue
            self.trainer.reset_head_state()
            self.logger.log('INFO', f'__Phase [{phase.name}] finished__')

        # return training and validation logs
        self.logger.log('INFO', '__Experiment Complete__')
        self.logger.log('INFO', 'All training phases finished')
        return t_logs, v_logs

    def _run_one_epoch(
        self,
        *,
        epoch: int,
        active_heads: list[str] | None,
        checkpoint_name_tag: str
    ) -> tuple[dict, dict]:
        '''doc'''

        # init
        t_logs = {}
        v_logs = {}

        # train the current epoch
        t_logs = self.trainer.train_one_epoch(epoch)
        # validate each epoch
        v_logs = self.evaluator.validate()
        # update preview if test data provided
        if self.trainer.dataloaders.test:
            self.evaluator.infer(
                self.config.paths.previews,
                preview_heads=active_heads
            )
        # save progress
        name = checkpoint_name_tag
        if epoch == self.trainer.state.metrics.best_epoch:
            self._save_progress(self.config.paths.best_checkpoint(name))
            self._end_checkpoint = self.config.paths.best_checkpoint(name)
        else:
            self._save_progress(self.config.paths.last_checkpoint(name))

        # return training and validation logs
        return t_logs, v_logs

    def _load_progress(self, name: str) -> int:
        '''Load checkpoint metadata for a given phase if available.'''

        last_ckpt = self.config.paths.last_checkpoint(name)
        best_ckpt = self.config.paths.best_checkpoint(name)
        if os.path.exists(last_ckpt):
            fp = last_ckpt
        elif os.path.exists(best_ckpt):
            fp = best_ckpt
        else:
            return 1

        # load and return epoch number
        meta = artifacts.load_checkpoint(
            model=self.trainer.model,
            fpath=fp,
            map_device=self.trainer.device,
            optimizer=self.trainer.optimization.optimizer,
            scheduler=self.trainer.optimization.scheduler,
        )
        self.trainer.state.progress.epoch = meta['epoch'] + 1
        self.trainer.state.progress.global_step = meta['step']
        self.trainer.state.metrics.best_value = meta['metric']
        return meta['epoch']

    def _save_progress(self, fpath: str) -> None:
        '''Save model checkpoint and training metadata to disk.'''

        ckpt_meta: artifacts.CheckpointMeta = {
            'metric': self.trainer.state.metrics.curr_value,
            'epoch': self.trainer.state.progress.epoch,
            'step': self.trainer.state.progress.global_step
        }
        artifacts.save_checkpoint(
            model=self.trainer.model,
            fpath=fpath,
            ckpt_meta=ckpt_meta,
            optimizer=self.trainer.comps.optimization.optimizer,
            scheduler=self.trainer.comps.optimization.scheduler,
        )
        self.logger.log('INFO', f'Checkpoint saved: {fpath}')

    @staticmethod
    def _print_phase(phase: training.TrainingPhaseLike):
        '''Pretty print a phase to console.'''

        print('__Phase details__')
        heads = phase.heads
        ss = '\n'.join([
            f'- Phase Name:\t{phase.name}',
            f'- Max Epochs:\t{phase.num_epochs}',
            f'- LR Scale:\t{phase.lr_scale}',
            '- Heads Specs',
            f'  - Active Heads:\t{heads.active_heads}',
            f'  - Frozen Heads:\t{heads.frozen_heads}',
            f'  - Excld. Class:\t{heads.excluded_cls}',
        ])
        print(ss)
