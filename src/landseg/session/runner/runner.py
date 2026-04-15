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
Curriculum-based training controller for multi-phase model execution.

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
import landseg.session.runner as runner
import landseg.utils as utils

class _RunnerScheduleShape(typing.Protocol):
    '''doc'''
    @property
    def patience(self) -> int: ...
    @property
    def val_every(self) -> int: ...

# --------------------------------Public  Class--------------------------------
class Runner:
    '''
    Orchestrates multi-phase curriculum training for a model engine.

    The Runner executes a sequence of training phases, where each phase
    can configure model heads, training behavior, and logit adjustments.
    It supports checkpoint resumption, validation scheduling, inference
    previews, and persistent phase tracking across runs.
    '''
    def __init__(
        self,
        *,
        trainer: engine.MultiHeadTrainer,
        evaluator: engine.MultiHeadEvaluator,
        schedule: _RunnerScheduleShape,
        phases: list[runner.Phase],
        run_paths: artifacts.ResultsPaths,
        logger: utils.Logger,
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
        self.schedule = schedule
        self.phases = phases
        self.paths = run_paths
        self.logger = logger.get_child('phase') # a child from base logger

        # init phase counter
        self.current_phase_idx = 0
        self.current_phase = self.phases[self.current_phase_idx]

        # check which phases have already finished
        c = artifacts.Controller[list[dict]].load_json_or_fail
        self.progress = c(self.paths.phase_status)
        try:
            status = self.progress.fetch()
            assert status # typing assertion
            for i, p in enumerate(status):
                if runner.Phase(**p).finished:
                    self.phases[i].finished = True
        except artifacts.ArtifactError:
            self._record_progress() # write phase status.json

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

        # iterate from the starting phase (default to first phase)
        for phase in self.phases:
            # skip already finished phases
            if phase.finished:
                self._next_phase() # progress
                continue
            print('__Phase details__')
            print(phase)
            # try load from previous checkpoint
            meta = self._load_progress(phase.name)
            # main execution
            self._train_phase(meta)
            # update phase sheme
            self.current_phase.finished = True
            self._record_progress()
            # advance
            self._next_phase()

            # check completion status
            if self.done:
                print('__Experiment Complete__')
                self.logger.log('INFO', 'All training phases finished')
                break

    def _record_progress(self):
        '''Persist current phase completion status to disk.'''
        status = [dataclasses.asdict(p) for p in self.phases]
        self.progress.persist(status)

    def _train_phase(self, meta) -> tuple[dict, dict]:
        '''Run training loop for a single phase and return logs.'''

        # if loaded from previous
        if meta:
            self.logger.log('INFO', 'Loading from previous checkpoint')
            self.trainer.state.progress.epoch = meta['epoch'] + 1
            self.trainer.state.progress.global_step = meta['step']
            self.trainer.state.metrics.best_value = meta['metric']
            start = meta['epoch'] + 1
        else:
            start = 1

        # context for current phase
        phase = self.current_phase
        num_epoch = self.current_phase.num_epochs
        t_logs = {}
        v_logs = {}

        # set trainer logit adjustments
        la_scheme = phase.logit_adjust
        self.trainer.model.set_logit_adjust_alpha(la_scheme.logit_adjust_alpha)
        self.trainer.config_logit_adjustment(
            enable_train_logit_adjustment=la_scheme.enable_train_logit_adjustment,
            enable_val_logit_adjustment=la_scheme.enable_val_logit_adjustment,
            enable_test_logit_adjustment=la_scheme.enable_test_logit_adjustment,
        )

        # train
        print(f'__Phase [{phase.name}] started__')
        for epoch in range(start, num_epoch + 1):
            # early stop check
            # - patience can be None = no early stop
            # - stop when patience reached
            # - first 10 epochs not affected
            patience = self.schedule.patience
            patience_counter = self.trainer.state.metrics.patience_n
            if patience and patience_counter >= patience and epoch >= 10:
                self.logger.log('INFO', 'Patience limit reached, stopping')
                break
            print(f'__Epoch: {epoch}/{num_epoch}__')
            # set trainer heads
            self.trainer.set_head_state(
                phase.heads.active_heads,
                phase.heads.frozen_heads,
                phase.heads.excluded_cls
            )
            # train the current epoch
            t_logs = self.trainer.train_one_epoch(epoch)
            # validate at set interval
            if self.schedule.val_every is not None and \
                epoch % self.schedule.val_every == 0:
                v_logs = self.evaluator.validate()
                # update preview if test data provided
                if self.trainer.dataloaders.test:
                    self.evaluator.infer(self.paths.previews)
            # save progress
            if epoch == self.trainer.state.metrics.best_epoch:
                self._save_progress(self.paths.best_checkpoint(phase.name))
            else:
                self._save_progress(self.paths.last_checkpoint(phase.name))
        print(f'__Phase [{phase.name}] finished__')

        # return training and validation logs
        return t_logs, v_logs

    def _next_phase(self) -> None:
        '''Advance to the next training phase and reset trainer state.'''

        # advance phase idx
        self.current_phase_idx += 1
        # if already done stop
        if self.done:
            return
        # update current active phase
        self.current_phase = self.phases[self.current_phase_idx]
        #  reset trainer state and continue
        self.trainer.reset_head_state()

    def _load_progress(self, phase_name: str):
        '''Load checkpoint metadata for a given phase if available.'''

        best_ckpt = self.paths.best_checkpoint(phase_name)
        if os.path.exists(best_ckpt):
            meta = artifacts.load_checkpoint(
                model=self.trainer.model,
                optimizer=self.trainer.optimization.optimizer,
                scheduler=self.trainer.optimization.scheduler,
                fpath=best_ckpt,
                device='cuda')
            return meta
        return None

    def _save_progress(self, fpath: str) -> None:
        '''Save model checkpoint and training metadata to disk.'''

        ckpt_meta: artifacts.CheckpointMeta = {
            'metric': self.trainer.state.metrics.curr_value,
            'epoch': self.trainer.state.progress.epoch,
            'step': self.trainer.state.progress.global_step
        }
        artifacts.save_checkpoint(
            model=self.trainer.model,
            ckpt_meta=ckpt_meta,
            optimizer=self.trainer.comps.optimization.optimizer,
            scheduler=self.trainer.comps.optimization.scheduler,
            fpath=fpath
        )
        self.logger.log('INFO', f'Checkpoint saved: {fpath}')

    @property
    def done(self) -> bool:
        '''Return whether all curriculum phases have completed.'''
        return self.current_phase_idx >= len(self.phases)
