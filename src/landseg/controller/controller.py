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

'''Curriculum controller class.'''

# standard imports
import dataclasses
import os
import typing
# local imports
import landseg.controller as controller
import landseg.training.trainer as trainer
import landseg.utils as utils

# --------------------------------private  type--------------------------------
class _CheckpointMeta(typing.TypedDict):
    '''Checkpont metadata'''
    metric: float
    epoch: int
    step: int

# --------------------------------Public  Class--------------------------------
class Controller:
    '''doc'''
    def __init__(
        self,
        engine: trainer.MultiHeadTrainer,
        phases: list[controller.Phase],
        exp_dir: str,
        logger: utils.Logger
    ):
        '''Initialization'''

        # parse arguments
        self.trainer = engine
        self.phases = phases
        self.logger = logger.get_child('phase') # a child from base logger
        # preview and checkpoint dir
        self.previews = os.path.join(exp_dir, 'previews')
        self.ckpts = os.path.join(exp_dir, 'checkpoints')

        # init phase counter
        self.current_phase_idx = 0
        self.current_phase = self.phases[self.current_phase_idx]

        # check which phases have already finished
        self.phase_status_path = f'{self.ckpts}/status.json'
        if os.path.exists(self.phase_status_path):
            scheme = utils.load_json(self.phase_status_path)
            for i, p in enumerate(scheme):
                if controller.Phase(**p).finished:
                    self.phases[i].finished = True
        else:
            self._record_progress() # write phase status.json

    def fit(
        self,
        stopat: str | None=None
    ) -> None:
        '''Main entry.'''

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
            if isinstance(stopat, str) and stopat == phase.name:
                print('__Experiment Complete__')
                self.logger.log('INFO', f'Training stopped@{stopat}')
                break

    def _record_progress(self):
        '''doc'''

        scheme = [dataclasses.asdict(p) for p in self.phases]
        utils.write_json(self.phase_status_path, scheme) # overwrite

    def _train_phase(self, meta) -> tuple[dict, dict]:
        '''Train the current phase.'''

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

        # align trainer config with phase specs
        self.trainer.config.schedule.max_epoch = num_epoch

        # set trainer logit adjustments
        la_scheme = phase.la_scheme
        self.trainer.model.set_logit_adjust_alpha(la_scheme.logit_adjust_alpha)
        self.trainer.config_logit_adjustment(**dataclasses.asdict(la_scheme))

        # train
        print(f'__Phase [{phase.name}] started__')
        for epoch in range(start, num_epoch + 1):
            # early stop check
            # - patience can be None = no early stop
            # - stop when patience reached
            # - first 10 epochs not affected
            patience = self.trainer.config.schedule.patience_epochs
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
            if self.trainer.config.schedule.eval_interval is not None and \
                epoch % self.trainer.config.schedule.eval_interval == 0:
                v_logs = self.trainer.validate()
                # update preview if test data provided
                if self.trainer.dataloaders.test:
                    self.trainer.infer(self.previews)
            # save progress
            if epoch == self.trainer.state.metrics.best_epoch:
                fpath = f'{self.ckpts}/{phase.name}_best.pt'
            else:
                fpath = f'{self.ckpts}/{phase.name}_last.pt'
            self._save_progress(fpath)
        print(f'__Phase [{phase.name}] finished__')

        # return training and validation logs
        return t_logs, v_logs

    def _next_phase(self) -> None:
        '''Move on to the next phase.'''

        # advance phase idx
        self.current_phase_idx += 1
        # if already done stop
        if self.done:
            return
        # update current active phase
        self.current_phase = self.phases[self.current_phase_idx]
        #  reset trainer state and continue
        self.trainer.reset_head_state()

    def _load_progress(self, phase: str):
        '''Load previous best checkpoint'''

        best_ckpt = f'{self.ckpts}/{phase}_best.pt'
        if os.path.exists(best_ckpt):
            meta = trainer.load(
                model=self.trainer.model,
                optimizer=self.trainer.optimization.optimizer,
                scheduler=self.trainer.optimization.scheduler,
                fpath=best_ckpt,
                device='cuda')
            return meta
        return None

    def _save_progress(self, fpath: str) -> None:
        '''Save at the current phase.'''

        ckpt_meta: _CheckpointMeta = {
            'metric': self.trainer.state.metrics.curr_value,
            'epoch': self.trainer.state.progress.epoch,
            'step': self.trainer.state.progress.global_step
        }
        trainer.save(
            model=self.trainer.model,
            ckpt_meta=ckpt_meta,
            optimizer=self.trainer.comps.optimization.optimizer,
            scheduler=self.trainer.comps.optimization.scheduler,
            fpath=fpath
        )
        self.logger.log('INFO', f'Checkpoint saved: {fpath}')

    @property
    def done(self) -> bool:
        '''Returns whether controller has reached the final phase.'''
        return self.current_phase_idx >= len(self.phases)
