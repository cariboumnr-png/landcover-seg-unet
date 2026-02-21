'''Curriculum controller class.'''

# standard imports
import dataclasses
import os
# local imports
import alias
import training.common
import training.controller
import training.trainer
import utils

# --------------------------------Public  Class--------------------------------
class Controller:
    '''doc'''
    def __init__(
            self,
            trainer: training.trainer.MultiHeadTrainer,
            phases: list[training.controller.Phase],
            config: alias.ConfigType,
            logger: utils.Logger
        ):
        '''Initialization'''

        # parse arguments
        self.trainer = trainer
        self.phases = phases
        self.logger = logger.get_child('phase') # a child from base logger
        # config accessor
        cfg = utils.ConfigAccess(config)
        self.cfg = _ExperimentConfig(
            cfg.get_option('ckpt_dpath'),
            cfg.get_option('preview_dpath')
        )

        # init phase counter
        self.current_phase_idx = 0
        self.current_phase = self.phases[self.current_phase_idx]

    def fit(self, stopat: str | int| None=None) -> None:
        '''Main entry.'''


        for phase in self.phases:

            print('__Phase details__')
            print(phase)

            meta = self._load_progress(phase.name)
            self._train_phase(meta)
            self._next_phase()

            if self.done:
                print('__Experiment Complete__')
                self.logger.log('INFO', 'All training phases finished')
                break
            if isinstance(stopat, str) and stopat == phase.name:
                print('__Experiment Complete__')
                self.logger.log('INFO', f'Training stopped@{stopat}')
                break
            if isinstance(stopat, int) and stopat == self.current_phase_idx:
                print('__Experiment Complete__')
                self.logger.log('INFO', f'Training stopped@Phase_{stopat + 1}')
                break

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
                active_heads=phase.active_heads,
                frozen_heads=phase.frozen_heads,
                excluded_cls=phase.excluded_cls
            )
            t_logs = self.trainer.train_one_epoch(epoch)
            # validate at set interval
            if self.trainer.config.schedule.eval_interval is not None and \
                epoch % self.trainer.config.schedule.eval_interval == 0:
                v_logs = self.trainer.validate()
                # also do a preview if inference data provided TODO
                if self.trainer.dataloaders.test:
                    self.trainer.infer(self.cfg.preview_dpath)
            # save progress
            if epoch == self.trainer.state.metrics.best_epoch:
                fpath = f'{self.cfg.ckpt_dpath}/{phase.name}_best.pt'
            else:
                fpath = f'{self.cfg.ckpt_dpath}/{phase.name}_last.pt'
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
        # else reset trainer state and continue
        self.trainer.reset_head_state()

    def _load_progress(self, phase: str):
        '''Load previous best checkpoint'''

        best_ckpt = f'{self.cfg.ckpt_dpath}/{phase}_best.pt'
        if os.path.exists(best_ckpt):
            meta = training.trainer.load(
                model=self.trainer.model,
                optimizer=self.trainer.optimization.optimizer,
                scheduler=self.trainer.optimization.scheduler,
                fpath=best_ckpt,
                device='cuda')
            return meta
        return None

    def _save_progress(self, fpath: str) -> None:
        '''Save at the current phase.'''

        ckpt_meta: training.common.CheckpointMetaLike = {
            'metric': self.trainer.state.metrics.curr_value,
            'epoch': self.trainer.state.progress.epoch,
            'step': self.trainer.state.progress.global_step
        }
        training.trainer.save(
            model=self.trainer.comps.model,
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

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _ExperimentConfig:
    '''Experiment configs.'''
    ckpt_dpath: str
    preview_dpath: str
