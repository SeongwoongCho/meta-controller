import os
import sys
import shutil
import tqdm
import copy
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy

from train.trainer import LightningTrainWrapper
from datasets.dmc import DMC
from textwrap import dedent
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO
from pytorch_lightning.callbacks import ModelSummary 


def configure_experiment(config, model, is_rank_zero=True):
    if config.stage < 2:
        save_dir = 'checkpoints'
    else:
        save_dir = 'play_results'

    # set directories
    log_dir, save_dir = set_directories(config,
                                        exp_name=config.exp_name,
                                        exp_subname=(config.exp_subname if config.stage >= 1 else ''),
                                        save_dir=save_dir,
                                        is_rank_zero=is_rank_zero,
                                        )

    # create lightning callbacks, logger, and checkpoint plugin
    if config.stage != 2:
        callbacks = set_callbacks(config, save_dir, config.monitor, ptf=config.save_postfix)
    else:
        callbacks = set_callbacks(config, save_dir)
    logger = CustomTBLogger(log_dir, name='', version='', default_hp_metric=False)
    
    # create profiler
    profiler = pl.profilers.PyTorchProfiler(log_dir) if config.profile_mode else None
        
    # parse precision
    precision = int(config.precision.strip('fp')) if config.precision in ['fp16', 'fp32'] else config.precision
        
    # choose accelerator
    strategy = set_strategy(config, precision)

    # choose plugins
    if config.stage == 1 and config.strategy == 'ddp':
        plugins = [CustomCheckpointIO()]
    else:
        plugins = None
    
    return logger, log_dir, save_dir, callbacks, profiler, precision, strategy, plugins


def generate_general_print_info(config, model=None):
    if model is None:
        model_config = config
    else:
        model_config = model.config

    print_info = dedent(
        f'''\
        Running Stage {config.stage} with {config.strategy} Strategy:
        [General Info]
            > Exp Name: {config.exp_name}
            > Hidden Dim: {model_config.hidden_dim}
            > Num Morph Blocks: {model_config.n_morph_blocks}
            > Num Task Blocks: {model_config.n_task_blocks}
            > Num Attn Heads: {model_config.n_attn_heads}
            > Morphology LoRA Rank: {model_config.morphology_lora_rank}
            > Task LoRA Rank: {model_config.task_lora_rank}
            > History Size: {model_config.history_size}
            > Max Num Joints: {model_config.max_n_joints}
            > Seed: {config.seed}
        '''
    )
    
    print_info = print_info.replace('LINEBREAK\n', '')

    return print_info


def generate_mt_print_info(mt_config):
    print_info = dedent(
        f'''\
        [Meta-Train Info]
            > Global Batch Size: {mt_config.global_batch_size}
            > Num Steps: {mt_config.n_steps}
            > Validation Iterations: {mt_config.val_iter}
            > Optimizer: {mt_config.optimizer}
            > Weight Decay: {mt_config.weight_decay}
            > Learning Rate: {mt_config.lr}
            > Learning Rate Schedule: {mt_config.lr_schedule}
            > Learning Rate Warmup: {mt_config.lr_warmup}
        '''
    )

    print_info = print_info.replace('LINEBREAK\n', '')

    return print_info


def generate_ft_print_info(ft_config):
    print_info = dedent(
        f'''\
        [Fine-Tune Info]
            > Environment: {ft_config.env_name}
            > Num Support Demo: {ft_config.demo}
            > Load Step: {ft_config.load_step}
            > Global Batch Size: {ft_config.global_batch_size}
            > Num Steps: {ft_config.n_steps}
            > Validation Iterations: {ft_config.val_iter}
            > Optimizer: {ft_config.optimizer}
            > Weight Decay: {ft_config.weight_decay}
            > Learning Rate: {ft_config.lr}
            > Morphology LoRA Rank: {ft_config.morphology_lora_rank}
            > Task LoRA Rank: {ft_config.task_lora_rank}
            > Morphology LayerScale: {ft_config.morphology_layerscale}
            > Task LayerScale: {ft_config.task_layerscale}
            > Learning Rate Schedule: {ft_config.lr_schedule}
            > Learning Rate Warmup: {ft_config.lr_warmup}
        '''
    )
    return print_info.replace('LINEBREAK\n', '')


def generate_ts_print_info(ts_config):
    print_info = dedent(f'''\
        [Test Info]
            > Environment: {ts_config.env_name}
            > Load Step: {ts_config.load_step}
        '''
    )

    return print_info


def print_configs(config, model=None, mt_config=None, ft_config=None, ts_config=None):
    print_info = generate_general_print_info(config, model)
    if config.stage >= 0 and mt_config is not None:
        print_info += generate_mt_print_info(mt_config)
    if config.stage >= 1 and ft_config is not None:
        print_info += generate_ft_print_info(ft_config)
    if config.stage >= 2 and ts_config is not None:
        print_info += generate_ts_print_info(ts_config)
    print(print_info)


def set_directories(config, root_dir='experiments', exp_name='', log_dir='logs', save_dir='checkpoints',
                    create_log_dir=True, create_save_dir=True, dir_postfix='', exp_subname='', is_rank_zero=True):
    
    def _get_exp_subname(exp_subname, config):
        if exp_subname == '':
            lr = config.lr
            exp_subname = f'env:{config.env_name}_ntrajs:{config.num_train_trajs}_lr:{lr}'
            exp_subname += f'{config.subname_postfix}'

        return exp_subname
    
    # create the root directory
    os.makedirs(root_dir, exist_ok=True)

    # set saving directory
    if create_save_dir:
        save_root = os.path.join(root_dir, config.save_dir, exp_name + dir_postfix)
        if config.stage == 1:
            exp_subname = _get_exp_subname(exp_subname, config)
        save_root = os.path.join(save_root, exp_subname)
        os.makedirs(save_root, exist_ok=True)
        save_dir = os.path.join(save_root, save_dir)

        # create the saving directory if checkpoint doesn't exist,
        # otherwise ask user to reset it
        if is_rank_zero and os.path.exists(save_dir):
            if config.continue_mode:
                print(f'resume from checkpoint ({save_dir})')
            elif config.reset_mode:
                print(f'remove existing checkpoint ({save_dir})')
                try:
                    shutil.rmtree(save_dir)
                except:
                    pass
            elif config.stage == 2 and config.result_postfix != '':
                pass
            else:
                while True:
                    print(f'redundant experiment name! ({save_dir}) remove existing checkpoints? (y/n)')
                    inp = input()
                    if inp == 'y':
                        try:
                            shutil.rmtree(save_dir)
                        except:
                            pass
                        break
                    elif inp == 'n':
                        print('quit')
                        sys.exit()
                    else:
                        print('invalid input')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    # set logging directory
    if create_log_dir:
        os.makedirs(os.path.join(root_dir, config.log_dir), exist_ok=True)
        log_root = os.path.join(root_dir, config.log_dir, exp_name + dir_postfix)
        os.makedirs(log_root, exist_ok=True)
        if config.stage == 1:
            exp_subname = _get_exp_subname(exp_subname, config)

        log_root = os.path.join(log_root, exp_subname)
        os.makedirs(log_root, exist_ok=True)
        log_dir = os.path.join(log_root, log_dir)

        # reset the logging directory if exists
        if config.stage < 2 and is_rank_zero and os.path.exists(log_dir) and not config.continue_mode:
            try:
                shutil.rmtree(log_dir)
            except:
                pass
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = config.result_dir

    return log_dir, save_dir


def set_strategy(config, precision):
    if config.strategy == 'ddp':
        strategy = pl.strategies.DDPStrategy()
    elif config.strategy == 'deepspeed':
        strategy = pl.strategies.DeepSpeedStrategy(offload_optimizer=(config.optimizer == 'cpuadam'),
                                                   precision_plugin=pl.plugins.precision.DeepSpeedPrecisionPlugin(precision))
    else:
        strategy = None
        
    return strategy


def set_callbacks(config, save_dir, monitor=None, ptf=''):
    callbacks = [
        CustomProgressBar(),
        ModelSummary(max_depth=3)
    ]

    if save_dir is not None:
        # step checkpointing
        if config.stage == 0:
            checkpoint_callback = CustomModelCheckpoint(
                config=config,
                dirpath=save_dir,
                filename='step:{step:06d}' + ptf,
                auto_insert_metric_name=False,
                every_n_epochs=config.save_interval,
                save_top_k=-1,
                save_last=False,
            )
            checkpoint_callback.CHECKPOINT_JOIN_CHAR = "_"
            callbacks.append(checkpoint_callback)

        # last checkpointing
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_dir,
            filename=f'last{ptf}',
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_top_k=1,
            save_last=False,
            monitor='epoch',
            mode='max',
        )
        checkpoint_callback.CHECKPOINT_JOIN_CHAR = "_"
        callbacks.append(checkpoint_callback)
        
        # best checkpointing
        if not (config.no_eval or monitor is None):
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=save_dir,
                filename=f'best{ptf}',
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_top_k=1,
                save_last=False,
                monitor=monitor,
                mode=config.monitor_mode,
            )
            checkpoint_callback.CHECKPOINT_JOIN_CHAR = "_"
            callbacks.append(checkpoint_callback)
            
    return callbacks


def get_ckpt_path(load_dir, exp_name, load_step, exp_subname='', save_postfix='', reduced=False, load_path=None):
    if load_path is None or load_path == 'none':
        if load_step == 0:
            ckpt_name = f'best{save_postfix}.ckpt'
        elif load_step < 0:
            ckpt_name = f'last{save_postfix}.ckpt'
        else:
            ckpt_name = f'step:{load_step:06d}.ckpt'
        if reduced:
            ckpt_name = ckpt_name.replace('.ckpt', '.pth')
        
        load_path = os.path.join('experiments', load_dir, exp_name, exp_subname, 'checkpoints', ckpt_name)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"checkpoint ({load_path}) does not exist!")
    
    return load_path


def copy_values(config_new, config_old):
    for key in config_new.__dir__():
        if key[:2] != '__':
            setattr(config_old, key, getattr(config_new, key))


def load_trained_ckpt(ckpt_path, verbose=True):
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, 'checkpoint', 'mp_rank_00_model_states.pt')
        ckpt = torch.load(ckpt_path)
        state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt['module'].items()}
        config = ckpt['hyper_parameters']['config']
        global_step = ckpt['global_step']
    else:
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt['state_dict']
        config = ckpt['config']
        global_step = ckpt['global_step']
    if verbose:
        print(f'load checkpoint from {ckpt_path} at {global_step} step')

    return state_dict, config


def filter_state_dict(config, model, state_dict):
    # filter morphology-specific parameters
    morphology = config.env_name.split('-')[0]
    m_idx = DMC.MORPHOLOGIES.index(morphology)
    morph_param_names = model.model.morphology_specific_parameter_names()
   
    # filter task-specific parameters
    environment = tuple(config.env_name.split('-'))
    t_idx = DMC.ENVIRONMENTS.index(environment)
    task_param_names = model.model.task_specific_parameter_names()

    if config.stage == 1 and config.env_name.startswith('reacher_four'):
        m_idx = DMC.MORPHOLOGIES.index('reacher_three')

    for name in morph_param_names:
        # replace position embeddings
        if name.endswith('position_embedding.0'):
            v = state_dict[f'model.{name.replace("position_embedding.0", f"position_embedding.{m_idx}")}']
            
            # interpolate position embeddings using the mode
            if config.stage == 1 and config.env_name.startswith('reacher_four'):
                state_dict[f'model.{name}'] = torch.stack([v[:, 0], (v[:, 1] + v[:, 0]) / 2, (v[:, 1] + v[:, 2]) / 2, v[:, 2], v[:, 3]], dim=1)
            else:
                state_dict[f'model.{name}'] = v
        
        # replace global projections
        if 'proj_global.0' in name:
            v = state_dict[f'model.{name.replace("proj_global.0", f"proj_global.{m_idx}")}']
            for key in list(state_dict.keys()):
                if 'obs_embedding' in name and key in 'obs_embedding':
                    if name.endswith('weight'):
                        if 'proj_global.' in key and key.endswith('weight'):
                            del state_dict[key]
                    if name.endswith('bias'):
                        if 'proj_global.' in key and key.endswith('bias'):
                            del state_dict[key]
                if 'act_embedding' in name and key in 'act_embedding':
                    if name.endswith('weight'):
                        if 'proj_global.' in key and key.endswith('weight'):
                            del state_dict[key]
                    if name.endswith('bias'):
                        if 'proj_global.' in key and key.endswith('bias'):
                            del state_dict[key]
            state_dict[f'model.{name}'] = v

        # replace lora, layerscale, and bias parameters
        elif name.endswith('bias') or 'lora' in name or 'ls1' in name or 'ls2' in name:
            state_dict[f'model.{name}'] = state_dict[f'model.{name}'][m_idx:m_idx+1]
            if ('ls1' in name or 'ls2' in name) and config.morphology_layerscale is not None and config.morphology_layerscale != config.mt_config.morphology_layerscale:
                state_dict[f'model.{name}'] = torch.ones_like(state_dict[f'model.{name}']) * config.morphology_layerscale

    for name in task_param_names:
        if f'model.{name}' not in state_dict: 
            continue
        
        # replace positional embeddings
        if name.endswith('position_embedding.0'):
            v = state_dict[f'model.{name.replace("position_embedding.0", f"position_embedding.{t_idx}")}']

            for key in list(state_dict.keys()):
                if 'position_embedding.' in key:
                    del state_dict[key]
            state_dict[f'model.{name}'] = v
        
        # replace lora, layerscale, and bias parameters 
        elif name.endswith('bias') or 'lora' in name or 'ls1' in name or 'ls2' in name:
            if state_dict[f'model.{name}'].ndim == 1:
                state_dict[f'model.{name}'] = state_dict[f'model.{name}'].unsqueeze(0)
            else:
                state_dict[f'model.{name}'] = state_dict[f'model.{name}'][t_idx:t_idx+1]
            if ('ls1' in name or 'ls2' in name) and config.task_layerscale is not None and config.task_layerscale != config.mt_config.task_layerscale:
                state_dict[f'model.{name}'] = torch.ones_like(state_dict[f'model.{name}']) * config.task_layerscale

    # remove unnecessary keys
    for key in list(state_dict.keys()):
        if 'position_embedding.' in key and not key.endswith('position_embedding.0'):
            del state_dict[key]
        if 'proj_global.' in key and not 'proj_global.0' in key:
            del state_dict[key]

    # update lora parameters
    for key, value in model.state_dict().items():
        if 'lora' in key and key not in state_dict:
            state_dict.update({key: value})
    
    # adapt unseen morph lora rank
    if config.morphology_lora_rank != config.mt_config.morphology_lora_rank:
        for key in list(state_dict.keys()):
            if key.startswith("model.obs_morphology_transformer.obs_morphology_encoder.blocks") and (key.endswith('lora_A') or key.endswith('lora_B')):
                if config.morphology_lora_rank == 0:
                    state_dict.pop(key)
                else:
                    state_dict[key] = model.state_dict()[key]

    # adapt unseen task lora rank
    if config.task_lora_rank != config.mt_config.task_lora_rank:
        for key in list(state_dict.keys()):
            if  ("model.task_encoder" or key.startswith("model.state_task_transformer.task_encoder.blocks")) and (key.endswith('lora_A') or key.endswith('lora_B')):
                if config.task_lora_rank== 0:
                    state_dict.pop(key)
                else:
                    state_dict[key] = model.state_dict()[key]

def load_model(config, verbose=True):
    load_path = None
    mt_config = None
    ft_config = None
    ts_config = None
    # create trainer for episodic training
    if config.stage == 0:
        if config.continue_mode:
            load_path = get_ckpt_path(config.load_dir, config.exp_name, config.load_step, save_postfix=config.save_postfix, reduced=False)
            model = None

        model = LightningTrainWrapper(config, verbose=verbose)
        mt_config = config

    elif config.stage == 1:
        ft_config = copy.deepcopy(config)

        # load meta-trained checkpoint
        if config.continue_mode:
            lr = config.lr_task_specific if config.load_step >= -1 else config.lr_shared
            exp_subname = f'env:{config.env_name}_demo:{config.demo}_lr:{lr}{config.subname_postfix}'
            load_path = get_ckpt_path(config.save_dir, config.exp_name, -1, exp_subname, reduced=False)

        ckpt_path = get_ckpt_path(config.load_dir, config.exp_name, config.load_step, reduced=True)
        state_dict, mt_config = load_trained_ckpt(ckpt_path)
        config = copy.deepcopy(mt_config)
        copy_values(ft_config, config)

        # create model
        config.mt_config = mt_config
        model = LightningTrainWrapper(config=config, verbose=verbose)

        filter_state_dict(config, model, state_dict)

        print(model.load_state_dict(state_dict, strict=True))
        ft_config = config

    elif config.stage == 2:
        ts_config = copy.deepcopy(config)
        
        ft_ckpt_path = get_ckpt_path(config.load_dir, config.exp_name, config.load_step, config.exp_subname, reduced=True)
        ft_state_dict, ft_config = load_trained_ckpt(ft_ckpt_path, verbose=verbose)
        
        ts_config.env_name = ft_config.env_name
        ts_config.num_eval_trajs = ft_config.num_train_trajs 

        mt_config = ft_config.mt_config
        mt_ckpt_path = get_ckpt_path(mt_config.load_dir, mt_config.exp_name, mt_config.load_step, reduced=True)
        mt_state_dict, _ = load_trained_ckpt(mt_ckpt_path)
        
        config = copy.deepcopy(ft_config)
        copy_values(ts_config, config)
        model = LightningTrainWrapper(config=config, verbose=verbose)
        
        state_dict = {}
        filter_state_dict(config, model, mt_state_dict)
        state_dict.update(mt_state_dict)
        state_dict.update(ft_state_dict)
        
        print(model.load_state_dict(state_dict, strict=True))
    else:
        raise NotImplementedError()

    return model, config, load_path, mt_config, ft_config, ts_config

        
class CustomProgressBar(TQDMProgressBar):
    def __init__(self, rescale_validation_batches=1):
        super().__init__()
        self.rescale_validation_batches = rescale_validation_batches

    def init_train_tqdm(self):
        """Override this to customize the tqdm bar for training."""
        bar = tqdm.tqdm(
            desc="Training",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self):
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        bar = tqdm.tqdm(
            desc="Validation",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar
    
    def init_test_tqdm(self):
        """Override this to customize the tqdm bar for testing."""
        bar = tqdm.tqdm(
            desc="Testing",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    
class CustomTBLogger(TensorBoardLogger):
    @pl.utilities.rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)


class CustomModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.step_size = 1

    def _monitor_candidates(self, trainer):
        monitor_candidates = deepcopy(trainer.callback_metrics)
        # cast to int if necessary because `self.log("epoch", 123)` will convert it to float. if it's not a tensor
        # or does not exist we overwrite it as it's likely an error
        epoch = monitor_candidates.get("epoch")
        monitor_candidates["epoch"] = epoch.int() if isinstance(epoch, torch.Tensor) else torch.tensor(trainer.current_epoch)
        step = monitor_candidates.get("step")
        step = step // self.step_size
        monitor_candidates["step"] = step.int() if isinstance(step, torch.Tensor) else torch.tensor(trainer.global_step)
        return monitor_candidates

    
class CustomCheckpointIO(TorchCheckpointIO):
    def __init__(self, save_parameter_names=None):
        self.save_parameter_names = save_parameter_names
    
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        # store only task-specific parameters
        if self.save_parameter_names is not None:
            state_dict = checkpoint['state_dict']
            state_dict = {key: value for key, value in state_dict.items() if key in self.save_parameter_names}
            checkpoint['state_dict'] = state_dict

        super().save_checkpoint(checkpoint, path, storage_options)

