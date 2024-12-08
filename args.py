import argparse
import yaml
from easydict import EasyDict


def str2bool(v):
    if v == 'True' or v == 'true':
        return True
    elif v == 'False' or v == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

# argument parser
def parse_args(shell_script=None):
    parser = argparse.ArgumentParser()

    # necessary arguments
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--continue_mode', '-cont', default=False, action='store_true')
    parser.add_argument('--no_eval', '-ne', default=False, action='store_true')
    
    parser.add_argument('--reset_mode', '-reset', default=False, action='store_true')
    parser.add_argument('--profile_mode', '-prof', default=False, action='store_true')
    
    parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--exp_subname', type=str, default='')
    parser.add_argument('--name_postfix', '-ptf', type=str, default='')
    parser.add_argument('--subname_postfix', '-snptf', type=str, default='')
    parser.add_argument('--save_postfix', '-sptf', type=str, default='')
    parser.add_argument('--result_postfix', '-rptf', type=str, default='')
    parser.add_argument('--result_dir', '-rdir', type=str, default='')

    # optional arguments
    parser.add_argument('--env_name', type=str, default=None)
    
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--strategy', '-str', type=str, default=None)
    parser.add_argument('--precision', '-prec', type=str, default=None)

    parser.add_argument('--num_workers', '-nw', type=int, default=None)
    parser.add_argument('--global_batch_size', '-gbs', type=int, default=None)
    
    parser.add_argument('--num_train_trajs', type=int, default=None)
    parser.add_argument('--num_eval_trajs', type=int, default=None)
    parser.add_argument('--num_eval_episodes', '-nee', type=int, default=None)
    
    parser.add_argument('--history_size', '-hsize', type=int, default=None)
    parser.add_argument('--max_n_joints', type=int, default=None)
    parser.add_argument('--window_size', '-ws', type=int, default=None)
    parser.add_argument('--drop_rate', '-dr', type=int, default=None)

    parser.add_argument('--hidden_dim', '-hd', type=int, default=None)
    parser.add_argument('--n_attn_heads', '-nah', type=int, default=None)
    parser.add_argument('--n_morph_blocks', '-nmb', type=int, default=None)
    parser.add_argument('--n_task_blocks', '-ntb', type=int, default=None)
    parser.add_argument('--morphology_lora_rank', '-mlora', type=int, default=None)
    parser.add_argument('--task_lora_rank', '-tlora', type=int, default=None)
    parser.add_argument('--morphology_layerscale', '-morls', type=float, default=None)
    parser.add_argument('--task_layerscale', '-taskls', type=float, default=None)
    parser.add_argument('--n_steps', '-nst', type=int, default=None)
    parser.add_argument('--n_schedule_steps', '-nscst', type=int, default=None)
    parser.add_argument('--optimizer', '-opt', type=str, default=None, choices=['sgd', 'adam', 'adamw', 'cpuadam'])
    parser.add_argument('--morphology_tuning', '-mt', type=str2bool, default=None)

    parser.add_argument('--lr_shared', '-lr', type=float, default=None)
    parser.add_argument('--lr_task_specific', '-lrts', type=float, default=None)
    parser.add_argument('--lr_warmup', '-lrw', type=int, default=None)
    parser.add_argument('--lr_schedule', '-lrs', type=str, default=None, choices=['constant', 'sqroot', 'cos', 'poly'])
    parser.add_argument('--schedule_from', '-scf', type=int, default=None)
    parser.add_argument('--weight_decay', '-wd', type=float, default=None)
    parser.add_argument('--log_dir', '-gdir', type=str, default=None)
    parser.add_argument('--save_dir', '-sdir', type=str, default=None)
    parser.add_argument('--load_dir', '-ldir', type=str, default=None)
    parser.add_argument('--val_iter', '-viter', type=int, default=None)
    parser.add_argument('--save_interval', '-sint', type=int, default=None)
    parser.add_argument('--load_step', '-ls', type=int, default=None)
    parser.add_argument('--load_path', '-lpath', type=str, default=None)
    parser.add_argument('--mt_load_step', '-mls', type=int, default=None)

    if shell_script is not None:
        if shell_script =="":
            args = parser.parse_args(args=[])
        else:
            args = parser.parse_args(args=shell_script.split(' '))
    else:
        args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
        config = EasyDict(config)

    # copy parsed arguments
    for key in args.__dir__():
        if key[:2] != '__' and getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))

    if config.stage < 2 and config.n_schedule_steps < 0:
        config.n_schedule_steps = config.n_steps

    if config.exp_name == '':
        config.exp_name = f"MetaController{config.name_postfix}"

    return config
