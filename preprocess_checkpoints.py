import argparse
import os
import torch
from train.zero_to_fp32 import _get_fp32_state_dict_from_zero_checkpoint


def reduce_checkpoint(ckpt_path, reduced_path, verbose=True):
    # load state dict and config
    if os.path.isdir(ckpt_path):
        state_dict = _get_fp32_state_dict_from_zero_checkpoint(os.path.join(ckpt_path, 'checkpoint'))
        state_dict = {k.replace('_forward_module.', ''): v for k, v in state_dict.items()}
        ckpt = torch.load(os.path.join(ckpt_path, 'checkpoint', 'mp_rank_00_model_states.pt'), map_location='cpu')
        for key in ckpt['module']:
            if 'ema_model' in key:
                state_dict[key.replace('_forward_module.', '')] = ckpt['module'][key]
    else:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['state_dict']

    # reduce memory
    ckpt_reduced = {}
    ckpt_reduced['state_dict'] = state_dict
    ckpt_reduced['config'] = ckpt['hyper_parameters']['config']
    ckpt_reduced['global_step'] = ckpt['global_step']
    torch.save(ckpt_reduced, reduced_path)
    if verbose:
        print(f'checkpoint converted to memory-reduced checkpoint: {ckpt_path}')


parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', '-rd', type=str, default=None)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--ckpt_name', '-cname', type=str, default=None)
parser.add_argument('--load_dir', '-ld', type=str, default='TRAIN')
parser.add_argument('--verbose', '-v', default=False, action='store_true')
parser.add_argument('--reset_mode', '-reset', default=False, action='store_true')
args = parser.parse_args()


if args.root_dir is None:
    root_dir = 'experiments'
else:
    root_dir = args.root_dir

if args.exp_name is None:
    exp_names = sorted(os.listdir(os.path.join(root_dir, args.load_dir)))
else:
    exp_names = [args.exp_name]

exp_names_ = []
for exp_name in exp_names:
    if exp_name == '.ipynb_checkpoints':
        continue
    if 'checkpoints' in os.listdir(os.path.join(root_dir, args.load_dir, exp_name)):
        exp_names_.append(exp_name)
    else:
        for exp_subname in os.listdir(os.path.join(root_dir, args.load_dir, exp_name)):
            if 'checkpoints' in os.listdir(os.path.join(root_dir, args.load_dir, exp_name, exp_subname)):
                exp_names_.append(os.path.join(exp_name, exp_subname))
exp_names = exp_names_

for exp_name in exp_names:
    if exp_name == '.ipynb_checkpoints':
        continue
    
    if args.ckpt_name is None:
        ckpt_names = sorted(os.listdir(os.path.join(root_dir, args.load_dir, exp_name, 'checkpoints')))
        ckpt_names = [name for name in ckpt_names if name.endswith('.ckpt')]
    else:
        ckpt_names = [args.ckpt_name]

    for ckpt_name in ckpt_names:
        ckpt_path = os.path.join(root_dir, args.load_dir, exp_name, 'checkpoints', f'{ckpt_name}')
        reduced_path = ckpt_path.replace('.ckpt', '.pth')
        if not os.path.exists(ckpt_path):
            if args.verbose:
                print(f'checkpoint not found: {ckpt_path}')
            continue
        if os.path.exists(reduced_path) and not args.reset_mode and (not (reduced_path.endswith('last.pth') or reduced_path.endswith('best.pth'))):
            if args.verbose:
                print(f'checkpoint already exists: {reduced_path}')
            continue

        try:
            reduce_checkpoint(ckpt_path, reduced_path, args.verbose)
        except Exception as e:
            print(f"fail at processing {ckpt_path}")
            print(e)
