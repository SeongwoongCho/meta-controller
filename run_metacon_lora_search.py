import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_processes', '-nprocs', type=int)
    parser.add_argument('--pid', '-pid', type=int)
    parser.add_argument('--name_postfix', '-ptf', type=str, default="")
    parser.add_argument('--shot', default=None, type=int)
    parser.add_argument('--lora_rank', default=None, type=int)
    parser.add_argument('--task', default=None, type=str)
    parser.add_argument('--reset', '-reset', action='store_true')
    args = parser.parse_args()

    if args.lora_rank is None:
        lora_ranks = [4, 8, 16]
    else:
        lora_ranks = [args.lora_rank]

    if args.task is None:
        tasks = ['hopper-hop', 'hopper-hop_backwards', 'hopper-stand', 'wolf-run', 'wolf-walk', 'reacher_four-easy','reacher_four-hard', 'walker-walk_backwards']
    else:
        tasks = [args.task]    
    
    SEEN_MORPH = ['walker-walk_backwards']

    cmds = []
    for lora_rank in lora_ranks:
        for task in tasks:
            if task in SEEN_MORPH:
                cmd = f"bash scripts/finetune.sh {task} --num_train_trajs {args.shot} -tlora {lora_rank} -mt False -snptf _lora:{lora_rank}"
            else:
                cmd = f"bash scripts/finetune.sh {task} --num_train_trajs {args.shot} -mlora {lora_rank} -tlora {lora_rank} -mt True -snptf _lora:{lora_rank}"

            if args.reset:
                cmd += f' --reset'
       
            cmd += f" {args.name_postfix}"
            cmds += [cmd]

    print(len(cmds))

    for ith, cmd in enumerate(cmds):
        if ith % args.n_processes == args.pid:
            print(f"run {ith + 1} / {len(cmds)}")
            print(cmd)
            subprocess.call(cmd, shell=True)
