# Meta-Controller

This repository contains official pytorch implementation for [Meta-Controller: Few-Shot Imitation of Unseen Embodiments and Tasks in Continuous Control](https://openreview.net/pdf?id=M5D5rMwLjj) (NeurIPS 2024).
![image-metacon](https://github.com/SeongwoongCho/meta-controller/blob/main/MetaControllerOverview.png)

## Setup
### Dataset
Prepare meta-training and downstream datasets (Replay buffers of DrQv2 agents) from [Here](https://drive.google.com/file/d/16SHG_AwqySJJ48frFuTksyCkMB8AsFNn/view?usp=sharing). The directory structure looks like:
```

meta-controller
|--main.py
|--args.py
|--...
|--DMCDATA
|   |--VALIDATIONSTATES
|   |  |--<embodiment1>_<task1>_rawobs.pt
|   |  |--<embodiment1>_<task1>_states.pt
|   |  |--<embodiment2>_<task2>_rawobs.pt
|   |  |--<embodiment2>_<task2>_states.pt
|   |  | ...
|   |
|   |--<embodiment1>_<task1>
|   |  |--<embodiment1>_<task1>_<file1>.npz 
|   |  |--<embodiment1>_<task1>_<file2>.npz
|   |  |--...
|   |
|   |--<embodiment2>_<task2>
|   |  |--<embodiment2>_<task2>_<file1>.npz 
|   |  |--<embodiment2>_<task2>_<file2>.npz
|   |  |--...
|   |
|   |...
```
### Meta-trained Checkpoints
We provide meta-trained checkpoints in [Here]().
Please locate it under `experiments/TRAIN/MetaController/checkpoints`


## Usage
### Meta-Training
```
$ bash scripts/train.sh

# convert deepspeed checkpoints into single .pth file
$ python preprocess_checkpoints.py -ld TRAIN
```

### Fine-Tuning
```
# Finetuning with specific lora rank
$ bash scripts/finetune.sh ${env_name} -tlora {lora_rank} -mlora {lora_rank} -snptf _lora:{lora_rank}

or 

# Finetuning for all tasks and lora rank in [4, 8, 16]
$ python run_metacon_lora_search.py 
```

### Evaluation
```
$ bash scripts/test.sh ${exp_subname}
```

## Citation
If you find this work useful, please consider citing:
```bib
@inproceedings{cho2024metacontroller,
  title={Meta-Controller: Few-Shot Imitation of Unseen Embodiments and Tasks in Continuous Control},
  author={Seongwoong Cho and Donggyun Kim and Jinwoo Lee and Seunghoon Hong},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=M5D5rMwLjj}
}
```
