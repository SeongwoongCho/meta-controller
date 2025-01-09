import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from easydict import EasyDict
from datasets.preprocess_utils import set_state, structurize
from einops import repeat 
from collections import deque, OrderedDict
import copy


class DMC(Dataset):
    ENVIRONMENTS = [
        ('acrobot', 'swingup'),
        ('cartpole', 'balance'),
        ('cartpole', 'swingup'),
        ('cartpole_two', 'poles'),
        ('cheetah', 'flip'),
        ('cheetah', 'flip_backwards'),
        ('cheetah', 'jump'),
        ('cheetah', 'legs_up'),
        ('cheetah', 'lie_down'),
        ('cheetah', 'run'),
        ('cheetah', 'run_back'),
        ('cheetah', 'run_backwards'),
        ('cheetah', 'run_front'),
        ('cheetah', 'stand_back'),
        ('cheetah', 'stand_front'),
        ('cup', 'catch'),
        ('cup', 'spin'),
        ('hopper', 'hop'),
        ('hopper', 'hop_backwards'),
        ('hopper', 'stand'),
        ('pendulum', 'spin'),
        ('pendulum', 'swingup'),
        ('pointmass', 'easy'),
        ('reacher', 'hard'),
        ('reacher_three', 'hard'),
        ('reacher_four', 'easy'),
        ('reacher_four', 'hard'),
        ('walker', 'arabesque'),
        ('walker', 'backflip'),
        ('walker', 'flip'),
        ('walker', 'headstand'),
        ('walker', 'legs_up'),
        ('walker', 'lie_down'),
        ('walker', 'run'),
        ('walker', 'stand'),
        ('walker', 'walk'),
        ('walker', 'walk_backwards'),
        ('wolf', 'run'),
        ('wolf', 'walk')
    ]
    MORPHOLOGIES = sorted(list(set([m for m, _ in ENVIRONMENTS])))
    
    MTRAIN_EVAL_ENVIRONMENTS = [
        ('acrobot', 'swingup'),
        ('cartpole_two', 'poles'),
        ('cheetah', 'run_backwards'),
        ('cup', 'catch'),
        ('pendulum', 'swingup'),
        ('pointmass', 'easy'),
        ('reacher_three', 'hard'),
        ('walker', 'run'),
    ]
    MTEST_ENVIRONMENTS = [
        ('hopper', 'hop'),
        ('hopper', 'hop_backwards'),
        ('hopper', 'stand'),
        ('reacher_four', 'easy'),
        ('reacher_four', 'hard'),
        ('wolf', 'run'),
        ('wolf', 'walk'),
        ('walker', 'walk_backwards'),
    ]

    def __init__(self, stage=0, split='train', demos=None, precision='fp32', env_name=None, dset_size=None,
                 history_size=1):
        super().__init__()
        self.split = split
        self.demos = demos
        self.precision = precision
        self.environment = tuple(env_name.split('-')) if env_name is not None else None
        self.dset_size = dset_size
        
        self.stage = stage
        self.history_size = history_size
        self.toten = ToTensor()

        self.train_environments = self.get_train_environments(self.environment)
        self.split_trajs() # setup data trajectories
        self.base_info = self.get_base_info()

    def __getitem__(self, idx):
        raise NotImplementedError

    def to_precision(self, data):
        if isinstance(data, dict):
            return { 
                k: self.to_precision(v) for k, v in data.items()
            }
        elif isinstance(data, tuple):
            return tuple([ self.to_precision(v) for v in data])
        elif isinstance(data, list):
            return [self.to_precision(v) for v in data]
        elif isinstance(data, str):
            return data
        elif data.dtype == torch.float64 or data.dtype == torch.float32 or data.dtype == torch.bfloat16:
            if self.precision == 'fp32':
                return data.float()
            elif self.precision == 'bf16':
                return data.bfloat16()
            else:
                raise ValueError(self.precision)
        else:
            return data

    def split_trajs(self):
        self.trajs = {}        
        self.traj_ids_all = torch.load('datasets/meta_info/train_traj_ids_all.pt')

        for morphology, task in self.train_environments:
            if self.demos is not None:
                self.traj_ids_all[(morphology, task)] = self.traj_ids_all[(morphology, task)][-self.demos:]
            self.trajs[(morphology, task)] = [f"DMCDATA/{morphology}_{task}/{morphology}_{task}_{traj_id}_500.npz"
                                                    for traj_id in self.traj_ids_all[(morphology, task)]]

        self.obs_dict = {}
        self.act_dict = {}
        self.state_dict = {}
        for environment in self.trajs:
            self.obs_dict[environment] = []
            self.act_dict[environment] = []
            self.state_dict[environment] = []
            for traj in self.trajs[environment]:
                data = np.load(traj)
                self.obs_dict[environment].append(torch.from_numpy(data['observation']))
                self.act_dict[environment].append(torch.from_numpy(data['action']))
                self.state_dict[environment].append(torch.from_numpy(data['state']))
        
        n_trajs = sum([len(trajs) for trajs in self.trajs.values()])
        print(f'loaded {n_trajs} trajectories to memory')

    @classmethod
    def get_train_environments(self, environment=None):
        if environment is not None:
            return [environment]
        else:
            return sorted(set(self.ENVIRONMENTS) - set(self.MTEST_ENVIRONMENTS))
    
    @classmethod
    def get_base_info(self):
        base_info = OrderedDict({})
        for morphology in self.MORPHOLOGIES:
            token_to_act, act_to_token, num_slide_token, num_hinge_token, num_global_token, num_global, dim_obs, dim_act = structurize(morphology, obs=None)
            base_info[morphology] = {
                "token_to_act": token_to_act,
                "act_to_token": act_to_token,
                "num_slide_token": num_slide_token,
                "num_hinge_token": num_hinge_token,
                "num_global_token": num_global_token,
                "num_global": num_global,
                "dim_obs": dim_obs,
                "dim_act": dim_act
            }
        return base_info

    def process_segment(self, obs, act, act_target, morphology, task):
        base_info = self.base_info[morphology]
        
        slide_token, hinge_token, global_token = structurize(morphology, obs=obs, dtype=torch.float32, device=torch.device('cpu'))
        slide_token = F.pad(slide_token, (0, 1), value=0)
        act = (act @ base_info['act_to_token'])[..., None] # (T, d_a) * (d_a, J)
        act_target = (act_target @ base_info['act_to_token'])[..., None] # (T, d_a) * (d_a, J)
        act_mask = (base_info['act_to_token'].abs().sum(dim=0) > 0).unsqueeze(0).expand(self.history_size, -1)
        
        if global_token.size(1) == 0:
            obs = torch.cat([slide_token, hinge_token], dim=1)
        elif global_token.size(1) == 5:
            obs = torch.cat([F.pad(slide_token, (0,2)), F.pad(hinge_token, (0,2)), global_token[:, None,:]], dim=1)
            act = F.pad(act, (0, 0, 0, 1), value=1)
            act_target = F.pad(act_target, (0, 0, 0, 1), value=1)
            act_mask = F.pad(act_mask, (0, 1), value=False)
            obs = torch.cat([slide_token, hinge_token, F.pad(global_token[:, None, :], (0, 1), value=0)], dim=1)
            act = F.pad(act, (0, 0, 0, 1), value=1)
            act_target = F.pad(act_target, (0, 0, 0, 1), value=1)
            act_mask = F.pad(act_mask, (0, 1), value=False)
        else:
            obs = torch.cat([slide_token, hinge_token, F.pad(global_token[:, None, :], (0, 1), value=0)], dim=1)
            act = F.pad(act, (0, 0, 0, 1), value=1)
            act_target = F.pad(act_target, (0, 0, 0, 1), value=1)
            act_mask = F.pad(act_mask, (0, 1), value=False)

        slide_mask = torch.zeros(self.history_size, obs.shape[1], dtype=torch.bool)
        hinge_mask = torch.zeros(self.history_size, obs.shape[1], dtype=torch.bool)
        global_mask = torch.zeros(self.history_size, obs.shape[1], dtype=torch.bool)
        slide_mask[:, :slide_token.size(1)] = True
        hinge_mask[:, slide_token.size(1):slide_token.size(1)+hinge_token.size(1)] = True
        global_mask[:, slide_token.size(1)+hinge_token.size(1):] = True
        morph_mask = torch.ones(obs.size(1), obs.size(1), dtype=torch.bool)
        
        # temporal padding for batching
        assert obs.size(0) == act.size(0) and act.size(0) == act_target.size(0)
        task_mask = torch.ones(self.history_size, dtype=torch.bool)
        task_mask[:self.history_size - obs.size(0)] = False 
        task_mask = repeat(task_mask, 't2 -> j t1 t2', j=obs.size(1), t1=self.history_size)

        obs = F.pad(obs, (0, 0, 0, 0, self.history_size - obs.size(0), 0), value=0)
        act = F.pad(act, (0, 0, 0, 0, self.history_size - act.size(0), 0), value=0)
        act_target = F.pad(act_target, (0, 0, 0, 0, self.history_size - act_target.size(0), 0), value=0)

        if self.stage == 0:
            t_idx = torch.tensor(self.ENVIRONMENTS.index((morphology, task)))
            m_idx = torch.tensor(self.MORPHOLOGIES.index(morphology))
        else:
            t_idx = torch.tensor(0, dtype=torch.long)
            m_idx = torch.tensor(0, dtype=torch.long)

        data = {
            "obs": obs,
            "act": act,
            "act_target": act_target,
            "slide_mask": slide_mask,
            "hinge_mask": hinge_mask,
            "global_mask": global_mask,
            "act_mask": act_mask,
            "morph_mask": morph_mask,
            "task_mask": task_mask,  
            "t_idx": t_idx,
            "m_idx": m_idx,
        }
        data = self.to_precision(data)
        return data
    
    def load_traj(self, environment, traj, end_idx):
        """ 
            obs : (T, J, d_o) # slide, hinge, and global observations
            act: (T, J, d_a) # slide, hinge, and global action values

            slide_mask: (T, J) # True if slide
            hinge_mask: (T, J) # True if hinge
            global_mask: (T, J) # True if global

            act_mask: (T, J) # True if actuable
            morph_mask: (J, J) # (i, j) is True if i-th and j-th joints are coming from the same instance
            task_mask: (J, T, T) # (i, j) is True if i-th and j-th time steps are task-related

            m_idx: () # morphology index
            t_idx: () # task index
        """
        # sample environment, seed, demo, time_idx
        morphology, task = environment
        assert end_idx <= 500
        start_idx = max(0, end_idx - self.history_size)
        
        traj_id = self.trajs[environment].index(traj)
        observation = self.obs_dict[environment][traj_id]
        action = self.act_dict[environment][traj_id]

        obs = observation[start_idx:end_idx].clone() # T, d_o
        act = action[start_idx:end_idx].clone() # T, d_a
        act_target = action[start_idx+1:end_idx+1].clone()
         
        return self.process_segment(obs, act, act_target, morphology, task)
            
    
class DMCEpisodicTrainDataset(DMC):
    """
    DMControl Suite Episodic Meta-Training Dataset Class
    """
    def __init__(self, sampling_shot, window_size, max_n_joints=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_shot = sampling_shot
        self.window_size = window_size
        self.max_n_joints = max_n_joints

    def __len__(self):
        assert self.dset_size is not None
        return self.dset_size

    def sample_single_traj(self, environment, trajs=None, end_idx_start=1):
        if trajs is None:
            assert not isinstance(environment, list)
            if self.demos < 0:
                trajs = self.trajs[environment]
            else:
                trajs = self.trajs[environment][-self.demos:] # nearby the last expert
        
        # sample a trajectory
        traj = random.choice(trajs)
        if isinstance(environment, list):
            assert len(trajs) == len(environment)
            environment = environment[trajs.index(traj)]
        
        # sample a segment
        end_idx = random.randint(end_idx_start, 500)

        return self.load_traj(environment, traj, end_idx)
    
    def sample(self, environment, *args, **kwargs):
        # sample a of trajectories and pad morphology tokens
        data = self.sample_single_traj(environment, *args, **kwargs)
        base_n_joints = data['obs'].size(1)
        while data['obs'].size(1) < self.max_n_joints:
            if data['obs'].size(1) + base_n_joints > self.max_n_joints:
                # pad with zeros
                data['obs'] = F.pad(data['obs'], (0, 0, 0, self.max_n_joints - data['obs'].size(1)), value=0)
                data['act'] = F.pad(data['act'], (0, 0, 0, self.max_n_joints - data['act'].size(1)), value=0)
                data['act_target'] = F.pad(data['act_target'], (0, 0, 0, self.max_n_joints - data['act_target'].size(1)), value=0)
                data['slide_mask'] = F.pad(data['slide_mask'], (0, self.max_n_joints - data['slide_mask'].size(1)), value=False)
                data['hinge_mask'] = F.pad(data['hinge_mask'], (0, self.max_n_joints - data['hinge_mask'].size(1)), value=False)            
                data['global_mask'] = F.pad(data['global_mask'], (0, self.max_n_joints - data['global_mask'].size(1)), value=False)
                data['act_mask'] = F.pad(data['act_mask'], (0, self.max_n_joints - data['act_mask'].size(1)), value=False)
                data['morph_mask'] = F.pad(data['morph_mask'], (0, self.max_n_joints - data['morph_mask'].size(1), 0, self.max_n_joints - data['morph_mask'].size(0)), value=False)
                data['task_mask'] = F.pad(data['task_mask'], (0, 0, 0, 0, 0, self.max_n_joints - data['task_mask'].size(0)), value=False)
            else:
                # sample another trajectory
                new_data = self.sample_single_traj(environment)
                # concatenate
                data['obs'] = torch.cat([data['obs'], new_data['obs']], dim=1)
                data['act'] = torch.cat([data['act'], new_data['act']], dim=1)
                data['act_target'] = torch.cat([data['act_target'], new_data['act_target']], dim=1)
                data['slide_mask'] = torch.cat([data['slide_mask'], new_data['slide_mask']], dim=1)
                data['hinge_mask'] = torch.cat([data['hinge_mask'], new_data['hinge_mask']], dim=1)
                data['global_mask'] = torch.cat([data['global_mask'], new_data['global_mask']], dim=1)
                data['act_mask'] = torch.cat([data['act_mask'], new_data['act_mask']], dim=1)
                data['morph_mask'] = F.pad(data['morph_mask'], (0, base_n_joints, 0, base_n_joints), value=False) # 
                data['morph_mask'][-base_n_joints:, -base_n_joints:] = new_data['morph_mask']
                data['task_mask'] = torch.cat([data['task_mask'], new_data['task_mask']], dim=0)
        
        return data 

    def __getitem__(self, idx):
        # sample support
        environment = random.choice(self.train_environments)
        if self.demos < 0:
            trajs = self.trajs[environment]
            traj_start_idx = random.randint(0, len(trajs) - self.window_size - 1)
            trajs = [trajs[i] for i in range(traj_start_idx, traj_start_idx + self.window_size + 1)] 
        else:
            trajs= self.trajs[environment][-self.demos:] # nearby the last expert
        
        data_S = None
        for i in range(self.sampling_shot):
            if self.stage == 0:
                data = self.sample(environment, trajs=trajs, end_idx_start=self.window_size)
            else:
                data = self.sample_single_traj(environment, trajs=trajs, end_idx_start=self.window_size)
            if data_S is None:
                data_S = {k: v[None] for k, v in data.items()}
            else:
                data_S = {k: torch.cat((data_S[k], data[k][None])) for k in data}

        data_Q = None
        for _ in range(self.sampling_shot):
            if self.stage == 0:
                data = self.sample(environment, trajs=trajs)
            else:
                data = self.sample_single_traj(environment, trajs=trajs)
            if data_Q is None:
                data_Q = {k: v[None] for k, v in data.items()}
            else:
                data_Q = {k: torch.cat((data_Q[k], data[k][None])) for k in data}
                
        return {"data_S": data_S, "data_Q": data_Q}


class SupportLoader(DMC):
    """
    DMControl Suite Support Data Loader for Play Evaluation
    """
    def __init__(self, window_size, max_n_joints=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.max_n_joints = max_n_joints

    def load_data(self, environment, demos):
        # load trajectories from expert demonstrations
        
        trajs = self.trajs[environment]
        support_trajs = trajs[-demos:]
             
        data_S = None
        for traj in support_trajs:
            end_idxs = list(range(self.window_size, 501, self.history_size))
            for end_idx in end_idxs:
                data = self.load_traj(environment, traj, end_idx)
                if data_S is None:
                    data_S = {k: v[None] for k, v in data.items()}
                else:
                    data_S = {k: torch.cat((data_S[k], data[k][None])) for k in data}

        return {k: v[None] for k, v in data_S.items()}


class DMCTestEnvironment(DMC):
    def __init__(self, env_name, stage, initial_states=None, history_size=1, precision='fp32', *args, **kwargs):
        self.env_name = env_name
        self.stage = stage

        morphology, task = tuple(env_name.split('-'))
        self.morphology = morphology
        self.task = task
        self.env = self.construct_env(morphology, task, *args, **kwargs)
        self.env.reset() # to set xml files
        
        self.initial_states = initial_states # list of initial states
        self.history_size = history_size
        self.clear_history()
        
        self.precision = precision
        self.base_info = self.get_base_info()

    def __len__(self):
        return len(self.initial_states)

    def __getitem__(self, idx):
        return self   

    @property
    def unwrapped(self):
        return self.env.unwrapped._env._env._env._env

    @property
    def mjmodel(self):
        return self.unwrapped.physics.named.model

    @property
    def mjdata(self):
        return self.unwrapped.physics.named.data

    def construct_env(self, morphology, task, *args, **kwargs):
        from datasets.dm_control import make_env
        cfg = EasyDict()
        cfg.task = f"{morphology}-{task}"
        cfg.obs = 'state'
        cfg.seed = 1 
        env = make_env(cfg, *args, **kwargs) 
        return env

    def render(self, size, camera_id=None):
        image = self.env.render(mode='rgb_array', width=size[1], height=size[0], camera_id=camera_id)
        image = torch.from_numpy(image.copy()) / 255
        image = image.permute(2, 0, 1)
        return image

    def get_current_state(self):
        return torch.from_numpy(self.unwrapped.physics.get_state())

    def clear_history(self):
        self.history = {
            'obs': deque(maxlen=self.history_size),
            'act': deque(maxlen=self.history_size),
        }
        self.full_history = {
            'obs': [],
            'act': [],
            'rewards': [],
            'states': []
        }

    def process_history(self):
        assert len(self.history['obs']) == len(self.history['act'])

        obs = torch.stack(list(self.history['obs'])) # (T, d_o)
        act = torch.stack(list(self.history['act'])) # (T, d_a)
        
        data = self.process_segment(obs, act, act, self.morphology, self.task)

        for key in data:
            data[key] = data[key].unsqueeze(0)
        return data

    def reset(self, initial_state=None):
        if initial_state is not None:
            obs = set_state(self.unwrapped, self.env_name, initial_state)          
        else:
            obs = self.env.reset()
        
        self.clear_history()
        self.history['obs'].append(torch.tensor(obs))
        self.full_history['obs'].append(torch.tensor(obs))
        self.history['act'].append(torch.zeros(self.unwrapped.action_spec().shape))
        self.full_history['act'].append(torch.zeros(self.unwrapped.action_spec().shape))
        self.full_history['rewards'].append(0)
        self.full_history['states'].append(self.get_current_state())
        return self.process_history()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.clone().float().cpu().numpy()
        
        if action.ndim == 4: # B, T, J, 1
            assert action.shape[0] == 1 and action.shape[3] == 1
            action = action[0, -1, :, 0]
        elif action.ndim == 5: # B, N, T, J, 1
            assert action.shape[0] == 1 and action.shape[1] == 1 and action.shape[4] == 1
            action = action[0, 0, -1, :, 0]
        else:
            raise ValueError(f"unknown action ndim {action.ndim}")
    
        if self.base_info[self.morphology]['num_global_token'] > 0:
            action = action[:-1] # remove global action

        action = action @ self.base_info[self.morphology]['token_to_act'].numpy()
        
        obs, reward, done, _ = self.env.step(action)
        self.history['obs'].append(torch.tensor(obs))
        self.history['act'].append(torch.from_numpy(action))
        self.full_history['obs'].append(torch.tensor(obs))
        self.full_history['act'].append(torch.tensor(action))
        self.full_history['rewards'].append(reward)
        self.full_history['states'].append(self.get_current_state())
        
        return self.process_history(), float(reward), done, _


class DMCEpisodicTestEnvironment(DMCTestEnvironment):
    def __init__(self, support_loader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.support_loader = support_loader
 
    def set_support_data(self, demos):
        self.support_data = self.support_loader.load_data((self.morphology, self.task), demos)

    def process_history(self):
        data_Q = super().process_history()
        data_Q = {k: v[None] for k, v in data_Q.items()}
        return data_Q
