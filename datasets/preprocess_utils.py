import numpy as np
import collections
import torch
import torch.nn.functional as F
from einops import rearrange


NORMALIZE_DICT = {
    'acrobot': {
        'ABS_HVEL_MAX': 33.4081
    },
    'cartpole': {
        'NUM_HINGE': 1,
        'ABS_SPOS_MAX': 1.8,
        'ABS_SVEL_MAX': 6.2579,
        'ABS_HVEL_MAX': 27.1258
    },
    'cartpole_two': {
        'NUM_HINGE': 2,
        'ABS_SPOS_MAX': 1.8,
        'ABS_SVEL_MAX': 5.4431,
        'ABS_HVEL_MAX': 33.1520
    },
    'pendulum': {
        'ABS_HVEL_MAX': 11.6488
    },
    'walker': {
        'NUM_HINGE': 7,
        'ABS_HEIGHT_MAX':1.7789253737905732,
        'ABS_SVEL_MAX': 8.4086,
        'ABS_HVEL_MAX': 88.0010
    },
    'cheetah': {
        'HINGE_NAMES': ['torso', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot'],
        'ABS_BASE_HEIGHT': 0.7,
        'ABS_HEIGHT_MAX': 0.7334679187102413,
        'ABS_SVEL_MAX': 13.2030,
        'ABS_HVEL_MAX': 36.2797
    },
    'cup': {
        'ABS_SPOS_MAX': 0.34,
        'ABS_SVEL_MAX': 6.6530,
    },
    'hopper':{
        'NUM_HINGE': 5,
        'ABS_BASE_HEIGHT': 1,
        'ABS_HEIGHT_MAX': 0.1256132012170696,
        'ABS_SVEL_MAX': 5.1547,
        'ABS_HVEL_MAX': 40.5725,
        'TOUCH_SCALE': 8.2326,
    },
    'pointmass': {
        'ABS_SPOS_MAX': 0.3,
        'ABS_SVEL_MAX': 0.0993,
    },
    'reacher': {
        'NUM_HINGE': 2,
        'ABS_HVEL_MAX': 6.8155,
        'ABS_TO_TARGET': 0.4655
    },
    'reacher_three': {
        'NUM_HINGE': 3,
        'ABS_HVEL_MAX': 9.5484,
        'ABS_TO_TARGET': 0.4655
    },
    'reacher_four': {
        'NUM_HINGE': 4,
        'ABS_HVEL_MAX': 10.9394,
        'ABS_TO_TARGET': 0.4655
    },
    'wolf': {
        'NUM_HINGE': 13,
        'ABS_HEIGHT_MAX': 2.158788014143396,
        'ABS_BASE_HEIGHT': 1.25,
        'ABS_SVEL_MAX': 10.5894,
        'ABS_HVEL_MAX': 97.4775
    },
}


def flatten_observation(observation, output_key='observations'):
    """Flattens multiple observation arrays into a single numpy array.
    
    Args:
        observation: A mutable mapping from observation names to numpy arrays.
        output_key: The key for the flattened observation array in the output.

    Returns:
        A mutable mapping of the same type as `observation`. This will contain a
        single key-value pair consisting of `output_key` and the flattened
        and concatenated observation array.

    Raises:
        ValueError: If `observation` is not a `collections.abc.MutableMapping`.
    """
    if not isinstance(observation, collections.abc.MutableMapping):
        raise ValueError('Can only flatten dict-like observations.')

    if isinstance(observation, collections.OrderedDict):
        keys = observation.keys()
    else:
        # Keep a consistent ordering for other mappings.
        keys = sorted(observation.keys())

    observation_arrays = [observation[key].ravel() for key in keys]
    return type(observation)([(output_key, np.concatenate(observation_arrays))])


def set_state(env, env_name, initial_state):
    env.reset()
    physics = env.physics
    
    states, raw_obs = initial_state
    if isinstance(states, torch.Tensor):
        states = states.cpu().numpy()
    for key in raw_obs:
        if isinstance(raw_obs[key], torch.Tensor):
            raw_obs[key] = raw_obs[key].cpu().numpy()

    physics.set_state(states)
    physics.forward()

    # Env-specific reset
    if env_name in ['reacher-easy', 'reacher-hard', 'reacher_three-easy', 'reacher_three-hard', 'reacher_four-easy', 'reacher_four-hard']:
        physics.named.model.geom_pos['target', ['x', 'y']] = physics.named.data.geom_xpos['finger', :2] + raw_obs['to_target']
        physics.forward()

    try: 
        observation = env._task.get_observation(env._physics)
    except:
        observation = env._observation_updater.get_observation()
    observation = flatten_observation(observation)
        
    return observation['observations']


def structurize(morphology, obs=None, normalize_dict=NORMALIZE_DICT, dtype=torch.float32, device=torch.device('cpu')):
    """
    obs: (b, d)
    morphology: 'str'

    slide_token: (b, n, 2) # n>=0
    hinge_token: (b, n, 3) # n>=0
    global_token: (b, d), d >= 0
    
    hinge_action_sign: (n2, )
    token_to_act: (n1 + n2, c)
    act_to_token: (c, n1 + n2 or n1 + n2 + 1) # 1 if global exists
    """
    normalize_dict = normalize_dict[morphology]

    if morphology == 'acrobot':
        # orientations (4, ): xmat[xx, xz] (upper_arm, lower_arm)
        # velocities (2, ): shoulder, elbow
        ABS_HVEL_MAX = normalize_dict['ABS_HVEL_MAX']
        
        dim_obs = 6
        dim_act = 1
        num_slide_token = 0
        num_hinge_token = 2
        num_global_token = 0
        num_global = 0

        token_to_act = torch.tensor([[0],[1]]).to(dtype).to(device) # (shoulder, elbow) -> elbow
        act_to_token = token_to_act.T
        assert token_to_act.abs().sum() == dim_act
        if obs is None:
            return token_to_act, act_to_token, num_slide_token, num_hinge_token, num_global_token, num_global, dim_obs, dim_act

        assert obs.size(1) == 6
        batch_size = obs.shape[0]
        
        # parse position
        qpos_shoulder = torch.atan2(obs[:, 0], obs[:, 2])
        qpos_elbow =  torch.atan2(obs[:, 1], obs[:, 3])
        qpos = torch.stack([qpos_shoulder, qpos_elbow], dim=-1).to(dtype).to(device)

        # parse velocity
        qvel = (obs[:, 4:] / ABS_HVEL_MAX).to(dtype).to(device)

        # structurize as tokens
        slide_token = torch.zeros(batch_size, num_slide_token, 2, dtype=dtype, device=device)
        hinge_token = token = torch.stack([torch.sin(qpos), torch.cos(qpos), qvel], dim=-1).to(dtype).to(device)
        global_token = torch.zeros(batch_size, 0, dtype=dtype, device=device)

    elif morphology == 'cartpole' or morphology == 'cartpole_two':
        ABS_SPOS_MAX = normalize_dict['ABS_SPOS_MAX']
        ABS_SVEL_MAX = normalize_dict['ABS_SVEL_MAX']
        ABS_HVEL_MAX = normalize_dict['ABS_HVEL_MAX']
        
        dim_obs = 2 + 3 * normalize_dict['NUM_HINGE']
        dim_act = 1
        num_slide_token = 1
        num_hinge_token = normalize_dict['NUM_HINGE']
        num_global_token = 0
        num_global = 0
        
        token_to_act = torch.zeros(num_slide_token + num_hinge_token, 1).to(dtype).to(device) # (cart, pole k) -> cart
        token_to_act[0, 0] = 1
        act_to_token = token_to_act.T
        assert token_to_act.abs().sum() == dim_act
        if obs is None:
            return token_to_act, act_to_token, num_slide_token, num_hinge_token, num_global_token, num_global, dim_obs, dim_act
            
        assert obs.size(1) == 2 + 3 * num_hinge_token
        batch_size = obs.shape[0]
        
        cart_position = obs[:, :1].clamp(-ABS_SPOS_MAX, ABS_SPOS_MAX)
        pole_position = rearrange(obs[:, 1:1+2*num_hinge_token], 'b (n k) -> b n k', k=2)
        pole_qpos = torch.atan2(pole_position[..., 1], pole_position[..., 0]).to(dtype) # b, n
        cart_velocity = obs[:, 1+2*num_hinge_token:2+2*num_hinge_token]
        pole_velocity = obs[:, 2+2*num_hinge_token:]
        
        slide_token = torch.stack([cart_position / ABS_SPOS_MAX, cart_velocity / ABS_SVEL_MAX], dim=-1).to(dtype).to(device)
        hinge_token = torch.stack([torch.sin(pole_qpos), torch.cos(pole_qpos), pole_velocity / ABS_HVEL_MAX], dim=-1).to(dtype).to(device)
        global_token = torch.zeros(batch_size, 0, dtype=dtype, device=device)

    elif morphology == 'cheetah':
        # position (8, ): root_z (slide), root_y (hinge), bthigh, bshin, bfoot, fthigh, fshin, ffoot)
        # velocity (9, ): root_x (slide), root_z(slide), root_y (hinge), bthigh, bshin, bfoot, fthigh, fshin, ffoot
        ABS_BASE_HEIGHT = normalize_dict['ABS_BASE_HEIGHT']
        ABS_HEIGHT_MAX = normalize_dict['ABS_HEIGHT_MAX']
        ABS_SVEL_MAX = normalize_dict['ABS_SVEL_MAX']
        ABS_HVEL_MAX = normalize_dict['ABS_HVEL_MAX']
        HINGE_NAMES = normalize_dict['HINGE_NAMES']

        HINGE_QPOS_INIT = {
            'torso': np.pi / 2,
            'bthigh': 3.9822545,
            'bshin': -4.52467281,
            'bfoot': 1.66805613 - np.pi, 
            'fthigh': 2.10220164 + np.pi,
            'fshin': -1.1186774,
            'ffoot': 0.01385836
        }
        dim_obs = 2 * len(HINGE_NAMES) + 3
        dim_act = len(HINGE_NAMES) - 1
        num_slide_token = 2
        num_global_token = 0
        num_hinge_token = len(HINGE_NAMES)
        num_global = 0
        
        token_to_act = torch.zeros(num_slide_token + num_hinge_token, num_hinge_token - 1).to(dtype).to(device)
        # for i in range(num_slide_token + 1, num_hinge_token):
        for i in range(num_slide_token + 1, num_slide_token + num_hinge_token):
            token_to_act[i, i - (num_slide_token + 1)] = 1
        act_to_token = token_to_act.T
        assert token_to_act.abs().sum() == dim_act
        if obs is None:
            return token_to_act, act_to_token, num_slide_token, num_hinge_token, num_global_token, num_global, dim_obs, dim_act
            
        assert obs.size(1) == 2 * num_hinge_token + 3
        batch_size = obs.shape[0]
        
        # parse position
        qpos_slide = F.pad(((obs[:, :1] + ABS_BASE_HEIGHT) / (ABS_BASE_HEIGHT + ABS_HEIGHT_MAX)), (1, 0)).to(dtype).to(device)
        qpos_hinge = obs[:, 1:num_hinge_token+1].to(dtype=dtype).to(device)
        
        qpos_hinge_dict = {hinge_name: qpos_hinge[:, i] + HINGE_QPOS_INIT[hinge_name] for i, hinge_name in enumerate(HINGE_NAMES)} 
        qpos_hinge_dict_cum = {'torso': qpos_hinge_dict['torso']}
        if 'bthigh' in qpos_hinge_dict:
            qpos_hinge_dict_cum['bthigh'] = qpos_hinge_dict['torso'] + qpos_hinge_dict['bthigh'] + np.pi
        if 'bshin' in qpos_hinge_dict:
            qpos_hinge_dict_cum['bshin'] = qpos_hinge_dict['torso'] + qpos_hinge_dict['bthigh'] + qpos_hinge_dict['bshin'] + np.pi
        if 'bfoot' in qpos_hinge_dict:
            qpos_hinge_dict_cum['bfoot'] = qpos_hinge_dict['torso'] + qpos_hinge_dict['bthigh'] + qpos_hinge_dict['bshin'] + qpos_hinge_dict['bfoot'] + np.pi
        if 'fthigh' in qpos_hinge_dict:
            qpos_hinge_dict_cum['fthigh'] = qpos_hinge_dict['torso'] + qpos_hinge_dict['fthigh']  + np.pi
        if 'fshin' in qpos_hinge_dict:
            qpos_hinge_dict_cum['fshin'] = qpos_hinge_dict['torso'] + qpos_hinge_dict['fthigh'] + qpos_hinge_dict['fshin']  + np.pi
        if 'ffoot' in qpos_hinge_dict:
            qpos_hinge_dict_cum['ffoot'] = qpos_hinge_dict['torso'] + qpos_hinge_dict['fthigh'] + qpos_hinge_dict['fshin'] + qpos_hinge_dict['ffoot']  + np.pi
        qpos_hinge = torch.stack([qpos_hinge_dict_cum[hinge_name] for hinge_name in HINGE_NAMES], dim=-1).to(dtype=dtype)
        
        # parse velocity
        velocity = obs[:, num_hinge_token+1:]
        qvel_slide = (velocity[:, :2] / ABS_SVEL_MAX).to(dtype).to(device)
        qvel_hinge = (velocity[:, 2:] / ABS_HVEL_MAX).to(dtype).to(device)

        slide_token = torch.stack([qpos_slide, qvel_slide], dim=-1)
        hinge_token = torch.stack([torch.sin(qpos_hinge), torch.cos(qpos_hinge), qvel_hinge], dim=-1)
        global_token = torch.zeros(batch_size, 0, dtype=dtype, device=obs.device)
    
    elif morphology == 'cup':
        # position (4, ): (cup_x, cup_z, ball_x, ball_z)
        # velocity (4, ): (cup_x, cup_z, ball_x, ball_z)
        ABS_SPOS_MAX = normalize_dict['ABS_SPOS_MAX']
        ABS_SVEL_MAX = normalize_dict['ABS_SVEL_MAX']
        
        dim_obs = 4
        dim_act = 2
        num_slide_token = 4
        num_hinge_token = 0
        num_global_token = 0
        num_global = 0
        
        token_to_act = torch.tensor([[1, 0, 0, 0],
                                     [0, 1, 0, 0]]).T.to(dtype).to(device) # (cup_x, cup_z, ball_x, ball_z) -> (cup_x, cup_z)
        act_to_token = token_to_act.T
        assert token_to_act.abs().sum() == dim_act
        if obs is None:
            return token_to_act, act_to_token, num_slide_token, num_hinge_token, num_global_token, num_global, dim_obs, dim_act
        
        assert obs.size(1) == 8
        batch_size = obs.shape[0]

        # parse position
        qpos_slide = obs[:, :4].clamp(-ABS_SPOS_MAX, ABS_SPOS_MAX) / ABS_SPOS_MAX
        qvel_slide = obs[:, 4:] / ABS_SVEL_MAX

        slide_token = torch.stack((qpos_slide, qvel_slide), dim=-1).to(dtype).to(device)
        hinge_token = torch.zeros(batch_size, num_hinge_token, 3, dtype=dtype, device=device)
        global_token = torch.zeros(batch_size, 0, dtype=dtype, device=device)

    elif morphology == 'pendulum':
        ABS_HVEL_MAX = normalize_dict['ABS_HVEL_MAX']
        
        dim_obs = 3
        dim_act = 1
        num_slide_token = 0
        num_hinge_token = 1
        num_global_token = 0
        num_global = 0

        token_to_act = torch.tensor([[1]]).to(dtype).to(device) # (pendulum -> pendulum)
        act_to_token = token_to_act.T
        assert token_to_act.abs().sum() == dim_act
        if obs is None:
            return token_to_act, act_to_token, num_slide_token, num_hinge_token, num_global_token, num_global, dim_obs, dim_act

        assert obs.size(1) == 3
        batch_size = obs.shape[0]
        
        qpos = torch.atan2(obs[:, 1], obs[:, 0])[:, None].to(dtype=dtype, device=obs.device)
        qvel = (obs[:, 2:] / ABS_HVEL_MAX).to(dtype).to(device)
        
        slide_token = torch.zeros(batch_size, num_slide_token, 2, dtype=dtype, device=device)
        hinge_token = torch.stack([torch.sin(qpos), torch.cos(qpos), qvel], dim=-1)
        global_token = torch.zeros(batch_size, 0, dtype=dtype, device=obs.device)

    elif morphology == 'pointmass':
        ABS_SPOS_MAX = normalize_dict['ABS_SPOS_MAX']
        ABS_SVEL_MAX = normalize_dict['ABS_SVEL_MAX']
        
        dim_obs = 4
        dim_act = 2
        num_slide_token = 2
        num_hinge_token = 0
        num_global_token = 0
        num_global = 0

        token_to_act = torch.tensor([[1, 0], 
                                     [0, 1]]).to(dtype).to(device) # (root_x, root_y -> tendon_x, tendon_y)
        act_to_token = token_to_act.T
        assert token_to_act.abs().sum() == dim_act
        if obs is None:
            return token_to_act, act_to_token, num_slide_token, num_hinge_token, num_global_token, num_global, dim_obs, dim_act

        assert obs.size(1) == 4
        batch_size = obs.shape[0]
        
        # parse_position
        qpos_slide = obs[: ,:2].clamp(-ABS_SPOS_MAX, ABS_SPOS_MAX) / ABS_SPOS_MAX
        qvel_slide = obs[:, 2:] / ABS_SVEL_MAX

        slide_token = torch.stack((qpos_slide, qvel_slide), dim=-1).to(dtype).to(device)
        hinge_token = torch.zeros(batch_size, num_hinge_token, 3, dtype=dtype, device=device)
        global_token = torch.zeros(batch_size, 0, dtype=dtype, device=device)
    
    elif morphology == 'walker':
        ABS_HEIGHT_MAX = normalize_dict['ABS_HEIGHT_MAX']
        ABS_SVEL_MAX = normalize_dict['ABS_SVEL_MAX']
        ABS_HVEL_MAX = normalize_dict['ABS_HVEL_MAX']

        dim_obs = 3 * normalize_dict['NUM_HINGE'] + 3
        dim_act = normalize_dict['NUM_HINGE'] - 1
        num_slide_token = 2
        num_global_token = 0
        num_hinge_token = normalize_dict['NUM_HINGE']
        num_global = 0
        
        hinge_action_sign = torch.tensor([1] + [-1] * (num_hinge_token-1)).to(dtype).to(device)
        token_to_act = torch.zeros(num_slide_token + num_hinge_token, num_hinge_token - 1).to(dtype).to(device)
        for i in range(num_slide_token + 1, num_slide_token + num_hinge_token):
            # token_to_act[i, i - (num_slide_token + 1)] = 1
            token_to_act[i, i - (num_slide_token + 1)] = -1
        act_to_token = token_to_act.T
        assert token_to_act.abs().sum() == dim_act
        if obs is None:
            return token_to_act, act_to_token, num_slide_token, num_hinge_token, num_global_token, num_global, dim_obs, dim_act 

        assert obs.size(1) == 3 * num_hinge_token + 3
        batch_size = obs.shape[0]
        
        # parse orientation, velocity, and height
        ori = obs[:, :2*num_hinge_token]
        ori = rearrange(ori, 'b (n d) -> b n d', n=num_hinge_token, d=2) 
        velocity = obs[:, 2*num_hinge_token + 1:]
        height = obs[:, 2*num_hinge_token]

        # construct slide token
        qpos_slide = F.pad((height[:, None] / ABS_HEIGHT_MAX).to(dtype), (0, 1), value=0) # root_z, root_x (B, 2)  
        qvel_slide = (velocity[:, :2] / ABS_SVEL_MAX).to(dtype)
        slide_token = torch.stack([qpos_slide, qvel_slide], dim=-1)
        
        # construct hinge token
        qpos_hinge = [torch.atan2(ori[:, 0, 1], ori[:, 0, 0])] # torso
        qpos_hinge.append(torch.atan2(ori[:, 1, 1], ori[:, 1, 0]) - np.pi) # right hip (> two) or left hip (two)
        if morphology != 'walker_two':
            qpos_hinge.append(torch.atan2(ori[:, 2, 1], ori[:, 2, 0]) - np.pi) # right knee (> four) or left hip (three, four)
            if morphology != 'walker_three':
                if morphology == 'walker':
                    qpos_hinge.append(torch.atan2(ori[:, 3, 1], ori[:, 3, 0]) - 3 * np.pi / 2) # right ankle
                else:
                    qpos_hinge.append(torch.atan2(ori[:, 3, 1], ori[:, 3, 0]) - np.pi) # left hip (> four) or left knee (four)
                if morphology != 'walker_four':
                    qpos_hinge.append(torch.atan2(ori[:, 4, 1], ori[:, 4, 0]) - np.pi) # left hip (> six) or left knee (five, six)
                    if morphology == 'walker':
                        qpos_hinge.append(torch.atan2(ori[:, 5, 1], ori[:, 5, 0]) - np.pi) # left knee
                        qpos_hinge.append(torch.atan2(ori[:, 6, 1], ori[:, 6, 0]) - 3 * np.pi / 2) # left ankle
                    elif morphology =='walker_six':
                        qpos_hinge.append(torch.atan2(ori[:, 5, 1], ori[:, 5, 0]) - 3 * np.pi / 2) # left ankle
                        
        qpos_hinge = torch.stack(qpos_hinge, dim=1).to(dtype).to(device)
        qvel_hinge = (hinge_action_sign[None] * velocity[:, 2:] / ABS_HVEL_MAX).to(dtype).to(device)
        hinge_token = torch.stack([torch.sin(qpos_hinge), torch.cos(qpos_hinge), qvel_hinge], dim=-1)
        global_token = torch.zeros(batch_size, 0, dtype=dtype, device=device)
    
    elif morphology == 'hopper':
        # position (6, ): root_z (slide), root_y (hinge), waist, hip, knee, ankle
        # velocity (7, ): root_x (slide), root_z(slide), root_y (hinge), waist, hip, knee, ankle
        ABS_BASE_HEIGHT = normalize_dict['ABS_BASE_HEIGHT']
        TOUCH_SCALE = normalize_dict['TOUCH_SCALE']
        ABS_SVEL_MAX = normalize_dict['ABS_SVEL_MAX']
        ABS_HVEL_MAX = normalize_dict['ABS_HVEL_MAX']
        ABS_HEIGHT_MAX = normalize_dict['ABS_HEIGHT_MAX']
        
        dim_obs = 2 * normalize_dict['NUM_HINGE'] + 5
        dim_act = normalize_dict['NUM_HINGE'] - 1
        num_slide_token = 2
        num_hinge_token = normalize_dict['NUM_HINGE']
        num_global_token = 1
        num_global = 2

        token_to_act = torch.zeros(num_slide_token + num_hinge_token, num_hinge_token - 1).to(dtype).to(device)
        for i in range(num_slide_token + 1, num_slide_token + num_hinge_token):
            token_to_act[i, i - (num_slide_token + 1)] = 1
        act_to_token = token_to_act.T
        assert token_to_act.abs().sum() == dim_act
        if obs is None:
            return token_to_act, act_to_token, num_slide_token, num_hinge_token, num_global_token, num_global, dim_obs, dim_act

        assert obs.size(1) == 2 * num_hinge_token + 5
        batch_size = obs.shape[0]
        
        position = obs[:, :1 + num_hinge_token]
        velocity = obs[:, 1 + num_hinge_token:-2]
        touch = obs[:, -2:]

        qpos_slide = F.pad(((position[:, :1] + ABS_BASE_HEIGHT) / (ABS_HEIGHT_MAX + ABS_BASE_HEIGHT)).to(dtype), (1, 0), value=0) # (B, 2)
        qvel_slide = (velocity[:, :2] / ABS_SVEL_MAX).to(dtype=dtype)
        slide_token = torch.stack([qpos_slide, qvel_slide], dim=-1)
        
        # structurize
        # qpos_hinge = position[:, 1:]
        qpos_hinge = [position[:, 1], position[:, 1:3].sum(dim=1)] # root_y, waist
        if morphology != 'hopper_three':
            qpos_hinge.append(position[:, 1:4].sum(dim=1) - np.pi) # hip
            if morphology == 'hopper':
                qpos_hinge.append(position[:, 1:5].sum(dim=1) - np.pi) # knee
                qpos_hinge.append(position[:, 1:6].sum(dim=1) - 3 * np.pi / 2) # ankle
            elif morphology == 'hopper_four':
                qpos_hinge.append(position[:, 1:5].sum(dim=1) - 3 * np.pi / 2) # ankle
        else:
            qpos_hinge.append(position[:, 1:4].sum(dim=1) - 3 * np.pi / 2) # ankle
        
        qpos_hinge = torch.stack(qpos_hinge, dim=1).to(dtype)
        qvel_hinge = (velocity[:, 2:] / ABS_HVEL_MAX).to(device).to(dtype)
        hinge_token = torch.stack([torch.sin(qpos_hinge), torch.cos(qpos_hinge), qvel_hinge], dim=-1)

        global_token = (touch / TOUCH_SCALE).to(dtype)

    elif morphology == 'reacher' or morphology == 'reacher_three' or morphology == 'reacher_four':
        ABS_HVEL_MAX = normalize_dict['ABS_HVEL_MAX']
        ABS_TO_TARGET = normalize_dict['ABS_TO_TARGET']
        
        dim_obs = 2 * normalize_dict['NUM_HINGE'] + 2
        dim_act = normalize_dict['NUM_HINGE']
        num_slide_token = 0
        num_hinge_token = normalize_dict['NUM_HINGE']
        num_global_token = 1
        num_global = 2

        token_to_act = torch.eye(num_hinge_token).to(dtype).to(device)
        act_to_token = token_to_act.T
        assert token_to_act.abs().sum() == dim_act
        if obs is None:
            return token_to_act, act_to_token, num_slide_token, num_hinge_token, num_global_token, num_global, dim_obs, dim_act

        assert obs.size(1) == 2 * num_hinge_token + 2
        batch_size = obs.shape[0]

        # parse 
        hinge_qpos = obs[:, :num_hinge_token]
        for hinge_idx in range(num_hinge_token-1, 0, -1):
            hinge_qpos[:, hinge_idx] += hinge_qpos[:, :hinge_idx].sum(1)
        hinge_qvel = obs[:, num_hinge_token+2:]
        to_target = obs[:, num_hinge_token:num_hinge_token+2]
        
        slide_token = torch.zeros(batch_size, 0, 2, dtype=dtype, device=device)
        hinge_token = torch.stack([torch.sin(hinge_qpos), torch.cos(hinge_qpos), hinge_qvel / ABS_HVEL_MAX], dim=-1).to(dtype).to(device)
        global_token = (to_target / ABS_TO_TARGET).to(dtype).to(device)
    
    elif morphology == 'wolf':
        ABS_HEIGHT_MAX = normalize_dict['ABS_HEIGHT_MAX']
        ABS_SVEL_MAX = normalize_dict['ABS_SVEL_MAX']
        ABS_HVEL_MAX = normalize_dict['ABS_HVEL_MAX']
        
        dim_obs = 3 * normalize_dict['NUM_HINGE'] + 3
        dim_act = normalize_dict['NUM_HINGE'] - 1
        num_slide_token = 2
        num_global_token = 0
        num_hinge_token = normalize_dict['NUM_HINGE']
        num_global = 0

        hinge_action_sign = torch.tensor([1] + [-1] * (num_hinge_token-1)).to(dtype).to(device)
        token_to_act = torch.cat((torch.zeros(num_slide_token + 1, dim_act), -torch.eye(dim_act)), dim=0)
        act_to_token = token_to_act.T
        assert token_to_act.abs().sum() == dim_act, f'{morphology} {token_to_act.abs().sum()} != {dim_act}'
        if obs is None:
            return token_to_act, act_to_token, num_slide_token, num_hinge_token, num_global_token, num_global, dim_obs, dim_act 
    
        assert obs.size(1) == 3 * num_hinge_token + 3
        batch_size = obs.shape[0]
        
        # parse orientation, velocity, and height
        ori = obs[:, :2*num_hinge_token]
        ori = rearrange(ori, 'b (n d) -> b n d', n=num_hinge_token, d=2) 
        velocity = obs[:, 2*num_hinge_token + 1:]
        height = obs[:, 2*num_hinge_token]
    
        # construct slide token
        qpos_slide = F.pad((height[:, None] / ABS_HEIGHT_MAX).to(dtype), (0, 1), value=0) # root_z, root_x (B, 2)  
        qvel_slide = (velocity[:, :2] / ABS_SVEL_MAX).to(dtype)
        slide_token = torch.stack([qpos_slide, qvel_slide], dim=-1)
        
        # construct hinge token
        v0 = torch.atan2(ori[:, 0, 1], ori[:, 0, 0]) - np.pi / 2
        v1 = torch.atan2(ori[:, 1, 1], ori[:, 1, 0]) + np.pi
        v2 = torch.atan2(ori[:, 2, 1], ori[:, 2, 0]) + np.pi
        v3 = torch.atan2(ori[:, 3, 1], ori[:, 3, 0]) + np.pi/2
        v4 = torch.atan2(ori[:, 4, 1], ori[:, 4, 0]) + np.pi
        v5 = torch.atan2(ori[:, 5, 1], ori[:, 5, 0]) + np.pi
        v6 = torch.atan2(ori[:, 6, 1], ori[:, 6, 0]) + np.pi/2
        v7 = torch.atan2(ori[:, 7, 1], ori[:, 7, 0]) + np.pi
        v8 = torch.atan2(ori[:, 8, 1], ori[:, 8, 0]) + np.pi
        v9 = torch.atan2(ori[:, 9, 1], ori[:, 9, 0]) + np.pi/2
        v10 = torch.atan2(ori[:, 10, 1], ori[:, 10, 0]) + np.pi
        v11 = torch.atan2(ori[:, 11, 1], ori[:, 11, 0]) + np.pi
        v12 = torch.atan2(ori[:, 12, 1], ori[:, 12, 0]) + np.pi/2
        qpos_hinge = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12]
                        
        qpos_hinge = torch.stack(qpos_hinge, dim=1).to(dtype).to(device)
        qvel_hinge = (hinge_action_sign[None] * velocity[:, 2:] / ABS_HVEL_MAX).to(dtype).to(device)
        hinge_token = torch.stack([torch.sin(qpos_hinge), torch.cos(qpos_hinge), qvel_hinge], dim=-1)
        global_token = torch.zeros(batch_size, 0, dtype=dtype, device=device)

    return slide_token, hinge_token, global_token
