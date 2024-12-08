from dm_control import suite
import numpy as np
import torch
import xmltodict
import json
import os
import torch
from tqdm import tqdm
from dm_control.suite.fish import _JOINTS


def get_xml(domain):
    xml_path = f'datasets/dmc_utils/xmls/{domain}.xml'
    xml_file = ''
    with open(xml_path, 'r') as f:
        for line in f.readlines():
            xml_file += line
    
    xml_dict = xmltodict.parse(xml_file)
    xml_json = json.dumps(xml_dict)
    xml_dict = json.loads(xml_json)

    return xml_dict


def get_reduced_xml(xml_dict):
    def reduce_xml(parent):
        if isinstance(parent, list):
            return [reduce_xml(child) for child in parent]
        elif isinstance(parent, dict):
            reduced_xml = {}
            for key in parent.keys():
                if key in ['body', 'joint', '@name']:
                    reduced_xml[key.replace('@name', 'name')] = reduce_xml(parent[key])
            return reduced_xml
        else:
            return parent
    
    root = xml_dict['mujoco']['worldbody']['body']
    
    return reduce_xml(root)


def get_edges(root, body_names, joint_names): 
    def append_edge(edges, edge_index, joint_dict, joint_index_dict, parent):   
        parent_node = body_names.index(parent['name'])
        if 'body' in parent:
            if isinstance(parent['body'], list):
                children = parent['body']
            elif isinstance(parent['body'], dict):
                children = [parent['body']]
            else:
                raise TypeError(parent['body'], type(parent['body']))
            
            for child in children:
                child_node = body_names.index(child['name'])
                edges.append([parent['name'], child['name']])
                edge_index.append([parent_node, child_node])

                if 'joint' in parent:
                    if isinstance(parent['joint'], list):
                        joints = parent['joint']
                    elif isinstance(parent['joint'], dict):
                        joints = [parent['joint']]
                    else:
                        raise TypeError(parent['joint'], type(parent['joint']))

                    for joint in joints:
                        if joint['name'] in joint_dict:
                            joint_dict[joint['name']].append([parent['name'], child['name']])
                            joint_index_dict[joint_names.index(joint['name'])].append([parent_node, child_node])
                        else:
                            joint_dict[joint['name']] = [[parent['name'], child['name']]]
                            joint_index_dict[joint_names.index(joint['name'])] = [[parent_node, child_node]]
                
                append_edge(edges, edge_index, joint_dict, joint_index_dict, child)
        else:
            edges.append([parent['name'], parent['name']])
            edge_index.append([parent_node, parent_node])

            if 'joint' in parent:
                if isinstance(parent['joint'], list):
                    joints = parent['joint']
                elif isinstance(parent['joint'], dict):
                    joints = [parent['joint']]
                else:
                    raise TypeError(parent['joint'], type(parent['joint']))

                for joint in joints:
                    if joint['name'] in joint_dict:
                        joint_dict[joint['name']].append([parent['name'], parent['name']])
                        joint_index_dict[joint_names.index(joint['name'])].append([parent_node, parent_node])
                    else:
                        joint_dict[joint['name']] = [[parent['name'], parent['name']]]
                        joint_index_dict[joint_names.index(joint['name'])] = [[parent_node, parent_node]]
            
    edges = []
    edge_index = []
    joint_dict = {}
    joint_index_dict = {}
    
    append_edge(edges, edge_index, joint_dict, joint_index_dict, root)
    return edges, edge_index, joint_dict, joint_index_dict

domain_tasks = [
    ('acrobot', 'swingup'),
    ('cartpole', 'balance'),
    ('cartpole', 'balance_sparse'),
    ('cartpole', 'swingup'),
    ('cartpole', 'swingup_sparse'),
    ('cheetah', 'jump'),
    ('cheetah', 'run'),
    ('cheetah', 'run_back'),
    ('cheetah', 'run_backwards'),
    ('cheetah', 'run_front'),
    ('ball_in_cup', 'catch'),
    ('ball_in_cup', 'spin'),
    ('finger', 'spin'),
    ('finger', 'turn_easy'),
    ('finger', 'turn_hard'),
    ('fish', 'swim'),
    ('fish', 'upright'),
    ('hopper', 'hop'),
    ('hopper', 'hop_backwards'),
    ('hopper', 'stand'),
    ('humanoid', 'run'),
    ('humanoid', 'stand'),
    ('humanoid', 'walk'),
    ('manipulator', 'bring_ball'),
    ('pendulum', 'spin'),
    ('pendulum', 'swingup'),
    ('point_mass', 'easy'),
    ('reacher', 'easy'),
    ('reacher', 'hard'),
    ('reacher', 'three_easy'),
    ('reacher', 'three_hard'),
    ('walker', 'run'),
    ('walker', 'run_backwards'),
    ('walker', 'stand'),
    ('walker', 'walk'),
    ('walker', 'walk_backwards')
]

domains = list(set([domain for domain, task in domain_tasks]))
xml_dicts = {}
envs = {}

for domain, task in tqdm(domain_tasks):
    xml_dicts[domain] = get_xml(domain)
    envs[(domain, task)] = suite.load(domain, task) 
xml_dicts['reacher_three_links'] = get_xml('reacher_three_links')

def extract_graph(domain, task):
    # parse_xml_tree
    if domain == 'reacher' and task in ['three_easy', 'three_hard']:
        xml_dict = xml_dicts['reacher_three_links']
    else:
        xml_dict = xml_dicts[domain]
    
    xml_tree = get_reduced_xml(xml_dict)
    if domain in ['cup', 'ball_in_cup', 'finger', 'manipulator' ]:
        xml_tree = xml_tree[0]

    # Extract graph structure
    if domain == 'point_mass':
        body_names = ['pointmass_1', 'pointmass_2']
        joint_names = ['root_x_1', 'root_y_1', 'root_x_2', 'root_y_2']
        edges = [['pointmass_1', 'pointmass_1'], ['pointmass_1', 'pointmass_1'], ['pointmass_2', 'pointmass_2'], ['pointmass_2', 'pointmass_2']]
        edge_index = [[0, 0], [0, 0], [1, 1], [1, 1]]
        joint_dict = {'root_x_1' : [['pointmass_1', 'pointmass_1']], 'root_y_1' : [['pointmass_1', 'pointmass_1']], 'root_x_2' : [['pointmass_2', 'pointmass_2']], 'root_y_2' : [['pointmass_2', 'pointmass_2']]}
        joint_index_dict = {0: [[0, 0]], 1: [[0, 0]], 2: [[1, 1]], 3: [[1, 1]]}
        actuable_joints = ['root_x_1', 'root_y_1', 'root_x_2', 'root_y_2']
        actuable_joint_index = [0, 1, 2, 3]
        action_index = [0, 0, 1, 1]
    elif domain == 'fish':
        body_names = [name for name in envs[(domain, task)].physics.named.data.xpos.axes.row.names if name != 'world']
        joint_names = [name for name in envs[(domain, task)].physics.named.data.qpos.axes.row.names]

        edges, edge_index, joint_dict, joint_index_dict = get_edges(xml_tree, body_names, joint_names)

        actuable_joints = [motor['@joint'] for motor in xml_dict['mujoco']['actuator']['position'] if '@joint' in motor] + ['finleft_roll', 'finright_roll']
        actuable_joint_index = [joint_names.index(motor['@joint'])
                                for motor in xml_dict['mujoco']['actuator']['position'] if '@joint' in motor] + [joint_names.index('finleft_roll'), joint_names.index('finright_roll')]
        action_index = [0, 1, 3, 4, 2, 2]
    elif domain in ['acrobot', 'cartpole', 'pendulum']:
        body_names = [name for name in envs[(domain, task)].physics.named.data.xpos.axes.row.names if name != 'world']
        joint_names = [name for name in envs[(domain, task)].physics.named.data.qpos.axes.row.names if name != 'root']

        edges, edge_index, joint_dict, joint_index_dict = get_edges(xml_tree, body_names, joint_names)

        actuable_joints = [motor['@joint'] for motor in [xml_dict['mujoco']['actuator']['motor']] if '@joint' in motor]
        actuable_joint_index = [joint_names.index(motor['@joint'])
                                for motor in [xml_dict['mujoco']['actuator']['motor']] if '@joint' in motor]
        action_index = list(range(len(actuable_joints)))
    elif domain in ['cheetah', 'ball_in_cup', 'finger', 'hopper', 'humanoid', 'manipulator', 'reacher', 'walker']:
        # extract graph structure
        body_names = [name for name in envs[(domain, task)].physics.named.data.xpos.axes.row.names if name != 'world']
        joint_names = [name for name in envs[(domain, task)].physics.named.data.qpos.axes.row.names if name != 'root']

        edges, edge_index, joint_dict, joint_index_dict = get_edges(xml_tree, body_names, joint_names)

        actuable_joints = [motor['@joint'] for motor in xml_dict['mujoco']['actuator']['motor'] if '@joint' in motor]
        actuable_joint_index = [joint_names.index(motor['@joint'])
                                for motor in xml_dict['mujoco']['actuator']['motor'] if '@joint' in motor]
        action_index = list(range(len(actuable_joints)))
    
    return body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index

node_dim = 19 # pos_xyz (3), ori_xyz^2 (9), vel_xyz (3), angvel_xyz (3), ang(1)
edge_dim = 2 # relpos (1), relvel (1)

# acrobot, (swingup, )
def _parse_observation_acrobot(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info

    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}
    
    # position
    for node_idx, node_feature in enumerate(obs['orientations']):
        if node_idx <2 :
            node_dict[node_idx][5] = node_feature
            node_mask_dict[node_idx][5] = True
        else :
            node_dict[node_idx-2][11] = node_feature
            node_mask_dict[node_idx-2][11] = True
    
    # velocity
    for edge_idx, edge_feature in enumerate(obs['velocity']):
        edge_dict[edge_idx][1] = edge_feature
        edge_mask_dict[edge_idx][1] = True
   
    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]
            
    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# cartpole, ('balance', 'balance_sparse', 'swingup', 'swingup_sparse')
def _parse_observation_cartpole(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info 
    
    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}
    
    # position_cart_position
    edge_idx = joint_names.index('slider')
    edge_dict[edge_idx][0] = obs['position'][0]
    edge_mask_dict[edge_idx][0] =True
    
    # position_cart_position
    node_idx = body_names.index('pole_1')
    node_dict[node_idx][3:5] = obs['position'][1:]
    node_mask_dict[node_idx][3:5] = np.array([True]*2)

    # velocity_slider
    edge_idx = joint_names.index('slider')
    edge_dict[edge_idx][1] = obs['velocity'][1]
    edge_mask_dict[edge_idx][1] =True
    
    # velocity_hinge_1
    edge_idx = joint_names.index('hinge_1')
    edge_dict[edge_idx][1] = obs['velocity'][1]
    edge_mask_dict[edge_idx][1] =True
    
    
    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]
            
    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# cheetah, (jump, run, run_back, run_backwards, run_front)
def _parse_observation_cheetah(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info
    
    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}
    
    
    # position
    for edge_idx, edge_feature in enumerate(obs['position']):
        edge_dict[edge_idx+1][0] = edge_feature
        edge_mask_dict[edge_idx+1][0] = True
    
    # velocity
    for edge_idx, edge_feature in enumerate(obs['velocity']):
        edge_dict[edge_idx][1] = edge_feature
        edge_mask_dict[edge_idx][1] = True
   
    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]
            
    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# ball_in_cup, (catch, spin)
def _parse_observation_ballincup(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info
 
    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}
    
    # position_global
    global_dict['position_ball_x'] = np.array([obs['position'][2]])
    global_dict['position_ball_z'] = np.array([obs['position'][3]])

    # velocity_global
    global_dict['velocity_ball_x'] = np.array([obs['velocity'][2]])
    global_dict['velocity_ball_z'] = np.array([obs['velocity'][3]])
    
    # joint_angles
    for edge_idx, edge_feature in enumerate(obs['position'][:2]):
        edge_dict[edge_idx][0] = edge_feature
        edge_mask_dict[edge_idx][0] = True
    
    # joint_velocities
    for edge_idx, edge_feature in enumerate(obs['velocity'][:2]):
        edge_dict[edge_idx][1] = edge_feature
        edge_mask_dict[edge_idx][1] = True
    
    
    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]
            
    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# finger, (spin,)
def _parse_observation_finger_spin(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info
 
    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}

    # position
    for edge_idx, pos in enumerate(obs['position'][:2]):
        edge_dict[edge_idx][0] = pos
        edge_mask_dict[edge_idx][0] = True

    # velocity
    for edge_idx, vel in enumerate(obs['velocity']):
        edge_dict[edge_idx][1] = vel
        edge_mask_dict[edge_idx][1] = True

    # tip position and touch
    global_dict['tip_position'] = obs['position'][2:]
    global_dict['touch'] = obs['touch']

    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]

    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# finger (turn_easy, turn_hard)
def _parse_observation_finger_turn(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info
 
    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}

    # position
    for edge_idx, pos in enumerate(obs['position'][:2]):
        edge_dict[edge_idx][0] = pos
        edge_mask_dict[edge_idx][0] = True

    # velocity
    for edge_idx, vel in enumerate(obs['velocity']):
        edge_dict[edge_idx][1] = vel
        edge_mask_dict[edge_idx][1] = True

    # tip position, touch, target_position, and dist_to_target
    global_dict['tip_position'] = obs['position'][2:]
    global_dict['touch'] = obs['touch']
    global_dict['target_position'] = obs['target_position']
    global_dict['dist_to_target'] = np.array([obs['dist_to_target']])
    
    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]
            
    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# fish (swim,)
def _parse_observation_fish_swim(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info
    
    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}

    # joint angles
    for i, joint in enumerate(_JOINTS):
        edge_idx = joint_names.index(joint)
        edge_dict[edge_idx][0] = obs['joint_angles'][i]
        edge_mask_dict[edge_idx][0] = True

    # velocity except root
    for edge_idx, vel in enumerate(obs['velocity'][6:]):
        edge_dict[edge_idx+1][1] = vel
        edge_mask_dict[edge_idx+1][1] = True

    # torso velocity
    node_idx = body_names.index('torso')
    node_dict[node_idx][12:18] = obs['velocity'][:6]
    node_mask_dict[node_idx][12:18] = np.array([True]*6)

    # upright
    node_idx = body_names.index('torso')
    node_dict[node_idx][11] = obs['upright']
    node_mask_dict[node_idx][11] = True

    # target
    global_dict['target'] = obs['target']

    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]

    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# fish, (upright,)
def _parse_observation_fish_upright(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info

    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}

    # joint angles
    for i, joint in enumerate(_JOINTS):
        edge_idx = joint_names.index(joint)
        edge_dict[edge_idx][0] = obs['joint_angles'][i]
        edge_mask_dict[edge_idx][0] = True

    # velocity except root
    for edge_idx, vel in enumerate(obs['velocity'][6:]):
        edge_dict[edge_idx+1][1] = vel
        edge_mask_dict[edge_idx+1][1] = True

    # torso velocity
    node_idx = body_names.index('torso')
    node_dict[node_idx][12:18] = obs['velocity'][:6]
    node_mask_dict[node_idx][12:18] = np.array([True]*6)

    # upright
    node_idx = body_names.index('torso')
    node_dict[node_idx][11] = obs['upright']
    node_mask_dict[node_idx][11] = True

    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]
            
    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# hopper, (hop, hop_backwardsm, stand)
def _parse_observation_hopper(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info
    
    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}

    # joint position
    for i, edge_feature in enumerate(obs['position']):
        edge_dict[i+1][0] = edge_feature
        edge_mask_dict[i+1][0] = True

    # joint velocity
    for i, edge_feature in enumerate(obs['velocity']):
        edge_dict[i][1] = edge_feature
        edge_mask_dict[i][1] = True

    # touch
    global_dict['touch'] = obs['touch']
    
    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]
            
    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# humanoid, (run, stand, walk)
def _parse_observation_humanoid(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info
    
    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}


    # head_height
    node_idx = body_names.index('head')
    node_dict[node_idx][2] = obs['head_height']
    node_mask_dict[node_idx][2] = True

    # extremities
    i = 0
    for side in ('left_', 'right_'):
        for limb in ('hand', 'foot'):
            node_idx = body_names.index(side + limb)
            node_dict[node_idx][:3] = obs['extremities'][i:i+3]
            node_mask_dict[node_idx][:3] = np.array([True]*3)
            i += 3

    # torso_vertical
    node_idx = body_names.index('torso')
    node_dict[node_idx][9:12] = obs['torso_vertical']
    node_mask_dict[node_idx][9:12] = np.array([True]*3)

    # body velocities
    node_idx = body_names.index('torso')
    node_dict[node_idx][12:18] = obs['velocity'][:6]
    node_mask_dict[node_idx][12:18] = np.array([True]*6)

    # joint_angles
    for edge_idx, edge_feature in enumerate(obs['joint_angles']):
        edge_dict[edge_idx][0] = edge_feature
        edge_mask_dict[edge_idx][0] = True

    # joint_velocities
    for edge_idx, edge_feature in enumerate(obs['velocity'][6:]):
        edge_dict[edge_idx][1] = edge_feature
        edge_mask_dict[edge_idx][1] = True

    # com_velocity
    global_dict['com_velocity'] = obs['com_velocity']

    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]

    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# manipulation, (bring_ball,)
def _parse_observation_manipulation_bring_ball(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info
    
    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}

    # arm pos & vel
    for i, arm_joint in enumerate(_ARM_JOINTS):
        edge_idx = joint_names.index(arm_joint)
        edge_dict[edge_idx][0] = np.arctan2(*obs['arm_pos'][i])
        edge_dict[edge_idx][1] = obs['arm_vel'][i]
        edge_mask_dict[edge_idx][0] = True
        edge_mask_dict[edge_idx][1] = True

    # hand pos
    node_idx = body_names.index('hand')
    node_dict[node_idx][0] = obs['hand_pos'][0]
    node_dict[node_idx][2] = obs['hand_pos'][1]
    node_mask_dict[node_idx][0] = True
    node_mask_dict[node_idx][2] = True
    
    quat = np.array([obs['hand_pos'][2], 0, obs['hand_pos'][3], 0])
    node_dict[node_idx][3:12] = R.from_quat(quat).as_matrix().reshape(-1)
    node_mask_dict[node_idx][3:12] = True

    # touch, object_pos, object_vel, target_pos
    for key in ['touch', 'object_pos', 'object_vel', 'target_pos']:
        global_dict[key] = obs[key]
    
    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]
            
    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# pendulum, (spin, swingup)
def _parse_observation_pendulum(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info
    
    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}
    
    # orientation
    node_idx = body_names.index('pole')
    node_dict[node_idx][5] = obs['orientation'][0]
    node_dict[node_idx][11] = obs['orientation'][1]
    node_mask_dict[node_idx][5] = True
    node_mask_dict[node_idx][11] = True

    #velocity
    edge_idx = joint_names.index('hinge')
    edge_dict[edge_idx][1] = obs['velocity'][0]
    edge_mask_dict[edge_idx][1] = True
    
    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]
            
    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# pointmass, easy
def _parse_observation_pointmass(obs, grpah_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info
    
    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}
    
    # joint_angles
    for edge_idx, edge_feature in enumerate(obs['position']):
        edge_dict[edge_idx][0] = edge_feature
        edge_dict[edge_idx + 2][0] = edge_feature
        edge_mask_dict[edge_idx][0] = True
        edge_mask_dict[edge_idx + 2][0] = True
    
    # joint_angles
    for edge_idx, edge_feature in enumerate(obs['velocity']):
        edge_dict[edge_idx][1] = edge_feature
        edge_dict[edge_idx + 2][1] = edge_feature
        edge_mask_dict[edge_idx][1] = True
        edge_mask_dict[edge_idx + 2][1] = True

    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]
            
    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# reacher, (easy, hard)
def _parse_observation_reacher(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info
    
    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}

    # position
    for edge_idx, edge_feature in enumerate(obs['position']):
        edge_dict[edge_idx][0] = edge_feature
        edge_mask_dict[edge_idx][0] = True

    # velocity
    for edge_idx, edge_feature in enumerate(obs['velocity']):
        edge_dict[edge_idx][1] = edge_feature
        edge_mask_dict[edge_idx][1] = True

    # to_target
    global_dict['to_target'] = obs['to_target']
    
    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]
            
    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# reacher_three_links, (easy, hard)
def _parse_observation_reacher_three_links(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info

    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}

    # position
    for edge_idx, edge_feature in enumerate(obs['position']):
        edge_dict[edge_idx][0] = edge_feature
        edge_mask_dict[edge_idx][0] = True

    # velocity
    for edge_idx, edge_feature in enumerate(obs['velocity']):
        edge_dict[edge_idx][1] = edge_feature
        edge_mask_dict[edge_idx][1] = True

    # to_target
    global_dict['to_target'] = obs['to_target']
    
    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]
            
    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict

# walker, (run, run_backwards, stand, walk, walk_backwards)
def _parse_observation_walker(obs, graph_info):
    body_names, joint_names, edges, edge_index, joint_dict, joint_index_dict, actuable_joints, actuable_joint_index, action_index = graph_info
    
    ## Create dictionary
    node_dict = {node_idx: np.zeros(node_dim, dtype=np.float32) for node_idx in range(len(body_names))}
    node_mask_dict = {node_idx: np.zeros(node_dim, dtype=bool) for node_idx in range(len(body_names))}
    edge_dict = {edge_idx: np.zeros(edge_dim, dtype=np.float32) for edge_idx in range(len(joint_names))}
    edge_mask_dict = {edge_idx: np.zeros(edge_dim, dtype=bool) for edge_idx in range(len(joint_names))}
    global_dict = {}

    # orientations
    for node_idx, ori in enumerate(obs['orientations'].reshape(-1, 2)):
        node_dict[node_idx][3] = ori[0]
        node_dict[node_idx][5] = ori[1]
        node_mask_dict[node_idx][3] = True
        node_mask_dict[node_idx][5] = True

    # height
    node_idx = body_names.index('torso')
    node_dict[node_idx][2] = obs['height']
    node_mask_dict[node_idx][2] = True

    # velocity
    for edge_idx, vel in enumerate(obs['velocity']):
        edge_dict[edge_idx][1] = vel
        edge_mask_dict[edge_idx][1] = True

    ## delete empty rows
    for node_idx in list(node_mask_dict.keys()):
        if node_mask_dict[node_idx].max() == 0:
            del node_mask_dict[node_idx]
            del node_dict[node_idx]

    for edge_idx in list(edge_mask_dict.keys()):
        if edge_mask_dict[edge_idx].max() == 0:
            del edge_mask_dict[edge_idx]
            del edge_dict[edge_idx]

    return node_dict, node_mask_dict, edge_dict, edge_mask_dict, global_dict


