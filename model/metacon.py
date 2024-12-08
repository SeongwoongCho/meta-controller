import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general import Transformer
from einops import rearrange, repeat
import math
import copy

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def morphology_specific_parameters(self):
        raise NotImplementedError();

    def morphology_specific_parameter_names(self):
        morphology_specific_parameters = [id(p) for p in self.morphology_specific_parameters()]
        names = []
        for n, p in self.named_parameters():
            if id(p) in morphology_specific_parameters:
                names.append(n)
        return names
    
    def task_specific_parameters(self):
        raise NotImplementedError();

    def task_specific_parameter_names(self):
        task_specific_parameters = [id(p) for p in self.task_specific_parameters()]
        names = []
        for n, p in self.named_parameters():
            if id(p) in task_specific_parameters:
                names.append(n)
        return names


class JointEmbedding(BaseModule):
    """
    Embed joint features consists of slide, hinge, and global features
    """
    def __init__(self, slide_dim, hinge_dim, num_global_list, hidden_dim=256, global_bias=True):
        super().__init__()
        self.slide_dim = slide_dim
        self.hinge_dim = hinge_dim
        self.num_global_list = num_global_list
        self.hidden_dim = hidden_dim

        self.proj_slide = nn.Linear(slide_dim, hidden_dim)
        self.proj_hinge = nn.Linear(hinge_dim, hidden_dim)
        self.proj_global = nn.ModuleList([nn.Linear(num_global, hidden_dim, bias=global_bias) if num_global > 0
                                          else None for num_global in num_global_list])
        
    def morphology_specific_parameters(self):
        for param in self.proj_global.parameters():
            yield param

    def forward(self, joint, slide_mask, hinge_mask, global_mask, m_idx):
        '''
        data:
            joint: (B, T, J, d_o) # slide, hinge, and global features
            slide_mask: (B, T, J) # True if slide
            hinge_mask: (B, T, J) # True if hinge
            global_mask: (B, T, J) # True if global
            m_idx: (B,) # morphology index

        embedding: (B, T, J, d_h)
        '''
        # create an embedding tensor
        B, T, J = joint.shape[:3]
        embedding = torch.zeros(B, T, J, self.hidden_dim, dtype=joint.dtype, device=joint.device)

        # process slide features
        emb_slide = self.proj_slide(joint[..., :self.slide_dim])
        embedding = torch.where(slide_mask[..., None], emb_slide, embedding)

        # process hinge features
        emb_hinge = self.proj_hinge(joint[..., :self.hinge_dim])
        embedding = torch.where(hinge_mask[..., None], emb_hinge, embedding)
        
        # process global features morphology-specifically
        for b, m_idx_ in enumerate(m_idx):
            num_global = self.num_global_list[m_idx_.item()]
            if num_global > 0:
                emb_global = self.proj_global[m_idx_.item()](joint[b, ..., :num_global])
                embedding[b] = torch.where(global_mask[b, ..., None], emb_global, embedding[b])

        return embedding 


class ObsMorphologyTransformer(BaseModule):
    def __init__(self, config, num_joint_list, num_global_list):
        super().__init__()
        self.num_joint_list = num_joint_list
        self.obs_embedding = JointEmbedding(slide_dim=2, hinge_dim=3, num_global_list=num_global_list, hidden_dim=config.hidden_dim, global_bias=True)
        self.actuator_embedding = nn.Linear(1, config.hidden_dim, bias=False)
        
        self.position_embedding = nn.ParameterList([nn.Parameter(torch.randn(1, num_joint, config.hidden_dim)) for num_joint in num_joint_list])

        self.obs_morphology_encoder = Transformer(config.n_morph_blocks, config.n_attn_heads, config.hidden_dim,
                                                  n_tsparam=len(num_joint_list), lora_rank=config.morphology_lora_rank,
                                                  attn_drop=config.drop_rate, drop=config.drop_rate, drop_path=config.drop_rate,
                                                  layerscale=config.morphology_layerscale)

    def morphology_specific_parameters(self):
        for param in self.position_embedding:
            yield param
        for param in self.obs_embedding.morphology_specific_parameters():
            yield param
        for param in self.obs_morphology_encoder.task_specific_parameters():
            yield param

    def forward(self, data):
        '''
        [Input]
        data:
            obs : (B, T, J, d_o) # slide, hinge, and global observations

            slide_mask: (B, T, J) # True if slide
            hinge_mask: (B, T, J) # True if hinge
            global_mask: (B, T, J) # True if global

            act_mask: (B, T, J) # True if actuable
            morph_mask: (B, J, J) # (i, j) is True if i-th and j-th joints are morphologically connected
            m_idx: (B,) # morphology index

        [Output]
        x: (B, T, J, d_h) # state tokens
        '''
        obs = data['obs']
        slide_mask, hinge_mask, global_mask = data['slide_mask'], data['hinge_mask'], data['global_mask']
        act_mask, morph_mask = data['act_mask'], data['morph_mask']
        m_idx = data['m_idx']
        T = obs.shape[1]

        # get actuator embedding
        actuator_embedding = self.actuator_embedding(act_mask[..., None].to(dtype=obs.dtype))

        # get joint embedding
        x = self.obs_embedding(obs, slide_mask, hinge_mask, global_mask, m_idx) # (B, T, J, d)
        
        # add actuator embedding
        x = x + actuator_embedding

        # get position embedding
        position_embedding = torch.zeros_like(x)
        for b, m_idx_ in enumerate(m_idx):
            num_joint = self.num_joint_list[m_idx_.item()]
            position_embedding[b, :, :num_joint] = self.position_embedding[m_idx_.item()].repeat(T, 1, 1)

        # add spatial embedding
        x = x + position_embedding

        # encode morphology in spatial axis
        x = rearrange(x, 'B T J d -> (B T) J d')
        morph_attn_mask = repeat(morph_mask, 'B J1 J2 -> (B T) J1 J2', T=T)
        

        m_idx = repeat(m_idx, 'B -> (B T)', T=T)
        x = self.obs_morphology_encoder(x, attn_mask=morph_attn_mask, t_idx=m_idx)
        x = rearrange(x, '(B T) J d -> B T J d', T=T)
        
        return x
    

class ActMorphologyTransformer(BaseModule):
    def __init__(self, config, num_joint_list, num_global_list):
        super().__init__()
        self.num_joint_list = num_joint_list
        self.act_embedding = JointEmbedding(slide_dim=1, hinge_dim=1, num_global_list=[1 if num_global > 0 else 0 for num_global in num_global_list],
                                            hidden_dim=config.hidden_dim, global_bias=False)
        self.actuator_embedding = nn.Linear(1, config.hidden_dim, bias=False)
        self.position_embedding = nn.ParameterList([nn.Parameter(torch.randn(1, num_joint, config.hidden_dim)) for num_joint in num_joint_list])
        self.act_morphology_encoder = Transformer(config.n_morph_blocks, config.n_attn_heads, config.hidden_dim,
                                                    n_tsparam=len(num_joint_list), lora_rank=config.morphology_lora_rank,
                                                    attn_drop=config.drop_rate, drop=config.drop_rate, drop_path=config.drop_rate,
                                                    layerscale=config.morphology_layerscale)

    def morphology_specific_parameters(self):
        for param in self.position_embedding:
            yield param
        for param in self.act_embedding.morphology_specific_parameters():
            yield param
        for param in self.act_morphology_encoder.task_specific_parameters():
            yield param

    def forward(self, data, y=None, encode_target=False):
        '''
        [Input]
        data:
            act or act_target: (B, T, J, d_a) # slide, hinge, and global action values

            slide_mask: (B, T, J) # True if slide
            hinge_mask: (B, T, J) # True if hinge
            global_mask: (B, T, J) # True if global

            act_mask: (B, T, J) # True if actuable
            morph_mask: (B, J, J) # (i, j) is True if i-th and j-th joints are morphologically connected
            m_idx: (B,) # morphology index

        [Output]
        y: (B, T, J, d_h) # action tokens
        '''
        m_idx = data['m_idx']
        act_mask, morph_mask = data['act_mask'], data['morph_mask']
        if encode_target:
            act = data['act_target']
        else:
            act = data['act']

        # get joint embedding
        slide_mask, hinge_mask, global_mask = data['slide_mask'], data['hinge_mask'], data['global_mask']
        y = self.act_embedding(act, slide_mask, hinge_mask, global_mask, m_idx) # (B, T, J, d)

        # get actuator embedding
        actuator_embedding = self.actuator_embedding(act_mask[..., None].to(dtype=y.dtype))

        T = y.shape[1]

        # add actuator embedding
        y = y + actuator_embedding

        # get position embedding
        position_embedding = torch.zeros_like(y)
        for b, m_idx_ in enumerate(m_idx):
            num_joint = self.num_joint_list[m_idx_.item()]
            position_embedding[b, :, :num_joint] = self.position_embedding[m_idx_.item()].repeat(T, 1, 1)

        # add spatial embedding
        y = y + position_embedding
        
        # encode morphology in spatial axis
        y = rearrange(y, 'B T J d -> (B T) J d')
        morph_attn_mask = repeat(morph_mask, 'B J1 J2 -> (B T) J1 J2', T=T)
        m_idx = repeat(m_idx, 'B -> (B T)', T=T)
        y = self.act_morphology_encoder(y, attn_mask=morph_attn_mask, t_idx=m_idx)
        y = rearrange(y, '(B T) J d -> B T J d', T=T)

        return y


class ControlTransformer(BaseModule):
    def __init__(self, config, num_tasks):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, config.history_size, config.hidden_dim))
        self.task_encoder = Transformer(config.n_task_blocks, config.n_attn_heads, config.hidden_dim,
                                        n_tsparam=num_tasks, lora_rank=config.task_lora_rank,
                                        attn_drop=config.drop_rate, drop=config.drop_rate, drop_path=config.drop_rate,
                                        layerscale=config.task_layerscale)
    
    def task_specific_parameters(self):
        return self.task_encoder.task_specific_parameters()

    def forward(self, data, x, y=None, is_causal=True, encode_actuable_only=False, **kwargs):
        '''
        [Input]
        data:
            task_mask: (B, J, T, T) # (i, j) is True if i-th and j-th time steps are task-related
            t_idx: (B,) # task index
        x: (B, T, J, d) # state tokens
        y: (B, T, J, d) # action tokens -- optional (else None)

        [Output]
        x: (B, T, J, d) # state tokens
        '''
        task_mask, t_idx = data['task_mask'], data['t_idx']
        B, T, J = x.shape[:3]
        position_embedding = self.position_embedding[:, :T]

        # encode task in temporal axis
        x = rearrange(x, 'B T J d -> (B J) T d')
        if encode_actuable_only:
            x_out = torch.zeros_like(x)
            act_mask = rearrange(data['act_mask'], 'B T J -> (B J) T')[:, 0]
            x = x[act_mask]

        x = x + position_embedding

        if y is not None:
            y = rearrange(y, 'B T J d -> (B J) T d')
            if encode_actuable_only:
                y = y[act_mask]

            y = y + position_embedding # add temporal embedding
            x = torch.cat([y, x], dim=1) # action comes first
            x = rearrange(x, '(B J) (n T) d -> (B J) (T n) d', J=J, n=2) # interleave action and observation
            task_mask = repeat(task_mask, 'B J T1 T2 -> B J (T1 n1) (T2 n2)', n1=2, n2=2)

        task_attn_mask = rearrange(task_mask, 'B J T1 T2 -> (B J) T1 T2')
        t_idx = repeat(t_idx, 'B -> (B J)', J=J)
        if encode_actuable_only:
            task_attn_mask = task_attn_mask[act_mask]
            t_idx = t_idx[act_mask]
        x = self.task_encoder(x, attn_mask=task_attn_mask, t_idx=t_idx, is_causal=is_causal, **kwargs)

        if y is not None:
            x = x[:, 1::2]

        if encode_actuable_only:
            x_out[act_mask] = x.to(dtype=x_out.dtype)
            x = x_out

        x = rearrange(x, '(B J) T d -> B T J d', J=J)

        return x


class MorphTaskTransformer(nn.Module):
    def __init__(self, config, num_joint_list, num_global_list, num_tasks):
        super().__init__()
        self.obs_morphology_transformer = ObsMorphologyTransformer(config, num_joint_list, num_global_list)
        self.act_morphology_transformer = ActMorphologyTransformer(config, num_joint_list, num_global_list)
        self.task_transformer = ControlTransformer(config, num_tasks)
        self.act_head = nn.Sequential(nn.Linear(config.hidden_dim, 1), nn.Tanh())

    def morphology_specific_parameters(self):
        for p in self.obs_morphology_transformer.morphology_specific_parameters():
            yield p
        for p in self.act_morphology_transformer.morphology_specific_parameters():
            yield p
    
    def morphology_specific_parameter_names(self):
        morphology_specific_parameters = [id(p) for p in self.morphology_specific_parameters()]
        names = []
        for n, p in self.named_parameters():
            if id(p) in morphology_specific_parameters:
                names.append(n)
        return names
    
    def task_specific_parameters(self):
        return self.task_transformer.task_specific_parameters()
    
    def task_specific_parameter_names(self):
        task_specific_parameters = [id(p) for p in self.task_specific_parameters()]
        names = []
        for n, p in self.named_parameters():
            if id(p) in task_specific_parameters:
                names.append(n)
        return names
   
    def forward(self, data):
        '''
        [Input]
        data:
            obs : (B, T, J, d_o) # slide, hinge, and global observations
            act: (B, T, J, d_a) # slide, hinge, and global action values

            slide_mask: (B, T, J) # True if slide
            hinge_mask: (B, T, J) # True if hinge
            global_mask: (B, T, J) # True if global

            act_mask: (B, T, J) # True if actuable
            morph_mask: (B, J, J) # (i, j) is True if i-th and j-th joints are morphologically connected
            task_mask: (B, J, T, T) # (i, j) is True if i-th and j-th time steps are task-related

            m_idx: (B,) # morphology index
            t_idx: (B,) # task index

        [Output]
        pred:
            act_pred: (B, T, J, d_a) # predicted action values
            act_target: (B, T, J, d_a) # target action values -- optional (else None)
        '''
        # morphology encoding
        x = self.obs_morphology_transformer(data)
        y = self.act_morphology_transformer(data)

        # task encoding
        x = self.task_transformer(data, x, y, encode_actuable_only=True)

        # decode action
        x = self.act_head(x)

        pred = {
            'act_pred': x,
        }

        return pred


class CrossAttention(nn.Module):
    '''
    Multi-Head Cross-Attention layer for Matching
    '''
    def __init__(self, dim, num_heads=4, temperature=-1, act_fn=nn.GELU, dr=0.1, pre_ln=True, ln=True, residual=True):
        super().__init__()
        # heads and temperature
        self.num_heads = num_heads
        self.dim_split = dim // num_heads
        if temperature > 0:
            self.temperature = temperature
        else:
            self.temperature = math.sqrt(dim)
        self.residual = residual
        
        # projection layers
        self.fc_q = nn.Linear(dim, dim, bias=False)
        self.fc_k = nn.Linear(dim, dim, bias=False)
        self.fc_v = nn.Linear(dim, dim, bias=False)
        self.fc_o = nn.Linear(dim, dim, bias=False)
        
        # nonlinear activation and dropout
        self.activation = act_fn()
        self.attn_dropout = nn.Dropout(dr)
        
        # layernorm layers
        if pre_ln:
            self.pre_ln = nn.LayerNorm(dim)
        else:
            self.pre_ln = nn.Identity()
        self.ln = nn.LayerNorm(dim) if ln else nn.Identity()

    def forward(self, Q, K, V, mask=None):
        # pre-layer normalization
        Q = self.pre_ln(Q)
        K = self.pre_ln(K)
        
        # lienar projection
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        # split into multiple heads
        Q_ = torch.cat(Q.split(self.dim_split, 2), 0)
        K_ = torch.cat(K.split(self.dim_split, 2), 0)
        V_ = torch.cat(V.split(self.dim_split, 2), 0)
        
        # scaled dot-product attention with mask and dropout
        L = Q_.bmm(K_.transpose(1, 2)) / self.temperature
        L = L.clip(-1e4, 1e4)
        
        # mask
        if mask is not None:
            L = L.masked_fill(~mask, -float('inf'))
            
        A = L.softmax(dim=2)
        if mask is not None:
            A.masked_fill(~mask, 0)
        A = self.attn_dropout(A)
        
        # apply attention to values
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        
        # layer normalization
        O = self.ln(O)
        
        # residual connection with non-linearity
        if self.residual:
            O = O + self.activation(self.fc_o(O))
        else:
            O = self.fc_o(O)
            
        return O


class MetaController(BaseModule):
    def __init__(self, config, num_joint_list, num_global_list, num_tasks):
        super().__init__()
        self.obs_morphology_transformer = ObsMorphologyTransformer(config, num_joint_list, num_global_list)
        self.state_task_transformer = ControlTransformer(config, num_tasks)

        # legacy 
        config_action_encoder_decoder = copy.deepcopy(config)
        config_action_encoder_decoder.task_lora_rank = 0 
        
        self.action_encoder = ControlTransformer(config_action_encoder_decoder, 0)
        self.action_decoder = ControlTransformer(config_action_encoder_decoder, 0)
        self.matching_module = CrossAttention(config.hidden_dim, num_heads=config.n_attn_heads)
        
        self.act_tokenizer = nn.Linear(1, config.hidden_dim)
        self.act_head = nn.Sequential(nn.Linear(config.hidden_dim, 1), nn.Tanh())
        self.reset_support()

    def morphology_specific_parameters(self):
        for p in self.obs_morphology_transformer.morphology_specific_parameters():
            yield p
   
    def task_specific_parameters(self):
        for param in self.state_task_transformer.task_specific_parameters():
            yield param
    
    def reset_support(self):
        self.K = self.V = None

    def encode_support(self, data_S):
        # merge batch and instance dimensions
        B, N, T = data_S['obs'].shape[:3]
        data_S = {k: v.view(B * N, *v.shape[2:]) for k, v in data_S.items()}

        # morphology encoding
        x_S = self.obs_morphology_transformer(data_S)
        y_S = self.act_tokenizer(data_S['act_target'])

        # task encoding
        x_S = self.state_task_transformer(data_S, x_S)
        y_S = self.action_encoder(data_S, y_S)

        # reshape data for matching
        self.K = rearrange(x_S, '(B N) T J d -> (B J) (N T) d', N=N)
        self.V = rearrange(y_S, '(B N) T J d -> (B J) (N T) d', N=N)

    def predict_query(self, data_Q):
        assert self.K is not None and self.V is not None, 'Support data must be encoded first'

        # merge batch and instance dimensions
        B, M, T = data_Q['obs'].shape[:3]
        data_Q = {k: v.view(B * M, *v.shape[2:]) for k, v in data_Q.items()}

        # morphology encoding
        x_Q = self.obs_morphology_transformer(data_Q)

        # task encoding
        x_Q = self.state_task_transformer(data_Q, x_Q)

        # reshape data for matching
        Q = rearrange(x_Q, '(B M) T J d -> (B J) (M T) d', M=M)

        # apply matching
        O = self.matching_module(Q, self.K, self.V)

        # reshape data for decoding
        y_Q = rearrange(O, '(B J) (M T) d -> (B M) T J d', B=B, M=M)

        # decode action
        y_Q = self.action_decoder(data_Q, y_Q, encode_actuable_only=True)
        y_Q = self.act_head(y_Q)

        # reshape back to the original shape
        y_Q = rearrange(y_Q, '(B M) T J d -> B M T J d', B=B, M=M)

        pred = {
            'act_pred': y_Q,
        }
        return pred
    
    def forward(self, data, *args, **kwargs):
        # merge batch and instance dimensions
        data_S, data_Q = data['data_S'], data['data_Q']

        B, N, T = data_S['obs'].shape[:3]
        data_S = {k: v.view(B * N, *v.shape[2:]) for k, v in data_S.items()}
        M = data_Q['obs'].shape[1]
        data_Q = {k: v.view(B * M, *v.shape[2:]) for k, v in data_Q.items()}

        data = {k: torch.cat([data_S[k], data_Q[k]], dim=0) for k in data_S}

        # morphology encoding
        x = self.obs_morphology_transformer(data)
        y_S = self.act_tokenizer(data_S['act_target'])

        # task encoding
        x = self.state_task_transformer(data, x, encode_actuable_only=True)
        x_S, x_Q = x[:B * N], x[B * N:]
        y_S = self.action_encoder(data_S, y_S, encode_actuable_only=True)

        # reshape data for matching
        Q = rearrange(x_Q, '(B M) T J d -> (B J) (M T) d', M=M)
        K = rearrange(x_S, '(B N) T J d -> (B J) (N T) d', N=N)
        V = rearrange(y_S, '(B N) T J d -> (B J) (N T) d', N=N)

        # apply matching
        O = self.matching_module(Q, K, V)

        # reshape data for decoding
        y_Q = rearrange(O, '(B J) (M T) d -> (B M) T J d', B=B, M=M)

        # decode action
        y_Q = self.action_decoder(data_Q, y_Q, encode_actuable_only=True)
        y_Q = self.act_head(y_Q)

        # reshape back to the original shape
        y_Q = rearrange(y_Q, '(B M) T J d -> B M T J d', B=B, M=M)

        pred = {
            'act_pred': y_Q,
        }

        return pred

