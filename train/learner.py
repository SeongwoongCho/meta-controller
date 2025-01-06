import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets.dmc import DMC, SupportLoader, DMCEpisodicTestEnvironment
from einops import rearrange


class Learner:
    def __init__(self, trainer, train_data):
        self.config = trainer.config
        self.trainer = trainer
        self.local_rank = self.trainer.local_rank
        self.n_devices = self.trainer.n_devices
        self.verbose = self.trainer.verbose

        self.topil = T.ToPILImage()
        self.tag = ''
        self.train_data = train_data
    
    def get_train_loader(self):
        assert self.config.stage < 2
        batch_size = self.config.global_batch_size // self.n_devices
        train_loader = DataLoader(self.train_data, batch_size=batch_size,
                                  shuffle=(self.n_devices == 1), pin_memory=True, drop_last=True, num_workers=self.config.num_workers)
        return train_loader

    def get_val_loaders(self):
        stage_prefix = 'mtrain' if self.config.stage == 0 else 'mtest'
        
        # Prepare Support Loader
        self.support_loader = SupportLoader(
            window_size=self.config.window_size,
            max_n_joints=self.config.max_n_joints,
            stage=self.config.stage,
            split='train',
            demos=self.config.num_train_trajs,
            precision=self.config.precision,
            env_name=self.config.env_name,
            history_size=self.config.history_size,
        )
        
        loaders_list = []
        loader_tags = []

        if self.config.stage == 0:
            eval_environments = DMC.MTRAIN_EVAL_ENVIRONMENTS
        else:
            eval_environments = [tuple(self.config.env_name.split('-'))]
                    
        loaders = {}
        for morphology, task in eval_environments: 
            valid_states = torch.load(f"DMCDATA/VALIDATIONSTATES/{morphology}_{task}_states.pt")
            valid_rawobs = torch.load(f"DMCDATA/VALIDATIONSTATES/{morphology}_{task}_rawobs.pt")
            
            initial_states = [] 
            for i in range(self.config.num_eval_episodes):
                initial_state = (valid_states[i], {key: valid_rawobs[key][i] for key in valid_rawobs.keys()})
                initial_states.append(initial_state)

            eval_env = DMCEpisodicTestEnvironment(
                support_loader=self.support_loader,
                env_name=f"{morphology}-{task}",
                stage=self.config.stage,
                initial_states=initial_states,
                history_size=self.config.history_size,
                precision=self.config.precision,
            )
            # eval_env.set_support_data(demos=3 if self.config.stage == 0 else self.config.demo)
            eval_env.set_support_data(self.config.num_eval_trajs)

            loaders[(morphology, task)] = eval_env
        
        loaders_list.append(loaders)
        loader_tags.append(f'{stage_prefix}_test')
            
        return loaders_list, loader_tags
    
    def log_metrics(self, morphology, task, metric, log_dict, avg_metric, key_avg_metric, valid_tag):
        '''
        log evaluation metrics
        '''
        for key in metric:
            if key in ['loss', 'rewards']:
                log_dict[f'{valid_tag}/{morphology}-{task}'] = metric[key]
            else:
                log_dict[f'{valid_tag}/{morphology}-{task}-{key}'] = metric[key]
            if f'{valid_tag}/{key}' not in key_avg_metric:
                key_avg_metric[f'{valid_tag}/{key}'] = []
            key_avg_metric[f'{valid_tag}/{key}'].append(metric[key])

        if 'rewards' in metric:
            avg_metric[valid_tag].append(metric['rewards'])
        else:
            avg_metric[valid_tag].append(metric['loss'])
    
    def compute_loss(self, pred, data):
        '''
        [Input]
        data:
            act: (B, T, J, d_a) or (B, N, T, J, d_a) # slide, hinge, and global action values
            act_mask: (B, T, J) or (B, N, T, J) or (B, T, C) # True if actuable
            task_mask: (B, J, T, T) or (B, N, J, T, T) # (i, j) is True if i-th and j-th time steps are task-related

        [Output]
        loss: scalar # total loss
        loss_values:
            loss_act: scalar # action loss
        '''
        loss_values = {}
        loss = 0

        act_mask = data['data_Q']['act_mask']
        act_target = data['data_Q']['act_target']
        task_mask = data['data_Q']['task_mask']
        
        mask = act_mask[..., None].float() * rearrange(task_mask[..., 0, :].float(), '... j t -> ... t j 1') # (B, T, J, 1) or (B, N, T, J, 1)
        loss_act = torch.mean(mask * F.mse_loss(act_target, pred['act_pred'], reduction='none'), dim=-1).sum() / mask.sum()
        loss = loss + loss_act
        
        loss_values['loss_act'] = loss_act.detach()
        loss_values['loss'] = loss.detach()

        return loss, loss_values
