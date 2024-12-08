import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.distributed
import torchvision.transforms as T
import deepspeed
import os
from lightning_fabric.utilities.seed import seed_everything
from torchvision.utils import make_grid
from einops import rearrange
from pathlib import Path
from datasets.dmc import DMCEpisodicTrainDataset, DMC
from datasets.utils import to_device
from train.optim import get_optimizer
from train.visualize import color_frame
from train.learner import Learner 
from model.metacon import MetaController 

 
class LightningTrainWrapper(pl.LightningModule):
    def __init__(self, config, verbose=True):
        import warnings; warnings.filterwarnings('ignore')
        super().__init__()
        self.config = config
        self.verbose = verbose
        self.n_devices = torch.cuda.device_count()
        self.toten = T.ToTensor()

        self.model = self.create_model()
        self.save_hyperparameters()

    def create_learner(self):
        if self.config.no_eval:
            dset_size = self.config.n_steps*self.config.global_batch_size # whole iterations in a single epoch
        else:
            dset_size = self.config.val_iter*self.config.global_batch_size # chunk iterations in validation steps        

        if self.config.stage < 2:
            train_data = DMCEpisodicTrainDataset(
                sampling_shot=self.config.shot,
                window_size=self.config.window_size,
                stage=self.config.stage,
                split='train',
                demos=self.config.num_train_trajs,
                precision=self.config.precision,
                env_name=self.config.env_name,
                dset_size=dset_size,
                history_size=self.config.history_size,
                max_n_joints=self.config.max_n_joints,
            )
        else:
            train_data = None

        # create learner class
        learner = Learner(self, train_data)

        return learner

    def create_model(self):
        # load model.
        base_info = DMC.get_base_info()
        if self.config.stage == 0:
            num_joint_list = [base_info[m]['num_slide_token'] + base_info[m]['num_hinge_token'] + base_info[m]['num_global_token'] for m in base_info]
            num_global_list = [base_info[m]['num_global'] for m in base_info]
            num_tasks = len(DMC.ENVIRONMENTS)
        else:
            morphology = self.config.env_name.split('-')[0]
            num_joint_list = [base_info[morphology]['num_slide_token'] + base_info[morphology]['num_hinge_token'] + base_info[morphology]['num_global_token']]
            num_global_list = [base_info[morphology]['num_global']]
            num_tasks = 1

        # set number of tasks for bitfit
        model = MetaController(self.config, num_joint_list, num_global_list, num_tasks)
        
        return model

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        
        device_id = int(os.environ['CUDA_VISIBLE_DEVICES'].split(",")[self.local_rank]) 
        os.environ['EGL_DEVICE_ID'] = f"{device_id}"
        os.environ['MUJOCO_EGL_DEVICE_ID'] = f"{device_id}"

        # create learner
        self.learner = self.create_learner()

    def configure_optimizers(self):
        '''
        Prepare optimizer and lr scheduler.
        '''
        optimizer, self.lr_scheduler = get_optimizer(self.config, self.model)
        return optimizer
    
    def train_dataloader(self):
        '''
        Prepare training loader.
        '''
        import warnings; warnings.filterwarnings('ignore')
        train_loader = self.learner.get_train_loader()

        return train_loader
        
    def val_dataloader(self):
        '''
        Prepare validation loaders.
        '''
        if not self.config.no_eval:
            val_loaders_list, loader_tags = self.learner.get_val_loaders()
            self.valid_tags = loader_tags
            self.valid_tasks = sum([[(loader_tag, key) for key in list(val_loader.keys())]
                                     for val_loader, loader_tag in zip(val_loaders_list, loader_tags)], [])
            return sum([list(val_loader.values()) for val_loader in val_loaders_list], [])
        
    def test_dataloader(self):
        '''
        Prepare validation loaders.
        '''
        test_loader, tag = self.learner.get_val_loaders()        
        self.test_environment =  list(test_loader[0].keys())[0]       
        return test_loader[0][self.test_environment]

    def forward(self, data, *args, **kwargs):
        '''
        Forward data to model.
        '''
        # forward data
        
        pred = self.model.forward(data, *args, **kwargs)
        return pred

    def on_train_start(self, *args, **kwargs):
        super().on_train_start(*args, **kwargs)
        import warnings; warnings.filterwarnings('ignore')
        torch.backends.cuda.matmul.allow_tf32 = True
        if self.n_devices > 1:
            self.trainer.train_dataloader.sampler.shuffle = False
            seed_everything(self.config.seed + self.local_rank, workers=True)
        self.barrier()

    def barrier(self):
        # synchronize at this point
        if self.config.strategy == 'deepspeed':
            deepspeed.comm.barrier()
        elif self.config.strategy == 'ddp':
            torch.distributed.barrier()

    def training_step(self, batch, batch_idx):
        '''
        A single training iteration.
        '''
        # forward model and compute loss.
        pred = self.model(batch)
        loss, loss_values = self.learner.compute_loss(pred, batch)

        # schedule learning rate.
        self.lr_scheduler.step(self.global_step + self.config.schedule_from)
    
        # create log dict
        log_dict = {
            f'training/lr{self.learner.tag}': self.lr_scheduler.lr,
            'step': self.global_step,
        }

        for key, value in loss_values.items():
            if value is not None:
                log_dict[f'training/{key}{self.learner.tag}'] = value

        log_dict = {k: log_dict[k] for k in sorted(log_dict)}

        # tensorboard logging
        self.log_dict(
            log_dict,
            logger=True,
            on_step=True,
            rank_zero_only=True
        )

        return loss
    
    def play(self, env, initial_state, environment, n_steps=-1, render=True, render_size=(64, 64)):        
        # initialize environment
        obs = env.reset(initial_state=initial_state)
        self.model.encode_support(to_device(env.support_data, device=torch.device("cuda")))

        steps = 0
        done = False
        if render:
            vis = [env.render(render_size)]
        traj_attn = []

        while not done:
            data = to_device(obs, device=torch.device("cuda"))
            # forward model
            out = self.model.predict_query(data)
            act = out['act_pred']
            
            # step environment
            obs, reward, done, _  = env.step(act)
            
            if render:
                frame = env.render(render_size)
                if reward < 1.0:
                    color_frame(frame, (1., 0., 0.))
                else:
                    color_frame(frame, (0., 1., 0.))
                vis.append(frame)
            
            steps += 1
            if n_steps > 0 and steps == n_steps:
                break

        self.model.reset_support()
            
        if self.config.stage < 1:
            metrics = {'rewards': torch.tensor(env.full_history['rewards']).sum()}
            if render:
                metrics['vis'] = torch.stack(vis)
            
            return metrics
        
        traj = {'states': torch.stack(env.full_history['states']), # (T + 1, d_s)
                'observations': torch.stack(env.full_history['obs']), # (T + 1, d_o)
                'actions': torch.stack(env.full_history['act']), # (T + 1, d_a)
                'rewards': torch.tensor(env.full_history['rewards'])} # (T + 1, )
        
        if render:
            traj['vis'] = torch.stack(vis)
        return traj
    

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def inference_play(self, env, batch_idx, dataloader_idx=0, render=True):
        '''
        Evaluate few-shot performance on validation dataset.
        '''
        if (batch_idx % self.n_devices) == self.local_rank:
            # inference
            assert env.initial_states is not None
            initial_state = env.initial_states[batch_idx]

            if self.config.stage < 2:
                environment = self.valid_tasks[dataloader_idx][1]
            else:
                environment = self.test_environment
            traj = self.play(env, initial_state=initial_state, environment=environment, render=render)
        else:
            traj = None
        
        return traj
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.inference_play(batch, batch_idx, dataloader_idx, render=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self.inference_play(batch, batch_idx)

    def all_gather_dict(self, x):
        if self.n_devices > 1:
            return {k: rearrange(self.all_gather(v), 'G B ... -> (B G) ...') for k, v in x.items()}
        else:
            return x

    def validation_epoch_end(self, validation_step_outputs):
        '''
        Aggregate losses of all validation datasets and log them into tensorboard.
        '''
        log_dict = {'step': self.global_step}
        avg_metric = {}
        key_avg_metric = {}
        for valid_tag in self.valid_tags:
            avg_metric[valid_tag] = []

        if len(self.valid_tasks) == 1:
            validation_step_outputs = [validation_step_outputs]
        for tag_idx, metrics in enumerate(validation_step_outputs):
            valid_tag, (morphology, task) = self.valid_tasks[tag_idx]

            metrics = {k: torch.stack([metric[k] for metric in metrics if metric is not None]) for k in metrics[0].keys()}

            # pop and gather visualizations
            if valid_tag in ['mtrain_test', 'mtest_test']:
                if self.config.stage == 1:
                    self.save_traj(metrics, (morphology, task), ptf=f"step:{self.global_step}") 
                    metrics.pop("states")
                    metrics.pop("observations")
                    metrics.pop("actions")
                    metrics.pop('vis')
                    metrics['rewards'] = metrics['rewards'].sum(-1)
                else:
                    vis = metrics.pop('vis')
                    if self.n_devices > 1:
                        vis = self.gather_play_results(vis)
                    if self.local_rank == 0:
                        vis = torch.stack([make_grid(vis[:, t], nrow=10) for t in range(vis.shape[1])])
                        save_dir = os.path.join(self.logger.save_dir, "videos", f"{morphology}-{task}")
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        self.save_gif_from_tensor(vis, os.path.join(save_dir, f'global_step:{self.global_step:06d}.gif'))  
            
            # gather metrics
            if self.n_devices > 1:
                if valid_tag in ['mtrain_test', 'mtest_test']:
                    metrics['rewards'] = self.gather_play_results(metrics['rewards'])
                else:
                    metrics = self.all_gather_dict(metrics)
            metrics = {k: v.float().nanmean(dim=0) for k, v in metrics.items()}

            # log task-specific errors
            self.learner.log_metrics(morphology, task, metrics, log_dict, avg_metric, key_avg_metric, valid_tag)
            
        # log task-averaged error
        for valid_tag in self.valid_tags:
            avg_metric_total = sum(avg_metric[valid_tag]) / len(avg_metric[valid_tag])
            log_dict[f'summary/{valid_tag}'] = avg_metric_total

        for key in key_avg_metric:
            key_avg_metric_total = sum(key_avg_metric[key]) / len(key_avg_metric[key])
            log_dict[key] = key_avg_metric_total

        log_dict = {k: log_dict[k] for k in sorted(log_dict)}

        # tensorboard logging
        # import warnings; warnings.filterwarnings('ignore')
        self.log_dict(
            log_dict,
            logger=True,
            rank_zero_only=True
        )

    def save_gif_from_tensor(self, vid_tensor, save_dir, fps=30):
        # vid_tensor: (T, C, H, W)
        topil = T.ToPILImage() 
        imgs = [topil(vid_tensor_).convert('RGB') for vid_tensor_ in vid_tensor]
        imgs[0].save(fp=save_dir, append_images=imgs[1:], save_all=True, duration=1000/fps, loop=0)

    def gather_play_results(self, play_results):
        max_size_per_device = self.config.num_eval_episodes // self.n_devices
        if self.config.num_eval_episodes % self.n_devices != 0:
            max_size_per_device += 1
        
        pad_dims = []
        for _ in range(play_results.ndim - 1):
            pad_dims.extend([0, 0])
        pad_dims.extend([0, max_size_per_device - play_results.size(0)])

        play_results = F.pad(play_results, pad_dims, value=torch.nan)
        play_results = rearrange(self.all_gather(play_results), 'G B ... -> (B G) ...')
        play_results = play_results[~torch.isnan(play_results).any(dim=[i for i in range(1, play_results.ndim)])]

        return play_results

    def test_epoch_end(self, test_step_outputs):
        '''
        Aggregate losses of all validation datasets and log them into tensorboard.
        '''
        self.save_traj(test_step_outputs, self.test_environment, log_to_tb=True)
    
    def save_traj(self, test_step_outputs , environment, ptf='', log_to_tb=False):
        if isinstance(test_step_outputs, dict):
            all_states = test_step_outputs['states']
            all_observations = test_step_outputs['observations']
            all_actions = test_step_outputs['actions']
            all_rewards = test_step_outputs['rewards']
            all_vis = test_step_outputs['vis']
        else:
            all_states = torch.stack([traj['states'] for traj in test_step_outputs if traj is not None]) # (B, T, d_s)
            all_observations = torch.stack([traj['observations'] for traj in test_step_outputs if traj is not None]) # (B, T, d_o)
            all_actions = torch.stack([traj['actions'] for traj in test_step_outputs if traj is not None]) # (B, T, d_a)
            all_rewards = torch.stack([traj['rewards'] for traj in test_step_outputs if traj is not None]) # (B, T)
            all_vis = torch.stack([traj['vis'] for traj in test_step_outputs if traj is not None]) # (B, T, C, H, W)

        if self.n_devices > 1:
            all_states = self.gather_play_results(all_states)
            all_observations = self.gather_play_results(all_observations)
            all_actions = self.gather_play_results(all_actions)
            all_rewards = self.gather_play_results(all_rewards)
            all_vis = self.gather_play_results(all_vis)

        if self.local_rank == 0:
            result_postfix = f"{self.config.result_postfix}{ptf}"

            # save trajectories
            torch.save(all_states, os.path.join(self.config.result_dir, f"states{result_postfix}.pth"))
            torch.save(all_observations, os.path.join(self.config.result_dir, f"observations{result_postfix}.pth"))
            torch.save(all_actions, os.path.join(self.config.result_dir, f"actions{result_postfix}.pth"))
            torch.save(all_rewards, os.path.join(self.config.result_dir, f"rewards{result_postfix}.pth"))

            # save rendered gifs
            topil = T.ToPILImage()
            vis = torch.stack([make_grid(all_vis[:, t], nrow=10) for t in range(all_vis.shape[1])])[None] # (1, T, C, H, W)
            images = [topil(vis[0, t]) for t in range(vis.size(1))]
            im = images[0]
            gif_path = os.path.join(self.config.result_dir, f"vis{result_postfix}.gif")
            im.save(gif_path, save_all=True, append_images=images[1:], loop=0, duration=1000/30)

            # log to tensorboard
            if log_to_tb:
                avg_rewards = all_rewards.cumsum(dim=1).mean(0)
                for step, avg_reward in enumerate(avg_rewards):
                    tag = f'mtest_play/{environment[0]}-{environment[1]}{result_postfix}'
                    self.logger.experiment.add_scalar(tag, avg_reward, step+1)
                tag = f'mtest_play/vis:{environment[0]}-{environment[1]}{result_postfix}'
                self.logger.experiment.add_video(tag, vis, self.global_step, fps=30)
