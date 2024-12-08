import os
os.environ['MUJOCO_GL'] = 'egl'

import pytorch_lightning as pl
import torch
import warnings

from args import parse_args
from train.train_utils import configure_experiment, load_model, print_configs
from lightning_fabric.utilities.seed import seed_everything

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    torch.set_num_threads(1)
    warnings.filterwarnings('ignore')
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
    # parse args
    config = parse_args()
    seed_everything(config.seed, workers=True)
    IS_RANK_ZERO = int(os.environ.get('LOCAL_RANK', 0)) == 0
    
    # load model
    model, config, ckpt_path, mt_config, ft_config, ts_config = load_model(config, verbose=IS_RANK_ZERO)

    # environmental settings
    logger, log_dir, save_dir, callbacks, profiler, precision, strategy, plugins = configure_experiment(config, model, is_rank_zero=IS_RANK_ZERO)
    model.config.result_dir = save_dir
    
    # print configs
    if IS_RANK_ZERO:
        print_configs(config, model, mt_config, ft_config, ts_config)

    # set max epochs
    if (not config.no_eval) and config.stage <= 1:
        max_epochs = config.n_steps // config.val_iter
    else:
        max_epochs = 1

    # create pytorch lightning trainer.
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir=save_dir,
        accelerator='gpu',
        max_epochs=max_epochs,
        log_every_n_steps=-1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        benchmark=True,
        devices=torch.cuda.device_count(),
        strategy=strategy,
        precision=precision,
        profiler=profiler,
        plugins=plugins,
        gradient_clip_val=config.gradient_clip_val,
    )

    # validation at start
    if (config.stage == 1 and not config.continue_mode) or config.stage == 0:
        trainer.validate(model, verbose=False)
        
    if config.stage == 2:
        trainer.test(model)
    else:
        trainer.fit(model, ckpt_path=ckpt_path)
