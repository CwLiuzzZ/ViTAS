from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.strategies import DDPStrategy
from toolkit.torch_lightning.pl_modules.ViTASIGEV import ViTASIGEV
from toolkit.torch_lightning.data_modules.custom import CustomDataModule
from toolkit.torch_lightning.lightning_function import hparam_resume
import torch

def trainer_func(hparams,train_dataset,valid_dataset=None):    
    
    if hparams.network == 'ViTASIGEV':
        system = ViTASIGEV
    else:
        raise ValueError('Invalid network type')

    # pl data module
    dm = CustomDataModule(hparams,train_dataset,valid_dataset)

    # pl logger
    logger = TensorBoardLogger(
        save_dir="torch_lightning/ckpts",
        name=hparams.save_name,
        default_hp_metric=False
    )
    print('=> log save in: ckpts/{}/version_{:d}'.format(hparams.save_name, logger.version))

    # save checkpoints
    ckpt_dir = 'torch_lightning/ckpts/{}/version_{:d}'.format(
        hparams.save_name, logger.version)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                        #   filename='{epoch}-{loss_epoch:.4f}',
                                          filename='{epoch}-{loss_epoch:.4f}-{valid_epoch_EPE:.4f}-{valid_epoch_D1:.4f}',
                                          monitor = 'valid_epoch_EPE', # loss_epoch,valid_epoch_EPE
                                          mode='min',
                                          save_last=False,
                                          save_weights_only=True,
                                          every_n_epochs = 1,
                                          save_top_k=8, # -1: save for every epoch
                                          )


    # restore from previous checkpoints
    if hparams.ckpt_path is not None:
        if hparams.resume or hparams.resume_model:
            hparams = hparam_resume(hparams)
            hparams.this_epoch = (int(hparams.ckpt_path.split('epoch=')[1].split('-')[0]))*hparams.epoch_steps
        print('load lightning pre-trained model from {}'.format(hparams.ckpt_path))
        # system = system.load_from_checkpoint(hparams.ckpt_path,**{'hparams':hparams})
        # system = system.load_from_checkpoint(hparams.ckpt_path,**{'hparams':hparams},map_location='cuda:0')
        system = system.load_from_checkpoint(hparams.ckpt_path,**{'hparams':hparams},map_location='cpu')
    else:
        system = system(hparams = hparams)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    gpu_count = torch.cuda.device_count()
    if len(hparams.devices) == 1:
        strategy = 'auto' # auto
    elif len(hparams.devices) > 1:
        strategy = 'ddp_find_unused_parameters_true'
    else:
        raise NotImplementedError('none GPU detected')
    print('{} GPUs detected, use {} strategy and {} devices'.format(gpu_count,strategy,hparams.devices))

    # set up trainer
    trainer = Trainer(
        accelerator='gpu',
        max_epochs=hparams.epoch_size,
        # limit_val_batches=200 if hparams.val_mode == 'photo' else 1.0,
        # limit_val_batches=1,
        num_sanity_val_steps = 2,
        log_every_n_steps = 50, # default = 50
        val_check_interval = 1.0, # validate every n training epoch, default=1.0
        callbacks=[checkpoint_callback,lr_monitor],
        devices = hparams.devices,
        strategy = strategy,
        logger=logger,
        benchmark=True,
        # reload_dataloaders_every_n_epochs = 5,
    )

    trainer.fit(system, dm)

