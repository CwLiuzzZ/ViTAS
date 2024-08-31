from pytorch_lightning import Trainer
from toolkit.torch_lightning.pl_modules.ViTASIGEV import ViTASIGEV
from toolkit.torch_lightning.lightning_function import hparam_resume

def evaluate_func(hparams,test_dataloader):
        
    if hparams.network == 'ViTASIGEV':
        system = ViTASIGEV
    else:
        raise ValueError('Invalid network type')

    # if not hparams.hparams_dir is None:
    #     with open(hparams.hparams_dir, 'r') as f:
    #         hparams = yaml.unsafe_load(f)['hparams']
    
    
    if hparams.ckpt_path is not None:
        if hparams.resume or hparams.resume_model:
            hparams = hparam_resume(hparams,evaluate=True)
        print('load lightning pre-trained model from {}'.format(hparams.ckpt_path))
        # system = system.load_from_checkpoint(hparams.ckpt_path,**{'hparams':hparams})
        system = system.load_from_checkpoint(hparams.ckpt_path,**{'hparams':hparams},map_location='cpu') # map_location={'cuda:1': 'cuda:0'}
    else:
        system = system(hparams = hparams)

    # set up trainer
    trainer = Trainer(accelerator='gpu',
            devices = hparams.devices,)
    
    # time0=time.time()
    trainer.test(system, test_dataloader)
    # print((time.time()-time0)/97)
