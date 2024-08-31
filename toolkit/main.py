import numpy as np
import random
import torch
torch.set_float32_matmul_precision('high') #highest,high,medium
torch.backends.cudnn.benchmark = True # # Accelate training
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*exists and is not empty.")
warnings.filterwarnings("ignore", ".*logging on epoch level in distributed setting*")
warnings.filterwarnings("ignore", ".*RoPE2D, using a slow pytorch version instead")

import sys
sys.path.append('..')
from toolkit.data_loader.dataset_function import generate_file_lists
from toolkit.data_loader.dataloader import prepare_dataset,dataloader_customization,optimizer_customization
from toolkit.args.args_default import get_opts
from toolkit.torch_lightning.pl_modules.train import trainer_func
from toolkit.torch_lightning.pl_modules.evaluate import evaluate_func

# For reproducibility
# torch.manual_seed(192)
# torch.cuda.manual_seed(192)
# np.random.seed(192)
# random.seed(192)

def inference(hparams):

    ###################### 
    # prepare dataloader # 
    ###################### 
    hparams,aug_config,valid_aug_config = dataloader_customization(hparams)   
    file_path_dic = generate_file_lists(dataset = hparams.dataset,if_train=hparams.dataset_type=='train',method='gt',save_method=hparams.save_name)
    dataset,n_img = prepare_dataset(file_path_dic,aug_config=aug_config)
    hparams = optimizer_customization(hparams,n_img)
    if hparams.if_use_valid:
        valid_file_path_dic = generate_file_lists(dataset = hparams.val_dataset,if_train=hparams.val_dataset_type=='train',method='gt',save_method='delete') 
        valid_dataset,_ = prepare_dataset(valid_file_path_dic,aug_config=valid_aug_config)
    else:
        valid_dataset = None

    ############################################
    # load model and select inference function #
    ############################################
    if hparams.inference_type == 'train':
        inference = trainer_func
    elif hparams.inference_type == 'evaluate':
        inference = evaluate_func
    ##########################
    # run inference function #
    ##########################
    if 'train' in hparams.inference_type:
        inference(hparams,dataset,valid_dataset)
    elif 'evaluate' in hparams.inference_type:
        test_dataloader = DataLoader(dataset, batch_size= 1, shuffle= False, num_workers= 1, drop_last=False)
        inference(hparams,test_dataloader)

if __name__ == '__main__':    
    
    hparams = get_opts()
    hparams.devices = [0]
    
    # ViTASIGEV evaluate
    hparams.inference_type = 'evaluate'
    hparams.dataset = 'KITTI2015'   
    hparams.network = 'ViTASIGEV' 
    hparams.save_name = 'ViTASIGEV_benchmark'
    hparams.ckpt_path = 'models/ViTASIGEV/KITTI.pth'
    hparams.ViTAS_dic['VFM_type'] = 'DINOv2'
    inference(hparams)
    
    # ViTASIGEV train
    hparams.inference_type = 'train' 
    hparams.if_use_valid = True 
    hparams.dataset = 'KITTI2015'   
    hparams.val_dataset = 'KITTI2012'   
    hparams.network = 'ViTASIGEV' 
    hparams.save_name = 'ViTASIGEV_benchmark'
    hparams.ckpt_path = 'models/ViTASIGEV/KITTI.pth'
    hparams.ViTAS_dic['ViTAS_fuse'] = 'PAFM' # ['SDFA','PAFM','VFM']
    hparams.batch_size = 2
    hparams.num_workers = 2
    hparams.epoch_size = 100
    inference(hparams)
    
    
    