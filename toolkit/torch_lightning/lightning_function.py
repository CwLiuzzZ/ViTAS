import torch
import yaml
import argparse



def schedule_select(optimizer,hparams):
    if hparams.schedule == 'Cycle': # for large dataset
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=hparams.min_lr, max_lr=hparams.lr, step_size_up=int(hparams.epoch_steps/500 + 5), step_size_down=int(hparams.epoch_steps*0.4), cycle_momentum=False,mode='triangular2', last_epoch = hparams.this_epoch) # 20,6000;2,200
        print('lr_schedule Cycle: base_lr',hparams.min_lr,'; max_lr',hparams.lr)
    elif hparams.schedule == 'OneCycle': # for small dataset
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=hparams.lr,
            total_steps=hparams.num_steps + 50,
            pct_start=0.03,
            cycle_momentum=False,
            anneal_strategy='cos',
            last_epoch = hparams.this_epoch,
            # initial_lr = hparams.lr/25,
        )   
        print('lr_schedule OneCycle: max_lr',hparams.lr)
    lr_scheduler = {
        'scheduler': scheduler,
        'name': 'my_logging_lr',
        'interval':'step'
    }
    return lr_scheduler

def hparam_resume(hparams,evaluate = False):
    # if hparams.hparams_dir is None: # search for hparam file
    hparams.hparams_dir = '/'.join(hparams.ckpt_path.split('/')[:-1])+'/hparams.yaml'
    # print(hparams.hparams_dir)
    with open(hparams.hparams_dir, 'r') as f:
        hparams_ = yaml.unsafe_load(f)['hparams']
    
    # set exception
    if evaluate:
        exception_ = ['resume','resume_model','ViTAS_dic','devices','ckpt_path','inference_type','num_workers','dataset','dataset_type','if_use_valid','val_dataset','val_dataset_type','save_name']
    else:
        if not hparams.resume_model:  # resume all
            exception_ = ['resume','resume_model','ViTAS_dic','devices','ckpt_path','inference_type','num_workers','dataset','dataset_type','if_use_valid','val_dataset','val_dataset_type','save_name','batch_size','num_steps','epoch_steps','epoch_size','schedule']
        else: # only resume the model
            exception_ = ['resume','resume_model','ViTAS_dic','devices','ckpt_path','inference_type','num_workers','batch_size','num_steps','epoch_steps','epoch_size','schedule']
            
    # merge the hparams
    hparams_ = vars(hparams_)
    hparams = vars(hparams)    
    for k in hparams.keys():
        if k in hparams_.keys() and not k in exception_:
            hparams[k] = hparams_[k]
    hparams['ViTAS_dic'].update(hparams_['ViTAS_dic'])
    if not hparams['resume_model']: # resume all, continue training
        last_epoch = int(hparams['ckpt_path'].split('epoch=')[1].split('-')[0])
        hparams['this_epoch'] = last_epoch*hparams['epoch_steps']
        print('resume training from epoch {}, step {}'.format(last_epoch,hparams['this_epoch']))
    else: # resume model and re-start the training
        hparams['this_epoch'] = -1
    return argparse.Namespace(**hparams)
