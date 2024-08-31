import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import cv2
import numpy as np
import time

from toolkit.function import base_function
from toolkit.torch_lightning.lightning_function import schedule_select
from toolkit.function.evaluator import calcu_EPE,calcu_PEP,calcu_D1all
from toolkit.function.models import prepare_model

class ViTASIGEV(LightningModule):
    def __init__(self, hparams):
        super(ViTASIGEV, self).__init__()   
        self.save_hyperparameters()
        # model
        self.model = prepare_model(hparams)        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.validation_metric = {'step_EPE':[],'step_PEP1':[],'step_PEP2':[],'step_PEP3':[],'step_D1all':[]}
        self.test_metric = {'step_EPE':[],'step_PEP1':[],'step_PEP2':[],'step_PEP3':[],'step_D1all':[],'step_time':[]}
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.hparams['hparams'].lr,weight_decay=1e-5)
        lr_scheduler = schedule_select(optimizer,self.hparams['hparams'])
        return [optimizer],[lr_scheduler]

    def training_step(self, batch, batch_idx):
        imgL   = batch['left']
        imgR   = batch['right']
        disp_true = batch['disp']
        
        padder = base_function.InputPadder(imgL.shape,64, mode = 'replicate')
        [imgL, imgR],_,_ = padder.pad(imgL, imgR)
        
        (disp_init_pred, disp_preds) = self.model(imgL, imgR, iters=22)
        
        disp_preds = [padder.unpad(disp_preds_.squeeze(1)) for disp_preds_ in disp_preds]
        disp_init_pred = padder.unpad(disp_init_pred.squeeze(1))

        loss = sequence_loss(disp_preds, disp_init_pred, disp_true,max_disp=self.hparams['hparams'].max_disp)
        self.training_step_outputs.append(loss.item())
        self.log('loss_step', loss) 
        return loss

    def on_train_epoch_end(self):
        self.training_step_outputs = np.array(self.training_step_outputs)[~np.isnan(self.training_step_outputs)]
        train_epoch_loss = np.mean(self.training_step_outputs)
        self.log('loss_epoch', train_epoch_loss) 
        self.training_step_outputs = []# free memory

    def validation_step(self, batch, batch_idx):
        
        imgL   = batch['left']
        imgR   = batch['right']
        disp_true = batch['disp']

        padder = base_function.InputPadder(imgL.shape,64, mode = 'replicate')
        [imgL, imgR],_,_ = padder.pad(imgL, imgR)
        
        pred = self.model(imgL, imgR, iters=32, test_mode=True)
        pred = padder.unpad(pred.squeeze(1))

        mask = (disp_true > 0)&(disp_true<self.hparams['hparams'].max_disp)
        mask.detach_()
        self.validation_metric['step_EPE'].append(calcu_EPE(pred[mask], disp_true[mask]).item())
        self.validation_metric['step_D1all'].append(calcu_D1all(pred[mask], disp_true[mask]).item())
        
    def on_validation_epoch_end(self):
        self.validation_metric['step_EPE'] = np.array(self.validation_metric['step_EPE'])[~np.isnan(self.validation_metric['step_EPE'])]
        valid_epoch_EPE = np.mean(self.validation_metric['step_EPE'])
        self.log('valid_epoch_EPE', valid_epoch_EPE, sync_dist=True) 
        self.validation_metric['step_EPE'] = []# free memory
        
        self.validation_metric['step_D1all'] = np.array(self.validation_metric['step_D1all'])[~np.isnan(self.validation_metric['step_D1all'])]
        valid_epoch_D1 = np.mean(self.validation_metric['step_D1all'])
        self.log('valid_epoch_D1', valid_epoch_D1, sync_dist=True) 
        self.validation_metric['step_D1all'] = []# free memory

    def test_step(self, batch, batch_idx): 
        
        imgL   = batch['left']
        imgR   = batch['right']
        disp_true = batch['disp']
        
        time0 = time.time()

        padder = base_function.InputPadder(imgL.shape,64, mode = 'replicate')
        [imgL, imgR],_,_ = padder.pad(imgL, imgR)
         
        pred = self.model(imgL, imgR, iters=32, test_mode=True)
        pred = padder.unpad(pred.squeeze(1))
        self.test_metric['step_time'].append(time.time()-time0)
        
        if not self.hparams['hparams'].resize is None:
        # if self.hparams['hparams'].dataset == 'MiddEval3F':
            img = cv2.imread(batch['left_dir'][0])
            pred = F.interpolate(pred.unsqueeze(0)*img.shape[1]/pred.shape[-1],size=(img.shape[0],img.shape[1]),mode='nearest').squeeze(0)
        
        if torch.mean(disp_true)==0:
            disp_true = pred
        mask = (disp_true > 0)&(disp_true<self.hparams['hparams'].max_disp) 
        
        self.test_metric['step_EPE'].append(calcu_EPE(pred[mask], disp_true[mask]).item())
        self.test_metric['step_PEP1'].append(calcu_PEP(pred[mask], disp_true[mask],thr=1).item())
        self.test_metric['step_PEP2'].append(calcu_PEP(pred[mask], disp_true[mask],thr=2).item())
        self.test_metric['step_PEP3'].append(calcu_PEP(pred[mask], disp_true[mask],thr=3).item())
        self.test_metric['step_D1all'].append(calcu_D1all(pred[mask], disp_true[mask]).item())
        
        pred_disp = pred.squeeze().detach().cpu().numpy()
                
        disp_true = disp_true.squeeze().detach().cpu().numpy()
        
        base_function.save_disp_results(batch['save_dir_disp'][0],batch['save_dir_disp_vis'][0],pred_disp,np.max(disp_true),np.min(disp_true),display=False)
        

    def on_test_epoch_end(self):
        results = {}
        results['EPE'] = np.mean(self.test_metric['step_EPE'])
        results['PEP1']  = np.mean(self.test_metric['step_PEP1'])
        results['PEP2']  = np.mean(self.test_metric['step_PEP2'])
        results['PEP3']  = np.mean(self.test_metric['step_PEP3'])
        results['D1all']  = np.mean(self.test_metric['step_D1all'])
        results['time_mean']  = np.mean(self.test_metric['step_time'])
        results['time_min']  = np.min(self.test_metric['step_time'])
        print(results)
        

def split_prediction_conf(predictions, with_conf=False):
    if not with_conf:
        return predictions, None
    conf = predictions[:,-1:,:,:]
    predictions = predictions[:,:-1,:,:]
    return predictions, conf

def sequence_loss(disp_preds, disp_init_pred, disp_gt, loss_gamma=0.9, max_disp=192,fg_mask=None):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    
    valid = disp_gt > 0
    valid[disp_gt > max_disp]=False
    valid.detach_()
    # fg_mask.detach_()
    
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid]).any()

    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[valid], disp_gt[valid], reduction='mean')
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()        
        # i_loss[fg_mask>0]*=1.2
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid].mean()

    return disp_loss
