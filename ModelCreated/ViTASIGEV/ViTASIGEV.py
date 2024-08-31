import torch.nn as nn
from ModelCreated.ViTAS.args.ViTAS_args import config_ViTASIGEV_args
from ModelCreated.ViTAS.ViTAS_model import ViTASBaseModel
from ModelCreated.ViTASIGEV.igev.igev_model import IGEVStereo,config_IGEV_args

def get_parameter_number(model,name=None):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(name, ' Total:', total_num, 'Trainable:', trainable_num)
    # return {'Total': total_num, 'Trainable': trainable_num}
    
class ViTASIGEVModel(nn.Module):
    def __init__(self,ViTAS_dic):
        super().__init__()
        self.ViTAS = self.load_ViTAS(ViTAS_dic)
        get_parameter_number(self.ViTAS,'ViTAS')
        self.igev = self.load_IGEV()

    def load_ViTAS(self,ViTAS_dic):
        args = config_ViTASIGEV_args(ViTAS_dic)
        model = ViTASBaseModel(**vars(args))
        return model
    
    def load_IGEV(self):
        igev_args = config_IGEV_args()
        model = IGEVStereo(igev_args)                
        return model
        
    def forward(self, img1, img2, iters, test_mode=False):  
        features_out = self.ViTAS(img1,img2) # 
        feature1_list = features_out['feature0_out_list']
        feature2_list = features_out['feature1_out_list']
        
        feature1_list.reverse() 
        feature2_list.reverse()    
        pred_disp = self.igev(feature1_list, feature2_list, img1, img2, iters=iters, test_mode=test_mode)   
        return pred_disp
   
