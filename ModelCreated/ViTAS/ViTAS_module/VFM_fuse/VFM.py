import torch.nn as nn
import torch.nn.functional as F
from ModelCreated.ViTAS.ViTAS_module.Uni_CA.CA_utils import feature_add_position_cross_feature
from ModelCreated.ViTAS.ViTAS_module.Uni_CA.CA_transformer import FeatureFuseTransformer
from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class VFM(nn.Module):
    def __init__(self, 
                 channels = [48,64,192,160],
                 num_heads = 4,
                 UsePatch = False,
                 ):
        super().__init__()    
        self.channels = channels                    
                  
        self.CA = nn.ModuleList([FeatureFuseTransformer(1,self.channels[i],1,2,self.channels[i+1]) for i in range(len(self.channels)-1)]) # i = 0,1,2
                    
        self.conv1 = nn.ModuleList([nn.Conv2d(self.channels[i], self.channels[i], kernel_size=(1,1), stride=1) for i in range(len(self.channels)-1)])
        self.conv3 = nn.ModuleList([nn.Conv2d(self.channels[i], self.channels[i], kernel_size=(3,3), stride=1,padding=1,padding_mode='replicate') for i in range(len(self.channels)-1)])
                    
    def fuse(self,i,x_l, x_r, x_l_high, x_r_high): # i=3,2,1
        
        x_l  = F.interpolate(x_l, scale_factor=2., mode='bilinear', align_corners=True) 
        x_r  = F.interpolate(x_r, scale_factor=2., mode='bilinear', align_corners=True)
        
        x_l_high = self.conv1[i-1](x_l_high)
        x_r_high = self.conv1[i-1](x_r_high)
        
        x_l, x_r, x_l_high, x_r_high, = feature_add_position_cross_feature(x_l, x_r, x_l_high, x_r_high,self.channels[i],self.channels[i-1])
        
        x_l_out = self.CA[i-1](x_l_high,x_l,attn_type='self_swin2d_cross_swin1d',attn_num_splits=4)
        x_r_out = self.CA[i-1](x_r_high,x_r,attn_type='self_swin2d_cross_swin1d',attn_num_splits=4)
                
        x_l_out = self.conv3[i-1](x_l_out)
        x_r_out = self.conv3[i-1](x_r_out)  
        return x_l_out,x_r_out
