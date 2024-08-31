import torch
import torch.nn as nn

from toolkit.args.model_args import get_dinov2_args_parser_1,dinoV2_config_dir_dic,dinoV2_ckpt_dir_dic
from model_pack.dinoV2.dinov2.eval.setup import setup_and_build_model as dinoV2_model
from ModelCreated.ViTAS.ViTAS_module.Uni_CA.CA_transformer import FeatureTransformer
from ModelCreated.ViTAS.ViTAS_module.Uni_CA.CA_utils import feature_add_position

from ModelCreated.ViTAS.ViTAS_module.SDFA_fuse.SDFA import SDFA
from ModelCreated.ViTAS.ViTAS_module.PAFM_fuse.PAFM import PAFM
from ModelCreated.ViTAS.ViTAS_module.VFM_fuse.VFM import VFM

def get_parameter_number(model,name=None):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(name, ' Total:', total_num, 'Trainable:', trainable_num)
    # return {'Total': total_num, 'Trainable': trainable_num}
    
class ViTASBaseModel(nn.Module):
    def __init__(self,VFM_type='DINOv2',scales=[4,8,16,32],ViTAS_hooks=[5,11,17,23],ViTAS_channel=1024,pre_channels=[64,128,256,512], out_channels=[48,64,192,160],attn_splits_list=[8,4,2,2],CA_layers=2,wo_fuse=False,wo_CA=False,wo_SDM=False,ViTAS_model='vit_l',ViTAS_unfreeze='0',ViTAS_pure=False,ViTAS_fuse='PAFM',fuse_dic = {}):
    # def __init__(self,scales=[4,8,16,32],ViTAS_hooks=[5,11,17,23],ViTAS_channel=1024,pre_channels=[64,128,256,512], out_channels=[48,64,192,160],attn_splits_list=[8,4,2,2],CA_layers=2,wo_fuse=False,wo_CA=False,ViTAS_model='vit_l',ViTAS_unfreeze='0',ViTAS_pure=False,fuse_dic = {}):
            
        # CA 2760128 with 2 layers
        # SDM 3033040 with 4 blocks
        # PAFM 224908 with LA and GA
        # dinov2 204368640: tranable: 62993408 of 3 unfrozen
        
        super().__init__()
        # self.scales = scales # unuse
        print('VFM_type:',VFM_type)
        print('ViTAS_model:',ViTAS_model)
        print('ViTAS_hooks:',ViTAS_hooks)
        print('ViTAS_unfreeze:',ViTAS_unfreeze)
        print('ViTAS_wo_fuse:',wo_fuse)
        print('ViTAS_wo_SDM:',wo_SDM)
        print('CA_layers:',CA_layers)
        print('ViTAS_fuse:',ViTAS_fuse, ' fuse_dic:',fuse_dic)
        assert len(scales)==len(ViTAS_hooks)==len(pre_channels)==len(out_channels)==len(attn_splits_list),str(len(scales))+' '+str(len(ViTAS_hooks))+' '+str(len(pre_channels))+' '+str(len(out_channels))+' '+str(len(attn_splits_list))
        # Dino config
        self.VFM_type = VFM_type
        self.ViTAS_model = ViTAS_model  
        self.ViTAS_hooks = ViTAS_hooks  
        self.ViTAS_fuse = ViTAS_fuse   
        self.CA_layers = CA_layers   
        # self.ViTAS_unfreeze = ViTAS_unfreeze
        self.pre_midd_channel = None
        # self.VFM = self.load_vfm(ViTAS_model=ViTAS_model)
        self.VFM  = load_vfm(VFM_type,ViTAS_model,ViTAS_unfreeze)
        
        # self.DinoV2 = self.load_DinoV2(ViTAS_model=ViTAS_model)
        
        if not ViTAS_pure:
            # Adapter config
            self.attn_splits_list = attn_splits_list # for CA
            self.pre_channels,self.out_channels,self.pre_midd_channel= channels_init(self.ViTAS_fuse,ViTAS_channel,pre_channels,out_channels)
            self.wo_fuse = wo_fuse
            self.wo_SDM = wo_SDM
            # self.wo_CA = wo_CA
            # assert (not wo_fuse) or (not wo_CA)
            if self.wo_fuse: # without fuse
                self.pre_channels = out_channels
            else: # with fuse
                if self.ViTAS_fuse == 'SDFA':
                    self.fuse_blocks = SDFA(in_channels = self.pre_channels, out_channels=self.out_channels)
                if self.ViTAS_fuse == 'PAFM':
                    self.fuse_blocks = PAFM(self.pre_channels,**fuse_dic)
                if self.ViTAS_fuse == 'VFM':
                    self.fuse_blocks = VFM(self.pre_channels,**fuse_dic)
                get_parameter_number(self.fuse_blocks,'fuse')
            # self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.preprocess = make_preprocess(scales,ViTAS_channel=ViTAS_channel,out_channels=self.pre_channels,midd_channels = self.pre_midd_channel,wo_SDM=wo_SDM)
            self.CA = make_CA(scales,CA_layers=self.CA_layers, channels=self.out_channels)
            get_parameter_number(self.VFM,VFM)
            if not self.wo_SDM:
                get_parameter_number(self.preprocess,'SDM')
            get_parameter_number(self.CA,'CA')
        
    
    def forward(self,img1, img2, get_ori_dino=False,get_before_CA=False): 
        features_out = {'feature0_out_list':[],'feature1_out_list':[],'feature0_before_CA_list':[],'feature1_before_CA_list':[]}
        if self.VFM_type in ['DINOv2','DepthAny']:
            img1 = nn.functional.interpolate(img1, scale_factor=56/64, mode='bilinear', align_corners=True)
            img2 = nn.functional.interpolate(img2, scale_factor=56/64, mode='bilinear', align_corners=True)
        
        image_concat = torch.cat((img1, img2), dim=0)  # [2B, C, H, W]
        # Dino_features = self.vfm.get_intermediate_layers(image_concat,n=self.ViTAS_hooks,reshape=True) # from high_res to low_res
        VFM_features = self.get_intermediate_layers(image_concat,n=self.ViTAS_hooks,reshape=True) # from high_res to low_res
        
        len_ = len(VFM_features)
        feature0_list = []
        feature1_list = []        
        features_list = []
    
        for i in range(len_):
            features_list.append(self.preprocess[i](VFM_features[i])) # from high_res to low_res
        if self.wo_SDM:
            features_list = interpolate_features(features_list)  # from high_res to low_res
        for i in range(len_):
            features = features_list[i]
            chunk = torch.chunk(features, chunks=2, dim=0)
            feature0_list.append(chunk[0])
            feature1_list.append(chunk[1]) 
        
        x_l = feature0_list[-1] # left , lowest_res
        x_r = feature1_list[-1] # right, lowest_res        
        for i in range(len_-1, -1, -1): # i = 3,2,1,0, fuse from low_res to high_res    
            if get_before_CA:  
                features_out['feature0_before_CA_list'].append(x_l)
                features_out['feature1_before_CA_list'].append(x_r)
            # add PE and cross-attention Transformer
            if not self.CA_layers == 0:
                x_l, x_r = feature_add_position(x_l, x_r, self.attn_splits_list[i], self.out_channels[i])
                x_l, x_r = self.CA[i](x_l, x_r,attn_type='self_swin2d_cross_swin1d',attn_num_splits=self.attn_splits_list[i])
                        
            features_out['feature0_out_list'].append(x_l)
            features_out['feature1_out_list'].append(x_r)
            if i > 0:
                if self.wo_fuse: # without fuse
                    x_l = feature0_list[i-1]
                    x_r = feature1_list[i-1]
                else: # with fuse
                    x_l,x_r = self.fuse_blocks.fuse(i,x_l,x_r,feature0_list[i-1],feature1_list[i-1])
                
        return features_out # return lists of features from low resolution to high resolution

    def get_intermediate_layers(self,img,n,reshape):
        return self.VFM.get_intermediate_layers(img,n,reshape)
    
    def forward_patch_features(self,img):
        return self.VFM.forward_patch_features(img)

def load_vfm(VFM_type,ViTAS_model,ViTAS_unfreeze):
    if VFM_type == 'DINOv2':
        knn_args_parser = get_dinov2_args_parser_1(add_help=False)
        args = knn_args_parser.parse_args()
        args.config_file = dinoV2_config_dir_dic[ViTAS_model]
        args.pretrained_weights = dinoV2_ckpt_dir_dic[ViTAS_model]
        model, autocast_dtype = dinoV2_model(args)
        model = DINOv2_weights_unfreeze(model,ViTAS_unfreeze)
    else:
        raise ValueError("Wrong VFM type is given.")
    return model 



def channels_init(fuse_mode,ViTAS_channel,pre_channels,out_channels):
    if fuse_mode == 'SDFA':
        # pre_channels[-1] = out_channels[-1]
        # pre_midd_channel = pre_channels
        pre_channels = out_channels
        pre_midd_channel = [int(ViTAS_channel/2)]*4
    elif 'PAFM' in fuse_mode:
        pre_channels = out_channels
        pre_midd_channel = [int(ViTAS_channel/2)]*4
    elif 'VFM' in fuse_mode:
        pre_channels = out_channels
        pre_midd_channel = [int(ViTAS_channel/2)]*4
    else:
        raise ValueError('Unknown fuse mode :{}'.format(fuse_mode))
    return pre_channels,out_channels,pre_midd_channel

def make_preprocess(scale,ViTAS_channel=1024,out_channels=[48,64,192,160],midd_channels=[48,64,192,160],wo_SDM=False):    
    assert len(scale) in [3,4]
    
    if not wo_SDM:
        act_1_preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=ViTAS_channel,
                out_channels=midd_channels[0],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=midd_channels[0],
                out_channels=out_channels[0],
                kernel_size=4, stride=4, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        act_2_preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=ViTAS_channel,
                out_channels=midd_channels[1],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=midd_channels[1],
                out_channels=out_channels[1],
                kernel_size=2, stride=2, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        act_3_preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=ViTAS_channel,
                out_channels=out_channels[2],
                kernel_size=1, stride=1, padding=0,
            )
        )

        if len(scale) == 3:
            return  nn.ModuleList([
                    act_1_preprocess, # 1/4
                    act_2_preprocess, # 1/8
                    act_3_preprocess, # 1/16
                ])

        act_4_preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=ViTAS_channel,
                out_channels=midd_channels[3],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=midd_channels[3],
                out_channels=out_channels[3],
                kernel_size=3, stride=2, padding=1,
            )
        )
    else:
        act_1_preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=ViTAS_channel,
                out_channels=out_channels[0],
                kernel_size=1, stride=1, padding=0,
            ),
        )

        act_2_preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=ViTAS_channel,
                out_channels=out_channels[1],
                kernel_size=1, stride=1, padding=0,
            ),
        )

        act_3_preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=ViTAS_channel,
                out_channels=out_channels[2],
                kernel_size=1, stride=1, padding=0,
            )
        )

        if len(scale) == 3:
            return  nn.ModuleList([
                    act_1_preprocess, # 1/4
                    act_2_preprocess, # 1/8
                    act_3_preprocess, # 1/16
                ])

        act_4_preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=ViTAS_channel,
                out_channels=out_channels[3],
                kernel_size=1, stride=1, padding=0,
            )
        )

    return  nn.ModuleList([
            act_1_preprocess, # 1/4
            act_2_preprocess, # 1/8
            act_3_preprocess, # 1/16
            act_4_preprocess  # 1/32
        ])
    
def make_CA(scale,CA_layers=2,channels=[48,64,192,160],ffn_dim_expansion=[2,2,2,2]):       
    assert len(scale) in [3,4] 
    CA_1 = FeatureTransformer(num_layers=CA_layers,d_model=channels[0],nhead=1,ffn_dim_expansion=ffn_dim_expansion[0])
    CA_2 = FeatureTransformer(num_layers=CA_layers,d_model=channels[1],nhead=1,ffn_dim_expansion=ffn_dim_expansion[1])
    CA_3 = FeatureTransformer(num_layers=CA_layers,d_model=channels[2],nhead=1,ffn_dim_expansion=ffn_dim_expansion[2])
    if len(scale) == 3:
        return  nn.ModuleList([CA_1,CA_2,CA_3])
    CA_4 = FeatureTransformer(num_layers=CA_layers,d_model=channels[3],nhead=1,ffn_dim_expansion=ffn_dim_expansion[3])
    return  nn.ModuleList([CA_1,CA_2,CA_3,CA_4])

# def make_fuse()


def interpolate_features(feature_list):
    output = []
    output.append(torch.nn.functional.interpolate(feature_list[0], scale_factor=4, mode='bilinear', align_corners=True))
    output.append(torch.nn.functional.interpolate(feature_list[1], scale_factor=2., mode='bilinear', align_corners=True))
    output.append(feature_list[2])
    output.append(torch.nn.functional.interpolate(feature_list[3], scale_factor=0.5, mode='bilinear', align_corners=True))
    return  output

def DINOv2_weights_unfreeze(model,ViTAS_unfreeze):
    if ViTAS_unfreeze=='0':
        for p in model.parameters():
            p.requires_grad = False
    elif ViTAS_unfreeze=='1':
        for k,v in model.named_parameters():
            if not ('norm.weight' in k or 'blocks.23' in k):
                v.requires_grad=False 
    elif ViTAS_unfreeze=='2':
        for k,v in model.named_parameters():
            if not ('norm.weight' in k or 'blocks.22' in k or 'blocks.23' in k):
                v.requires_grad=False 
    elif ViTAS_unfreeze=='3':
        for k,v in model.named_parameters():
            if not ('norm.weight' in k or 'blocks.21' in k or 'blocks.22' in k or 'blocks.23' in k):
                v.requires_grad=False 
    elif ViTAS_unfreeze=='4':
        for k,v in model.named_parameters():
            if not ('norm.weight' in k or 'norm.bias' in k or 'blocks.20' in k or 'blocks.21' in k or 'blocks.22' in k or 'blocks.23' in k):
                v.requires_grad=False 
    elif ViTAS_unfreeze=='5':
        for k,v in model.named_parameters():
            if not ('norm.weight' in k or 'norm.bias' in k or 'blocks.19' in k or 'blocks.20' in k or 'blocks.21' in k or 'blocks.22' in k or 'blocks.23' in k):
                v.requires_grad=False 
    return model