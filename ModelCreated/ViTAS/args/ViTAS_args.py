import argparse

def config_ViTAS_args(ViTAS_dic):   
    parser = argparse.ArgumentParser()
    parser.add_argument('--ViTAS_channel', type=int, default = 1024, help='channels of the feature from VFM')
    parser.add_argument('--scales', type=int, nargs='+', default = [4,8,16,32], help='select the resolution of the final outputs') # undone
    parser.add_argument('--pre_channels', type=int, nargs='+', default = [64,128,256,256], help='select the channels of the mid-features after preprocess')
    parser.add_argument('--VFM_type', type=str, default = 'DINOv2', choices=['DINOv2'],help='args for load VFM')
    parser.add_argument('--out_channels', type=int, nargs='+', default = [48,64,192,256], help='select the channels of the final outputs')
    parser.add_argument('--attn_splits_list', type=int, nargs='+', default = [8,4,2,2], help='select the split for swin-transformer')
    parser.add_argument('--CA_layers', type=int, default = 2, help='select the number of CA blocks')
    parser.add_argument('--wo_fuse', type=bool, default = False, help='if need the fuse module')
    parser.add_argument('--wo_SDM', type=bool, default = False, help='if need the SDM module')
    parser.add_argument('--ViTAS_fuse', type=str, default = 'PAFM', help='select the fuse method')
    parser.add_argument('--ViTAS_model', type=str, default = 'vit_l', choices=['vit_l','vit_b','vit_s'], help='select the ViTAS_model')
    parser.add_argument('--ViTAS_hooks', type=int, nargs='+', default = [5,11,17,23], help='select the layers of outputs from VFM')
    parser.add_argument('--ViTAS_unfreeze', default = '4', type=str, help='if freeze the last n blocks of VFM')
    parser.add_argument('--ViTAS_pure', default = False, type=bool, help='if load pure VFM')
    parser.add_argument('--fuse_dic', type=dict, default = {}, help='args for fuse module')
    args = parser.parse_args()
    args = args_refine(args,**ViTAS_dic)
    return args

def config_ViTASUni_args(ViTAS_dic):  
    args = config_ViTAS_args(ViTAS_dic)
    args.pre_channels = [128,256,512,512]
    args.out_channels = [128,128,256,512]
    return args

def config_ViTASIGEV_args(ViTAS_dic): 
    args = config_ViTAS_args(ViTAS_dic)  
    args.pre_channels = [64,128,256,256]
    args.out_channels = [48,64,192,160]
    return args

def config_ViTASCre_args(ViTAS_dic):   
    args = config_ViTAS_args(ViTAS_dic)
    args.pre_channels = [256,256,256,256]
    args.out_channels = [256,256,256,256]
    # args.pre_channels = [256,256,256,256]
    # args.out_channels = [256,256,256,256]
    return args

def config_ViTASCroco_args(ViTAS_dic):   
    args = config_ViTAS_args(ViTAS_dic)
    args.ViTAS_pure = True
    assert args.ViTAS_channel == 1024
    return args


ViTAS_config = {'DINOv2':{'vit_l': {'hook_select':[5,11,17,23],'channel':1024,'hook_check':23},
                'vit_b': {'hook_select':[2,5,8,11],'channel':768,'hook_check':11},},
                }

def ViTAS_hooks_init(args): # select the init VFM hooks
    assert args.ViTAS_model in ViTAS_config[args.VFM_type].keys()
    args.ViTAS_hooks = ViTAS_config[args.VFM_type][args.ViTAS_model]['hook_select']
    return args

def ViTAS_channels_init(args): # select the dinoV2 feature channels
    assert args.ViTAS_model in ViTAS_config[args.VFM_type].keys()
    args.ViTAS_channel = ViTAS_config[args.VFM_type][args.ViTAS_model]['channel']
    return args

def check_dino_hooks(args): # check if the hooks is valid
    assert args.ViTAS_model in ViTAS_config[args.VFM_type].keys()
    assert args.ViTAS_hooks[-1] <= ViTAS_config[args.VFM_type][args.ViTAS_model]['hook_check']

def args_refine(args,VFM_type,ViTAS_model,ViTAS_hooks,ViTAS_unfreeze,wo_fuse,wo_SDM,ViTAS_fuse,ViTAS_fuse_patch,ViTAS_fuse_weight,CA_layers):
    if not ViTAS_model is None:
        args.ViTAS_model = ViTAS_model
    args = ViTAS_hooks_init(args)
    args = ViTAS_channels_init(args)
    if not wo_fuse is None:
        args.wo_fuse = wo_fuse
    if not wo_SDM is None:
        args.wo_SDM = wo_SDM
    if not VFM_type is None:
        args.VFM_type = VFM_type
    if not ViTAS_hooks is None:
        args.ViTAS_hooks = ViTAS_hooks
    if not ViTAS_unfreeze is None:
        args.ViTAS_unfreeze = ViTAS_unfreeze
    if not ViTAS_fuse is None:
        args.ViTAS_fuse = ViTAS_fuse
    if not CA_layers is None:
        args.CA_layers = CA_layers
    if not ViTAS_fuse_patch is None:
        args.fuse_dic['UsePatch'] = ViTAS_fuse_patch # use patch in PAFM module
    if not ViTAS_fuse_patch is None:
        args.fuse_dic['UseWeight'] = ViTAS_fuse_weight # use DANE weight in PAFM module
    assert args.ViTAS_fuse in ['SDFA','PAFM','VFM']
    assert args.ViTAS_unfreeze in ['0','1','2','3','4','5'], str(args.ViTAS_unfreeze)
    check_dino_hooks(args)
    return args