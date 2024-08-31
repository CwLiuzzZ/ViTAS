import configargparse

def get_opts():
    parser = configargparse.ArgumentParser()

    # dataset options
    parser.add_argument('--dataset', default = 'KITTI2015', type=str)
    parser.add_argument('--dataset_type', type=str, default='train', choices=['train','test'],help = 'use train or test dataset')
    parser.add_argument('--if_use_valid', type=bool, default=False, help = "if use validation dataset, effects on the generate_file_path")
    parser.add_argument('--val_dataset', default = 'MiddEval3H', type=str)
    parser.add_argument('--val_dataset_type', type=str, default='train', choices=['train','test'])
    parser.add_argument('--keep_size', default=False, type=bool, help='do not resize the input image')
    parser.add_argument('--max_disp', type=int, default=192, help="max_disparity of the datasets")
    parser.add_argument('--max_disp_sr', type=int, default=192, help="max_disparity when estimate the disparity")
    parser.add_argument('--save_name', type=str, default='Delete', help="name for saving the results")
    parser.add_argument('--resize', type=float, default=None, help="resize the image")

    # model options
    parser.add_argument('--network', type=str, default='ViTASIGEV') # , choices=['AANet','PSMNet','RAFT_Stereo','DFM','LacGwc','Unimatch','delete','BGNet','VGG','SGBM','SRP','FBS']
    parser.add_argument('--ckpt_path', type=str, default='models/ViTASIGEV/KITTI.pth', help='pretrained checkpoint path to load from YoYo')
    parser.add_argument('--pre_trained', type=bool, default=False, help='pretrained checkpoint path to load from github')

    # training options
    parser.add_argument('--inference_type', type=str, default='evaluate', help='main inference, and has effect on the aug_config', choices=['evaluate','train'])
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate') 
    parser.add_argument('--min_lr', type=float, default=2e-5, help='minimum learning rate') 
    # parser.add_argument('--freeze_bn', default=True, type=bool)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_steps', type=int, default=2000, help='number of training steps')
    parser.add_argument('--epoch_steps', type=int, default=800, help='number of training steps of each epoch')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers')
    parser.add_argument('--epoch_size', type=int, default=400, help='number of training epochs')
    parser.add_argument('--devices', type=int, nargs='+', default = [0], help='GPU devices used')
    parser.add_argument('--schedule', type=str, default='Cycle', help='select lr schedule')
    parser.add_argument('--this_epoch', type=int, default='-1', help='if resume training, use this epoch')
    parser.add_argument('--resume', type=bool, default=False, help='if resume the training')
    parser.add_argument('--resume_model', type=bool, default=False, help='if resume the training')
    parser.add_argument('--hparams_dir', type=str, default=None, help="name for saving the results")
    
    # ViTAS options
    parser.add_argument('--ViTAS_dic', type=dict, default = {'VFM_type':None,'ViTAS_model':None,'ViTAS_unfreeze':'5','ViTAS_hooks':None,'wo_SDM':None,'wo_fuse':None,'ViTAS_fuse':'PAFM','ViTAS_fuse_patch':None,'ViTAS_fuse_weight':None,'CA_layers':None}, help='args for ViTAS')
    
    return parser.parse_args()