import numpy as np
# from PIL import Image
import math
from torch.utils.data import Dataset
import cv2

# import sys
# sys.path.append('../..')
from toolkit.function.base_function import io_disp_read,dic_merge
from toolkit.data_loader.transforms import Augmentor 


aug_config_dic_train = {
                        'ViTASIGEV':{'RandomColor':True,'VFlip':False,'crop':(320,700),'rotate':False,'scale':True,'erase':True,'color_diff':False}, # (320,700), (470,940) for batchsize=1 
                        }
aug_config_dic_evaluate = {
                        'ViTASIGEV':{'crop':None},
                        }


def dataloader_customization(hparams):
    network = hparams.network

    #########################
    ### adjust aug_config ### 
    #########################
    
    if 'train' in hparams.inference_type:
        aug_config = aug_config_dic_train[network].copy()
    elif 'evaluate' in hparams.inference_type:
        aug_config = aug_config_dic_evaluate[network].copy()
    valid_aug_config = aug_config_dic_evaluate[network].copy()
    if hparams.keep_size:
        aug_config['resize']=None

    if not hparams.resize is None:
        aug_config['resize']=hparams.resize
    
    # 'scale can be deployed only when crop_size is set'
    if aug_config['crop'] is None:
        if 'scale' in aug_config.keys():
            assert not aug_config['scale'], 'scale can be deployed only when crop_size is set'

base_lr_dic = {'ViTASIGEV':{'lr':1e-4,'min_lr':1e-5}, # 7~0.8
               }

max_disp_dic = {'ViTASIGEV':700,
               }

def optimizer_customization(hparams,n_img):
    hparams.epoch_steps = math.ceil(n_img/(hparams.batch_size*len(hparams.devices)))
    hparams.num_steps = hparams.epoch_steps*hparams.epoch_size
    print('total steps:', hparams.num_steps, ' epoch steps:', hparams.epoch_steps,' total epoch: ',hparams.epoch_size)
    if hparams.num_steps > 500000: # 50000
        hparams.schedule = 'Cycle' # for large dataset
    else:
        hparams.schedule = 'OneCycle' # for small dataset
    if hparams.network in base_lr_dic.keys():
        hparams.lr = base_lr_dic[hparams.network]['lr']
        hparams.min_lr = base_lr_dic[hparams.network]['min_lr'] 
    if hparams.network in max_disp_dic.keys():
        hparams.max_disp = max_disp_dic[hparams.network]
    return hparams

def prepare_dataset(file_paths_dic, aug_config):

    '''
    function: make dataloader
    input:
        file_paths_dic: store file paths
        aug_config: configuration for augment
    output:
        dataloader
    '''
    # augmentation
    transformer = Augmentor(**aug_config)
    dataset = StereoDataset(file_paths_dic,transform=transformer)
    
    n_img = len(dataset)
    print('Use a dataset with {} image pairs'.format(n_img))
    return dataset,n_img

def default_loader(path):
    '''
        function: read left and right images
        output: array
    '''
    # return Image.open(path).convert('RGB')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class StereoDataset(Dataset):
    def __init__(self, file_paths_dic,transform, loader=default_loader, dploader=io_disp_read):
        super(StereoDataset, self).__init__()
        self.transform = transform
        self.loader = loader
        self.disploader = dploader
        self.samples = []
        self.load_disp = True

        self.lefts = file_paths_dic['left_list']
        self.rights = file_paths_dic['right_list']
        self.disps = file_paths_dic['disp_list']
        self.save_dirs1 = file_paths_dic['save_path_disp']
        self.save_dirs2 = file_paths_dic['save_path_disp_image']
        self.addition = file_paths_dic['addition_list'] # shoule be a mask
        
        # print('number of files: left image {}, right image {}, disp {}'.format(len(self.lefts), len(self.rights), len(self.disps)))
        assert len(self.lefts) == len(self.rights), "{},{}".format(len(self.lefts),len(self.rights))
        if not len(self.disps) == len(self.lefts):
            print('warning: disp file numbers not equal image pair numbers, use zero disparity map')
            self.load_disp = False
        # assert len(self.lefts) == len(self.rights) == len(self.disps), "{},{},{}".format(len(self.lefts),len(self.rights),len(self.disps))
        for i in range(len(self.lefts)):
            sample = dict()
            sample['left'] = self.lefts[i]
            sample['right'] = self.rights[i]
            sample['mask'] = self.addition[i]
            if self.load_disp:
                sample['disp'] = self.disps[i]
            sample['save_dir1'] = self.save_dirs1[i]
            sample['save_dir2'] = self.save_dirs2[i]
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        
        sample = {}
        sample_path = self.samples[index]
    
        sample['left'] = self.loader(sample_path['left']) # array
        sample['right'] = self.loader(sample_path['right']) # array
        if not sample_path['mask'] is None:
            sample['mask'] = cv2.imread(sample_path['mask'],-1) # array
        else:
            sample['mask'] = np.zeros((sample['left'].shape[0],sample['left'].shape[1]))
        if self.load_disp:
            sample['disp'] = self.disploader(sample_path['disp'])
            sample['disp_dir'] = sample_path['disp']
        else:
            sample['disp'] = np.zeros(shape=(np.array(sample['left']).shape[0],np.array(sample['left']).shape[1]))
        sample['left_dir'] = sample_path['left']
        sample['right_dir'] = sample_path['right']
        sample['save_dir_disp'] = sample_path['save_dir1']
        sample['save_dir_disp_vis'] = sample_path['save_dir2']
        # sample['ori_shape'] = sample['disp'].shape
        
        sample = self.transform(sample)
            
        return sample
