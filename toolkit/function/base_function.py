import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import re
# from pylab import math
import math
import sys
import skimage.io

# merge two dics
def dic_merge(dic1,dic2):
    if not len(list(dic1.keys()))==0:
        assert dic1.keys() == dic2.keys(),dic1.keys()+'---'+dic2.keys()
        ans = {}
        for key in dic2.keys():
            ans[key] = dic1[key]+dic2[key]
    else:
        return dic2
    return ans

def single_image_warp(img,disp,mode='right', tensor_output = False):
    '''
    function:
        warp single image with disparity map to another perspective
    input:
        img: image; should be 2D or 3D array
        disp: disparity map; should be 2D array or tensor
        mode: perspective of the input image
    output:
    '''
    assert mode in ['left','right']

    if not isinstance(img, torch.Tensor):
        if len(img.shape)==3:
            img = np.transpose(img,(2,0,1))
        img = img.astype(np.float32)
        img = torch.from_numpy(img) # [C,H,W] or [H,W]

    if not isinstance(disp, torch.Tensor):
        disp = torch.tensor(disp)
    disp = (disp/disp.shape[1]).float()

    if mode == 'left':
        # should be negative disparity
        if torch.mean(disp)<0:
            disp=-disp
    elif mode == 'right':
        # should be positive disparity
        if torch.mean(disp)>0:
            disp=-disp
    
    # disp = torch.from_numpy(disp/disp.shape[1]).float()
    
    assert img.shape[-1] == disp.shape[-1], str(img.shape)+' and '+str(disp.shape)
    assert img.shape[-2] == disp.shape[-2], str(img.shape)+' and '+str(disp.shape)

    disp = disp.unsqueeze(0).unsqueeze(0)
    if len(img.shape)==2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif len(img.shape)==3:
        img = img.unsqueeze(0)
        
    batch_size, _, height, width = img.shape
    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                width, 1).transpose(1, 2).type_as(img)
    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                            padding_mode='zeros',align_corners=True)

    if tensor_output: 
        return output # [1,C,H,W]

    if output.shape[1]==1:
        output = output[0][0].detach().numpy()
    else:
        output = output[0].detach().numpy()
    if len(output.shape)==3:
        output = np.transpose(output,(1,2,0))

    return output

# COLORMAP_JET, COLORMAP_PARULA, COLORMAP_MAGMA, COLORMAP_PLASMA, COLORMAP_VIRIDIS
# disp should be [H,W] numpy.array
def disp_vis(save_dir,disp,max_disp=None,min_disp=None,colormap=cv2.COLORMAP_JET,inverse=False):
    assert len(disp.shape) == 2, len(disp.shape)
    if torch.is_tensor(disp):
        disp = disp.detach().cpu().numpy()
    if max_disp is None:
        max_disp = np.max(disp)
    if min_disp is None:
        min_disp = np.min(disp)
    max_disp = (int(max_disp/5)+1)*5
    min_disp = int(min_disp/5)*5
    disp = np.clip(disp,min_disp,max_disp)
    disp = 255 * (disp-min_disp)/(max_disp-min_disp)
    # disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)        
    disp=disp.astype(np.uint8)
    if inverse:
        disp = 255 - disp
    disp = cv2.applyColorMap(disp,colormap)
    
    # save_dir_split = save_dir.split('.png')
    # save_dir = save_dir_split[0]+'_{}_{}'.format(str(max_disp),str(min_disp))+'.png'
    if not save_dir is None:
        cv2.imwrite(save_dir,disp)
    else:
        return disp



# COLORMAP_JET, COLORMAP_PARULA, COLORMAP_MAGMA, COLORMAP_PLASMA, COLORMAP_VIRIDIS
# disp should be [H,W] numpy.array
def disp_D1_vis(save_dir,disp,max_disp=None,min_disp=0,colormap=cv2.COLORMAP_JET,inverse=False,max_color=255,min_color=0,backupcolor=0,error_color=255):
    
    assert len(disp.shape) == 2, len(disp.shape)
    if torch.is_tensor(disp):
        disp = disp.detach().cpu().numpy()
    if max_disp is None:
        max_disp = np.mean(disp)

    mask1 = disp>max_disp
    mask2 = disp==0

    disp[~mask1] = min_color+(max_color-min_color) * (disp[~mask1]-min_disp)/(max_disp-min_disp)
    disp[mask1] = error_color
    disp[mask2]=backupcolor 
    disp=disp.astype(np.uint8)
    # if inverse:
    #     disp = 255 - disp
    disp = cv2.applyColorMap(disp,colormap)
    # print(disp.shape)
    disp[mask1,:] = [40, 100, 255]
    disp[mask2,:]=[255,255,255] 
    cv2.imwrite(save_dir,disp)

def disp_vis_PLA(save_dir,disp,max_disp=None,min_disp=None):
    '''
    input:
        save_dir: dir to save image
        disp: input disparity map; should be a 2D array or tensor
        max_disp: max disparity; used for normalizing disparity map
    '''
    assert len(disp.shape) == 2, len(disp.shape)
    if torch.is_tensor(disp):
        disp = disp.detach().cpu().numpy()
    if max_disp is None:
        max_disp = np.max(disp)
    if min_disp is None:
        min_disp = np.min(disp)
        
    print(max_disp,min_disp)    
    
    max_disp = (int(max_disp/5)+1)*5
    min_disp = int(min_disp/5)*5
    disp = np.clip(disp,min_disp,max_disp)
    disp = 255 * (disp-min_disp)/(max_disp-min_disp)
    
    disp=disp.astype(np.uint8)
    # disp = 255 - disp
    disp = cv2.applyColorMap(disp,cv2.COLORMAP_PLASMA)
    
    save_dir_split = save_dir.split('.png')
    save_dir = save_dir_split[0]+'_{}_{}'.format(str(max_disp),str(min_disp))+'.png'
    if not save_dir is None:
        cv2.imwrite(save_dir,disp)
    else:
        return disp


def dirs_walk(dir_list):
    '''
    output:
        all the files in dir_list
    '''
    file_list = []
    for dir in dir_list:
        paths = os.walk(dir)
        for path, dir_lst, file_lst in paths:
            file_lst.sort()
            for file_name in file_lst:
                file_path = os.path.join(path, file_name)
                file_list.append(file_path)
    file_list.sort()
    return file_list 

def disparity_transfer(disp,mode='left'):
    '''
    function: transform between left disparity and right disparity; there should be no occlusion
    input:
        disp: disparity map; should be a 2D array or tensor
        mode: the input disparity be left or right disparity
    output:
        transformed disparity map
    '''
    assert mode in ['left','right']
    if not isinstance(disp, torch.Tensor):
        disp=torch.from_numpy(disp)
    disp = disp.float()
    if mode == 'left':
        # should be negative disparity
        if torch.mean(disp)>0:
            disp=-disp
    elif mode == 'right':
        # should be positive disparity
        if torch.mean(disp)<0:
            disp=-disp
            
    # initialization
    H,W = disp.shape
    pixel_undecided = torch.ones((H,W))
    disp_trans = torch.zeros((H,W))
    max_disp = torch.max(torch.abs(disp))
    _,base = torch.meshgrid(torch.from_numpy(np.arange(H)),torch.from_numpy(np.arange(W)))
    target = disp + base
    all = np.array(np.arange(W)).reshape(-1,1)
    all = torch.from_numpy(all).unsqueeze(1).repeat(1, H, W).float()

    # (1) give unseen pixel 0 disparity 
    if mode == 'left':
        row_max_target,_ = torch.max(target,dim=1,keepdim=True)
        row_max_target = row_max_target.repeat(1,W)
        _ = base - row_max_target
        disp_trans[_>0] = 0
        pixel_undecided[_>0] = 0
    elif mode == 'right':
        row_min_target,_ = torch.min(target,dim=1,keepdim=True)
        row_min_target = row_min_target.repeat(1,W)
        _ = base - row_max_target
        disp_trans[_<0] = 0
        pixel_undecided[_<0] = 0
    
    # (2) target中找整数, 直接赋值
    target_int_abs = torch.abs(target-target.int())
    mask = torch.zeros((H,W))
    mask[target_int_abs == 0]=1
    # delete pixels out of boundary
    mask[target>W-1]=0
    mask[target<0]=0
    y0 = mask.nonzero()[:,0]
    # x1 = mask.nonzero()[:,1]
    mask = mask.bool()
    disp_trans = disp_trans.index_put(indices=[y0.long(),  target[mask].long()], values=disp[mask])
    pixel_undecided = pixel_undecided.index_put(indices=[y0.long(),  target[mask].long()], values=torch.tensor(0.))
    
    # (3) decide disparity for other pixels
    # (3.1) find nearset pixel of one side
    # in dist_all, index0: x_axis in disp_trans，index0, y-axis, index2: x-axis in disp 
    dist_all = all-target.unsqueeze(0).repeat(W,1,1)
    dist_all[dist_all>=0]=-W-max_disp-1
    dis,index = torch.max(dist_all,dim=2)
    left_weight = torch.abs(dis.T)
    left_weight = torch.exp(-torch.mul(left_weight,left_weight)/2/10**2)
    x_base = index.T.unsqueeze(0)/(W-1)
    y_base = torch.linspace(0, 1, H).repeat(1,
                W, 1).transpose(1, 2)
    flow_field = torch.stack((x_base, y_base), dim=3)
    left_nearest = F.grid_sample(disp.unsqueeze(0).unsqueeze(0), 2*flow_field - 1, mode='bilinear',
                            padding_mode='zeros',align_corners=True).squeeze()
    # (3.2) find nearset pixel of another side
    dist_all = all-target.unsqueeze(0).repeat(W,1,1)
    dist_all[dist_all<=0]=+W+max_disp+1
    dis,index = torch.min(dist_all,dim=2)
    
    right_weight = torch.abs(dis.T)
    right_weight = torch.exp(-(right_weight*right_weight)/2/10**2)
    x_base = index.T.unsqueeze(0)/(W-1)
    y_base = torch.linspace(0, 1, H).repeat(1,
                W, 1).transpose(1, 2)
    flow_field = torch.stack((x_base, y_base), dim=3)
    right_nearest = F.grid_sample(disp.unsqueeze(0).unsqueeze(0), 2*flow_field - 1, mode='bilinear',
                            padding_mode='zeros',align_corners=True).squeeze()
    ans = (right_weight*right_nearest+left_weight*left_nearest)/(right_weight+left_weight)
    disp_trans = disp_trans+ans*pixel_undecided
    return (-disp_trans).numpy()

# input img: cv2.imread() numpy
def get_disp(imgL, imgR, method,max_disp=None):

    '''
    fuction: generate disparity map with SGBM or BM algorithm
    input:
        imgL: left image; should be a 3D array
        imgR: right image; should be a 3D array
        method: method   
    '''
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    min_disp = 0
    # SGBM Parameters: wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    window_size = 5
    if not max_disp is None:
        max_disp=255

    max_disp = 36

    if method == 'SGBM':    
        img_channel = 3
        imgL=cv2.imread(imgL)
        imgR=cv2.imread(imgR)
        # imgL_flip = np.flip(imgL,1)
        # imgR_flip = np.flip(imgR,1)
        # print(imgL.shape)
        ori_H,ori_W,_ = imgL.shape
        W=int(ori_W/2)
        H=int(ori_H/2)
        imgL=cv2.resize(imgL,(W,H),interpolation=cv2.INTER_LINEAR)
        imgR=cv2.resize(imgR,(W,H),interpolation=cv2.INTER_LINEAR)

        left_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=max_disp,  # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=window_size,
            P1 = 8 * img_channel * window_size ** 2,
            P2 = 32 * img_channel * window_size ** 2,
            disp12MaxDiff = 12,
            preFilterCap = 0,
            uniquenessRatio = 12,
            speckleWindowSize = 60, # 60
            speckleRange = 32, # 32
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
    elif method == 'BM':
        imgL=cv2.imread(imgL, cv2.IMREAD_GRAYSCALE)
        imgR=cv2.imread(imgR, cv2.IMREAD_GRAYSCALE)
        left_matcher = cv2.StereoBM_create(numDisparities=max_disp,
                                        blockSize=window_size)
    # time0 = time.time()
    displ = left_matcher.compute(imgL, imgR).astype(np.float32)/16

    # dispr = left_matcher.compute(imgR_flip, imgL_flip).astype(np.float32)/16
    # dispr = np.flip(dispr,1)


    # print('runtime: ',time.time()-time0)
    return displ

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims,divide,mode=None):
        self.mode = mode
        self.ht, self.wd = dims[-2:]
        self.pad_ht = (((self.ht // divide) + 1) * divide - self.ht) % divide
        self.pad_wd = (((self.wd // divide) + 1) * divide - self.wd) % divide
        # self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
        # if mode == 'sintel':
        #     self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        # else:
        #     self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
        self._pad = [0, self.pad_wd, 0, self.pad_ht]

    def pad(self, *inputs):
        # if self.mode is None:
        #     return [F.pad(x, self._pad) for x in inputs]
        # else:
        #     return [F.pad(x, self._pad, mode=self.mode) for x in inputs]
        # return [F.pad(x, self._pad, mode='replicate') for x in inputs]
        return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs],self._pad[1],self._pad[3]

    def pad_numpy(self, *inputs):
        # if self.mode is None:
        #     return [F.pad(x, self._pad) for x in inputs]
        # else:
        #     return [F.pad(x, self._pad, mode=self.mode) for x in inputs]
        # return [F.pad(x, self._pad, mode='replicate') for x in inputs]
        return [np.pad(x,((0,self.pad_ht),(0,self.pad_wd),(0,0))) for x in inputs],self._pad[1],self._pad[3]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        # c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

# class sam_InputPadder:
#     """ Pads images such that dimensions are divisible by 8 """
#     def __init__(self, dims,divide,mode=None):
#         assert divide%16==0 # divide for SAM have to be n*16
#         self.ht, self.wd = dims[-2:]
#         pad_ht = (((self.ht // divide) + 1) * divide - self.ht) % divide
#         pad_wd = (((self.wd // divide) + 1) * divide - self.wd) % divide
#         # self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
#         # if mode == 'sintel':
#         #     self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
#         # else:
#         #     self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
#         self._pad = [0, pad_wd, 0, pad_ht]

#     def pad(self, *inputs):
#         # if self.mode is None:
#         #     return [F.pad(x, self._pad) for x in inputs]
#         # else:
#         #     return [F.pad(x, self._pad, mode=self.mode) for x in inputs]
#         # return [F.pad(x, self._pad, mode='replicate') for x in inputs]
#         return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs],self._pad[1],self._pad[3]

    # def unpad(self,x):
        # ht, wd = x.shape[-2:]
        # # c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        # c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        # return x[..., c[0]:c[1], c[2]:c[3]]

def io_disp_read(dir):   
    '''
    function: load disparity map from disparity file
    input:
        dir: dir of disparity file
    ''' 
    # load disp from npy
    if dir.endswith('npy'):
        disp = np.load(dir)
        disp = disp.astype(np.float32)
    elif dir.endswith('pfm'):
        with open(dir, 'rb') as pfm_file:
            header = pfm_file.readline().decode().rstrip()
            channels = 3 if header == 'PF' else 1
            dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
            if dim_match:
                width, height = map(int, dim_match.groups())
            else:
                raise Exception("Malformed PFM header.")
            scale = float(pfm_file.readline().decode().rstrip())
            if scale < 0:
                endian = '<' # littel endian
                scale = -scale
            else:
                endian = '>' # big endian
            disp = np.fromfile(pfm_file, endian + 'f')
            disp = np.reshape(disp, newshape=(height, width, channels))  
            disp[np.isinf(disp)] = 0
            disp = np.flipud(disp) 
            if channels == 1:
                disp = disp.squeeze(2)
        disp = disp.astype(np.float32)
    elif dir.endswith('png'):
        if 'Cre' in dir:
            disp = cv2.imread(dir, cv2.IMREAD_UNCHANGED)
            disp = disp.astype(np.float32) / 32
        elif 'middlebury' in dir:
            disp = cv2.imread(dir, -1)
            disp = disp.astype(np.float32)
        elif 'KITTI' in dir:
            disp = cv2.imread(dir, cv2.IMREAD_ANYDEPTH) / 256.0
            disp = disp.astype(np.float32)
        elif 'real_road' in dir:
            _bgr = cv2.imread(dir)
            R_ = _bgr[:, :, 2]
            G_ = _bgr[:, :, 1]
            B_ = _bgr[:, :, 0]
            normalized_= (R_ + G_ * 256. + B_ * 256. * 256.) / (256. * 256. * 256. - 1)
            disp = 500*normalized_
            disp = disp.astype(np.float32)
        elif 'vkitti' in dir:
            # read depth
            depth = cv2.imread(dir, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # in cm
            depth = (depth / 100).astype(np.float32)  # depth clipped to 655.35m for sky
            valid = (depth > 0) & (depth < 655)  # depth clipped to 655.35m for sky
            # convert to disparity
            focal_length = 725.0087  # in pixels
            baseline = 0.532725  # meter
            disp = baseline * focal_length / depth
            disp[~valid] = 0  # invalid as very small value        
    else:
        raise ValueError('unknown disp file type: {}'.format(dir))
    return disp

# recover the resolution of "input" from "reference"
def reso_recover(input,reference_dir):
    ori_img = cv2.imread(reference_dir)
    img_H,img_W = ori_img.shape[0],ori_img.shape[1]
    input = cv2.resize(input*img_W/input.shape[1],(img_W,img_H),interpolation=cv2.INTER_LINEAR)
    return input

def seed_record(img_W,img_H,Key_points_coordinate,disp_min=0):
    image1_seed = torch.zeros((img_H,img_W),device='cuda').long()
    # image2_seed = torch.zeros((img_H,img_W))
    co_row = Key_points_coordinate[:,1] == Key_points_coordinate[:,3]
    positive_disp = Key_points_coordinate[:,0] > Key_points_coordinate[:,2]
    saved = torch.logical_and(co_row,positive_disp)
    Key_points_coordinate = Key_points_coordinate[saved,:]
    image1_seed[Key_points_coordinate[:,1],Key_points_coordinate[:,0]]=Key_points_coordinate[:,0]-Key_points_coordinate[:,2]
    return image1_seed #,image2_seed

# resample the sim: coordinate based <--> disparity based
def sim_remap(sim):
    image_x = sim.shape[-1]
    w_base,d_base = torch.meshgrid(torch.from_numpy(np.arange(image_x)),torch.from_numpy(np.arange(image_x)))    
    d_base = w_base - d_base
    w_base = ((w_base)/(image_x-1)).unsqueeze(0).to(sim.device)
    d_base = ((d_base)/(image_x-1)).unsqueeze(0).to(sim.device)
    coords = torch.stack((d_base, w_base), dim=3)
    # re-coordinate sim from [H,W,W] to [H,W,D]
    sim = F.grid_sample(sim.unsqueeze(0), 2*coords - 1, mode='nearest',
                            padding_mode='zeros',align_corners=True).squeeze()
    sim[sim==0] = -1
    return sim

# return points in [n,2] [width,height]
def SparseDisp2Points(disp,remove_margin=False):

    if remove_margin:
        # remove the correspondences at the image margin
        disp[:,0]=0
        disp[:,-1]=0
        disp[0,:]=0
        disp[-1,:]=0
        # grid = torch.arange(0, disp.shape[1], device='cuda').unsqueeze(0).expand(disp.shape[0],disp.shape[1]) # [H,W]: H * 0~W
        # disp[disp==grid]=0

    _ = disp.nonzero() # [n,2]

    disp = disp[disp>0]
    points_A = torch.zeros(_.shape).cuda()
    points_B = torch.zeros(_.shape).cuda()
    points_A[:,1] = _[:,0]
    points_B[:,1] = _[:,0]
    points_A[:,0] = _[:,1]
    points_B[:,0] = points_A[:,0]-disp

    return (points_A.long().t(),points_B.long().t())

# return disp
def Points2SparseDisp(H,W,points_A,points_B):
    # point [W,H]
    disp=torch.zeros(size=(H,W)).long().cuda()
    disp[points_A[1,:],points_A[0,:]]=points_A[0,:]-points_B[0,:]
    return disp

def disp_remap_correct(disp,img_A,img_B,thr=10):
    mask = disp>0
    img_B = F.interpolate(img_B, size=disp.shape, mode='bilinear', align_corners=True)
    img_A = F.interpolate(img_A, size=disp.shape, mode='bilinear', align_corners=True)
    if torch.mean(img_A)<1:
        thr = thr/255
    est = single_image_warp(img_B,disp,tensor_output=True)
    diff = torch.abs(img_A - est).squeeze()
    disp[diff<torch.mean(diff[mask])]=0
    return disp

# confidence matrix initialization 
def sim_construct(feature_A,feature_B,remap=True,LR = False,R=False,down_size=None):
    if not R:
        d1 = feature_A/torch.sqrt(torch.sum(torch.square(feature_A), 0)).unsqueeze(0) # [C,H,W]
        d2 = feature_B/torch.sqrt(torch.sum(torch.square(feature_B), 0)).unsqueeze(0) # [C,H,W]
        # d1 = d1.detach().cpu()
        # d2 = d2.detach().cpu()
        sim = torch.einsum('ijk,ijh->jkh', d1, d2) # [H,W,W] 166,240,240

        if LR:
            sim_l = sim_remap(sim)

            sim_r = sim.permute(0,2,1).contiguous()
            sim_r = sim_r.flip([-1,-2])
            sim_r = sim_remap(sim_r)

            if not down_size is None:
                sim_l = sim_down_size(sim_l,down_size)
                sim_r = sim_down_size(sim_r,down_size)
            return sim_l.cuda(),sim_r.cuda()
        if remap:
            # return similarity volume with the 3rd dim at the disp
            sim = sim_remap(sim)
            if not down_size is None:
                sim = sim_down_size(sim,down_size)
            return sim

        else:
            # return similarity volume with the 3rd dim at the coordinate
            return sim 
    elif R:
        d1 = feature_A/torch.sqrt(torch.sum(torch.square(feature_A), 0)).unsqueeze(0) # [C,H,W]
        d2 = feature_B/torch.sqrt(torch.sum(torch.square(feature_B), 0)).unsqueeze(0) # [C,H,W]            
        d1 = torch.flip(d1,[-1])
        d2 = torch.flip(d2,[-1])
        sim_r = torch.einsum('ijk,ijh->jkh', d2, d1) # [H,W,W] 166,240,240
        sim_r = sim_remap(sim_r)
        if not down_size is None:
            sim_r = sim_down_size(sim_r,down_size)
        return sim_r


# def sim_construct_c2r(sim):
#     sim_r = sim.permute(0,2,1).contiguous()
#     sim_r = torch.flip(sim_r,[-1,-2])
#     sim_r = sim_remap(sim_r)
#     return sim_r

def get_pt_disp(image_y,image_x,points=None,disp=None,offset=None,ratio=1):
    # disp should be numpy
    assert not isinstance(disp, torch.Tensor)
    assert points is None or disp is None
    if points is None:
        _ = disp.nonzero()
        u = _[1]
        v = _[0]
        dxs = disp[_]
    else:
        u = points[0] # column # width
        v = points[1] # row # height
        dxs = points[2] # disp
    PT_disp = getPT(u,v,dxs,image_y,image_x)
    PT_disp = PT_disp*ratio
    for j in range(image_y):
        ans = np.min(PT_disp[j,:])
        PT_disp[j,:] = ans
    if not offset is None:
        PT_disp = PT_disp - offset
    PT_disp[PT_disp<0]=0
    return PT_disp

def getPT(u,v,d,vmax,umax):
    v_map_1 = np.mat(np.arange(0, vmax)) # 
    v_map_1_transpose = v_map_1.T # (1030, 1)
    umax_one = np.mat(np.ones(umax)).astype(int) # (1, 1720)
    v_map = v_map_1_transpose * umax_one # (1030, 1720)
    vmax_one = np.mat(np.ones(vmax)).astype(int)
    vmax_one_transpose = vmax_one.T # (1030, 1)
    u_map_1 = np.mat(np.arange(0, umax)) # (1, 1720)
    u_map = vmax_one_transpose * u_map_1 # (1030, 1720)
    Su = np.sum(u)
    Sv = np.sum(v)
    Sd = np.sum(d)
    Su2 = np.sum(np.square(u))
    Sv2 = np.sum(np.square(v))
    Sdu = np.sum(np.multiply(u, d))
    Sdv = np.sum(np.multiply(v, d))
    Suv = np.sum(np.multiply(u, v))
    n= len(u)
    beta0 = (np.square(Sd) * (Sv2 + Su2) - 2 * Sd * (Sv * Sdv + Su * Sdu) + n * (np.square(Sdv) + np.square(Sdu)))/2
    beta1 = (np.square(Sd) * (Sv2-Su2) + 2 * Sd * (Su*Sdu-Sv*Sdv) + n * (np.square(Sdv) - np.square(Sdu)))/2
    beta2 = -np.square(Sd) * Suv + Sd * (Sv * Sdu + Su * Sdv) - n * Sdv * Sdu
    gamma0 = (n * Sv2 + n * Su2 - np.square(Sv) - np.square(Su))/2
    gamma1 = (n * Sv2 - n * Su2 - np.square(Sv) + np.square(Su))/2
    gamma2 = Sv * Su - n * Suv
    A = (beta1 * gamma0 - beta0 * gamma1)
    B = (beta0 * gamma2 - beta2 * gamma0)
    C = (beta1 * gamma2 - beta2 * gamma1)
    delta = np.square(A) + np.square(B) - np.square(C)
    tmp1 = (A + np.sqrt(delta))/(B-C)
    tmp2 = (A - np.sqrt(delta))/(B-C)
    theta1 = math.atan(tmp1)
    theta2 = math.atan(tmp2)
    u=np.mat(u)
    v=np.mat(v)
    d=np.mat(d)
    d=d.T
    u=u.T
    v=v.T
    t1 = v * math.cos(theta1) - u * math.sin(theta1)
    t2 = v * math.cos(theta2) - u * math.sin(theta2)
    n_ones = np.ones(n).astype(int)
    n_ones = (np.mat(n_ones)).T
    T1 = np.hstack((n_ones, t1))
    T2 = np.hstack((n_ones, t2))
    f1 = d.T * T1 * np.linalg.inv (T1.T * T1) * T1.T * d
    f2 = d.T * T2 * np.linalg.inv (T2.T * T2) * T2.T * d
    if f1 < f2:
        theta = theta2
    else:
        theta = theta1
    t = v * math.cos(theta) - u * math.sin(theta)
    T = np.hstack((n_ones, t))
    a = np.linalg.inv(T.T * T) * T.T * d
    t_map = v_map * math.cos(theta) - u_map * math.sin(theta)
    newdisp = (a[0] + np.multiply(a[1], t_map))# - 20
    return newdisp

# def get_mutual_ncc_sim(left,right,ncc_rad=7,down_size=None,max_disp=None):
#     w = left.shape[-1]
#     if not down_size is None:
#         max_disp = int(w/down_size)

#     left_sim = get_ncc_sim(left,right,ncc_rad,max_disp)
#     print(left_sim.shape)
#     print((left.flip(-1)).shape)
#     print(left[0,0,10,-100:])
#     print(left.flip(-1)[0,0,10,:100])
#     print((right.flip(-1)).shape)
#     right_sim = get_ncc_sim(right.flip(-1),left.flip(-1),ncc_rad,max_disp)
#     print(right_sim.shape)

#     return left_sim,right_sim

def get_ncc_sim(left, right, ncc_rad = 4, max_disp = None):
    _,C,H,W = left.shape
    if max_disp is None:
        max_disp = W

    # ncc initial 
    ncc_pool =  torch.nn.AvgPool2d((ncc_rad*2+1), stride=1,padding=0)
    ncc_Unfold=torch.nn.Unfold(kernel_size=((ncc_rad*2+1),(ncc_rad*2+1)),dilation=1,padding=0,stride=1)
    # pad
    left_padded = F.pad(left, [ncc_rad,ncc_rad,ncc_rad,ncc_rad],mode='replicate') # [1,1, H，W]
    right_padded = F.pad(right, [ncc_rad,ncc_rad,ncc_rad,ncc_rad],mode='replicate') # [1,1, H，W]

    # avg
    left_avg = ncc_pool(left_padded) # [1,C,H,W]
    right_avg = ncc_pool(right_padded)
    left_avg = left_avg.unsqueeze(2) # [1,C,1,H,W]
    right_avg = right_avg.unsqueeze(2) 
    # unfold
    left_sum = ncc_Unfold(left_padded)
    left_sum = left_sum.view(1,C,(ncc_rad*2+1)*(ncc_rad*2+1),H,W) # [1,C,rad^,H,W]
    right_sum = ncc_Unfold(right_padded)
    right_sum = right_sum.view(1,C,(ncc_rad*2+1)*(ncc_rad*2+1),H,W) # [1,C,rad^,H,W]
    # minus
    left_minus = left_sum-left_avg # [1,C,rad^,H,W]
    right_minus = right_sum-right_avg

    var_left = torch.sum(torch.square(left_minus),dim=2) # [1,C,H,W]
    var_left[var_left==0]=0.01
    var_right = torch.sum(torch.square(right_minus),dim=2) # [1,C,H,W]
    var_right[var_right==0]=0.01
    
    # calculate
    ncc_b = torch.matmul(var_left.unsqueeze(-1),var_right.unsqueeze(-2)) # [1,C,H,W,W]
    ncc_b = torch.sqrt(ncc_b) # [1,C,H,W,W]

    # ncc_a =  torch.matmul(left_minus.permute(0,1,3,4,2).contiguous(),right_minus.permute(0,1,3,2,4).contiguous()) # [1,C,H,W,W] 
    Conf = (torch.matmul(left_minus.permute(0,1,3,4,2).contiguous(),right_minus.permute(0,1,3,2,4).contiguous()))/ncc_b # [1,C,H,W,W] 
    Conf = torch.mean(Conf,dim=1) # [1,H,W,W]

    Conf = sim_remap(Conf.squeeze()) # [H,W,W]
    Conf = Conf[:,:,:max_disp] 

    return Conf

# ### ranger ###
# def get_ncc_sim(left, right, ncc_rad = 1, max_disp = None):
#     _,C,H,W = left.shape
#     N = (2*ncc_rad+1)*(2*ncc_rad+1)

#     if max_disp is None:
#         max_disp = W

#     # ncc initial 
#     ncc_pool =  torch.nn.AvgPool2d((ncc_rad*2+1), stride=1,padding=0)
#     ncc_Unfold=torch.nn.Unfold(kernel_size=((ncc_rad*2+1),(ncc_rad*2+1)),dilation=1,padding=0,stride=1)
#     # pad
#     left_padded = F.pad(left, [ncc_rad,ncc_rad,ncc_rad,ncc_rad],mode='replicate') # [1,1, H，W]
#     right_padded = F.pad(right, [ncc_rad,ncc_rad,ncc_rad,ncc_rad],mode='replicate') # [1,1, H，W]

#     # avg
#     left_avg = ncc_pool(left_padded) # [1,C,H,W]
#     right_avg = ncc_pool(right_padded)
#     avg =  N*torch.matmul(left_avg.unsqueeze(-1),right_avg.unsqueeze(-2)) # [1,C,H,W,W]

#     # unfold
#     left_unfold = ncc_Unfold(left_padded)
#     left_unfold = left_unfold.view(1,C,(ncc_rad*2+1)*(ncc_rad*2+1),H,W) # [1,C,rad^,H,W]
#     right_unfold = ncc_Unfold(right_padded)
#     right_unfold = right_unfold.view(1,C,(ncc_rad*2+1)*(ncc_rad*2+1),H,W) # [1,C,rad^,H,W]
#     matmul = torch.matmul(left_unfold.permute(0,1,3,4,2).contiguous(),right_unfold.permute(0,1,3,2,4).contiguous())   # [1,C,H,W,W]

#     # var
#     left_minus = left_unfold-left_avg.unsqueeze(2) # [1,C,rad^,H,W]
#     right_minus = right_unfold-right_avg.unsqueeze(2)
#     var_left = torch.sum(torch.square(left_minus),dim=2) # [1,C,H,W]
#     var_left[var_left==0]=0.01
#     var_right = torch.sum(torch.square(right_minus),dim=2) # [1,C,H,W]
#     var_right[var_right==0]=0.01
#     var = torch.matmul(var_left.unsqueeze(-1),var_right.unsqueeze(-2)) # [1,C,H,W,W]
#     var = torch.sqrt(var) # [1,C,H,W,W]

#     Conf = (matmul - avg)/var
#     Conf = torch.mean(Conf,dim=1) # [1,H,W,W]
#     Conf = sim_remap(Conf.squeeze()) # [H,W,W]
#     Conf = Conf[:,:,:max_disp] 
#     return Conf

### ranger ###
def get_ncc_sim(left, right, ncc_rad = 3, max_disp = None):
    _,C,H,W = left.shape
    N = (2*ncc_rad+1)*(2*ncc_rad+1)
    if max_disp is None:
        max_disp = W
    Conf = left.new_zeros(H,W,max_disp)-1

    # ncc initial 
    ncc_pool =  torch.nn.AvgPool2d((ncc_rad*2+1), stride=1,padding=0)
    ncc_Unfold=torch.nn.Unfold(kernel_size=((ncc_rad*2+1),(ncc_rad*2+1)),dilation=1,padding=0,stride=1)
    # pad
    left_padded = F.pad(left, [ncc_rad,ncc_rad,ncc_rad,ncc_rad],mode='replicate') # [1,1, H，W]
    right_padded = F.pad(right, [ncc_rad,ncc_rad,ncc_rad,ncc_rad],mode='replicate') # [1,1, H，W]

    # avg
    left_avg = ncc_pool(left_padded) # [1,C,H,W]
    right_avg = ncc_pool(right_padded)
    # unfold
    left_unfold = ncc_Unfold(left_padded)
    left_unfold = left_unfold.view(1,C,(ncc_rad*2+1)*(ncc_rad*2+1),H,W) # [1,C,rad^,H,W]
    # left_unfold = left_unfold.permute(0,1,3,4,2).contiguous()
    right_unfold = ncc_Unfold(right_padded)
    right_unfold = right_unfold.view(1,C,(ncc_rad*2+1)*(ncc_rad*2+1),H,W) # [1,C,rad^,H,W]
    # right_unfold = right_unfold.permute(0,1,3,2,4).contiguous()
    # var
    left_minus = left_unfold-left_avg.unsqueeze(2) # [1,C,rad^,H,W]
    right_minus = right_unfold-right_avg.unsqueeze(2)
    var_left = torch.sum(torch.square(left_minus),dim=2) # [1,C,H,W]
    var_left[var_left==0]=0.01
    var_right = torch.sum(torch.square(right_minus),dim=2) # [1,C,H,W]
    var_right[var_right==0]=0.01

    for i in range(max_disp):
        if i > 0:
            conf_ = (((left_unfold[:,:,:,:,i:]*right_unfold[:,:,:,:,:-i]).sum(dim=2) - N*left_avg[:,:,:,i:]*right_avg[:,:,:,:-i])/torch.sqrt(var_left[:,:,:,i:]*var_right[:,:,:,:-i]))
            Conf[:, i:, i] = torch.mean(conf_,dim=1).squeeze()
        else:
            conf_ = (((left_unfold[:,:,:,:,:]*right_unfold[:,:,:,:,:]).sum(dim=2) - N*left_avg[:,:,:,:]*right_avg[:,:,:,:])/torch.sqrt(var_left[:,:,:,:]*var_right[:,:,:,:]))
            Conf[:, i:, i] = torch.mean(conf_,dim=1).squeeze()
    return Conf

def writePFM(file, image, scale=1):
    file = open(file, 'wb')
 
    color = None
 
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
 
    image = np.flipud(image)
 
    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
 
    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))
 
    endian = image.dtype.byteorder
 
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale
 
    file.write('%f\n'.encode() % scale)
 
    image.tofile(file)

def save_disp_results(disp_dir,png_dir,result,max_disp=None,min_disp=None,display=True,PLA=False):
    if display:
        print('save in {} and {}, shape = {}'.format(png_dir,disp_dir,result.shape))
    if disp_dir is not None:
        if 'npy' in disp_dir:
            np.save(disp_dir, result)
        elif 'pfm' in disp_dir:
            writePFM(disp_dir,result.astype(np.float32))
    if png_dir is not None:
        if not PLA:
            disp_vis(png_dir,result,max_disp,min_disp)
        else:
            disp_vis_PLA(png_dir,result,max_disp)

def sim_down_size(sim,down_size=1):
    img_x = sim.shape[-1]
    max_disp = int(img_x/down_size)
    # max_disp = img_x
    sim = sim[:,:,:max_disp]
    return sim

def sim_restore(sim,value=-1):
    H,W,D = sim.shape
    if W==D:
        return sim
    expand = torch.zeros((H,W,W-D),device='cuda')-1
    expand = expand+value
    sim = torch.cat((sim,expand),-1)
    return sim

def remove_edges(input,edge=3):
    # [H,W]
    if len(input.shape) == 2:
        H,W = input.shape
        input = input[edge:H-edge,edge:W-edge]
        return input
    # [H,W,3]
    elif len(input.shape) == 3:
        H,W,_ = input.shape
        input = input[edge:H-edge,edge:W-edge,:]
        return input
    else:
        raise ValueError('empty')
    
def KITTI_test_submission(save_dir,disp):
    save_dir = save_dir.replace('vis','submitted/disp_0')
    save_dir_base = '/'.join(save_dir.split('/')[:-1])
    if not os.path.exists(save_dir_base):
        os.makedirs(save_dir_base)
    disp = np.clip(disp,0,255)
    disp = disp*256.
    disp = disp.astype(np.uint16)
    skimage.io.imsave(save_dir, disp)

def middeval3_submission(save_dir,disp):
    split = save_dir.split('/')
    method = split[-2]
    save_dir = '/'.join(split[:4])+'/submit'+'/'+split[4]+'/{}'.format(method)+'/'+split[5]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,'disp0{}.pfm'.format(method))
    save_disp_results(save_path,None,disp,display=False)
    save_path_time = os.path.join(save_dir,'time{}.txt'.format(method))
    with open(save_path_time,'w') as f:
        f.write('999')

def test_submission(save_dir,disp):
    if 'KITTI' in save_dir and 'testing' in save_dir:
        KITTI_test_submission(save_dir,disp)
    if 'MiddEval3' in save_dir and 'test' in save_dir:
        middeval3_submission(save_dir,disp)


# def KITTI_2015_test_submission(save_dir,disp):
#     save_dir = save_dir.replace('vis','submitted/disp_0')
#     save_dir_base = '/'.join(save_dir.split('/')[:-1])
#     print(save_dir_base)
#     if not os.path.exists(save_dir_base):
#         os.makedirs(save_dir_base)
#     disp = np.clip(disp,0,255)
#     disp_save = disp.copy()
#     disp_save = disp_save*256.
#     disp_save = disp_save.astype(np.uint16)
#     cv2.imwrite(save_dir,disp_save)
#     skimage.io.imsave(save_dir, disp_save)
#     disp_read = cv2.imread(save_dir, cv2.IMREAD_ANYDEPTH) / 256.0
#     disp_read = disp_read.astype(np.float32)
#     print(np.mean(np.abs(disp_read-disp)))
    

    


# def results_decouple(results,img_H,img_W,reference_dir):
#     mkpts0 = results['points_A'].T # [n,2] numpy
#     mkpts1 = results['points_B'].T
#     seed_disp = results['disp']

#     seed_disp = seed_disp.detach().cpu().numpy()
#     key_points_coordinate = np.concatenate((mkpts0,mkpts1),axis=1)
#     img1_seed,img2_seed = seed_record(img_W,img_H,key_points_coordinate) # [H,W]
#     seed_disp = reso_recover(seed_disp,reference_dir)

#     return seed_disp,img1_seed

# 

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}