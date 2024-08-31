import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFA(nn.Module):
    def __init__(self,
                 in_channels = [64,128,256,160], 
                 out_channels = [48,64,192,160],
                 ):
        super().__init__()
        # self.convblocks = {}
        self.num_layers = len(in_channels) - 1

        self.dec_0 = self.make_dec_0(out_channels)
        self.sdfa_0 = nn.ModuleList([nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channel, out_channel, kernel_size=3, bias=False),
                nn.ELU(inplace=True)) for in_channel,out_channel in zip(in_channels[:3],out_channels[:3])])  
        self.sdfa_1 = nn.ModuleList([nn.Sequential(
                nn.Conv2d(out_channel * 2, out_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channel, 2, 3, bias=False)) for out_channel in out_channels[:3]])  
        self.sdfa_2 = nn.ModuleList([nn.Sequential(
                nn.Conv2d(out_channel * 2, out_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channel, 2, 3, bias=False)) for out_channel in out_channels[:3]])  
        self.dec_1 = nn.ModuleList([nn.Sequential(nn.ReflectionPad2d(1),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, bias=True),
                nn.ELU(inplace=True)) for out_channel in out_channels[:3]])
        for sdfa in self.sdfa_1:
            sdfa[-1].weight.data.zero_()
        for sdfa in self.sdfa_2:
            sdfa[-1].weight.data.zero_()

        # for i in range(self.num_layers, 0, -1): # 3,2,1
        #     self.convblocks[("dec", i, 0)] = nn.Sequential(
        #         nn.ReflectionPad2d(1),
        #         nn.Conv2d(in_channels[i], out_channels[i-1], kernel_size=3, bias=True),
        #         nn.ELU(inplace=True))
        #     self.convblocks[("dec-sdfa", i, 0)] = nn.Sequential(
        #         nn.ReflectionPad2d(1),
        #         nn.Conv2d(in_channels[i-1], out_channels[i-1], kernel_size=3, bias=False),
        #         nn.ELU(inplace=True))
        #     self.convblocks[("dec-sdfa", i, 1)] = nn.Sequential(
        #         nn.Conv2d(out_channels[i-1] * 2, out_channels[i-1], kernel_size=1, bias=False),
        #         nn.BatchNorm2d(out_channels[i-1]),
        #         nn.ELU(),
        #         nn.ReflectionPad2d(1),
        #         nn.Conv2d(out_channels[i-1], 2, 3, bias=False))
        #     self.convblocks[("dec-sdfa", i, 2)] = nn.Sequential(
        #         nn.Conv2d(out_channels[i-1] * 2, out_channels[i-1], kernel_size=1, bias=False),
        #         nn.BatchNorm2d(out_channels[i-1]),
        #         nn.ELU(),
        #         nn.ReflectionPad2d(1),
        #         nn.Conv2d(out_channels[i-1], 2, 3, bias=False))
        #     self.convblocks[("dec-sdfa", i, 1)][-1].weight.data.zero_()
        #     self.convblocks[("dec-sdfa", i, 2)][-1].weight.data.zero_()
            
        #     self.convblocks[("dec", i, 1)] = nn.Sequential(
        #         nn.ReflectionPad2d(1),
        #         nn.Conv2d(out_channels[i-1], out_channels[i-1], kernel_size=3, bias=True),
        #         nn.ELU(inplace=True))
        # self._convs = nn.ModuleList(list(self.convblocks.values()))
           
    def make_dec_0(self,out_channels):
        dec_1 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels[1], out_channels[0], kernel_size=3, bias=True),
                nn.ELU(inplace=True))
        dec_2 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels[2], out_channels[1], kernel_size=3, bias=True),
                nn.ELU(inplace=True))
        if len(out_channels)==3:
            return nn.ModuleList([dec_1,dec_2])  
        dec_3 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels[3], out_channels[2], kernel_size=3, bias=True),
                nn.ELU(inplace=True))
        return nn.ModuleList([dec_1,dec_2,dec_3])    
        
        
    def fuse(self,i, x_l, x_r, x_l_high, x_r_high): # x_l: left, x_r: right
        x_l = self.dec_0[i-1](x_l)
        x_r = self.dec_0[i-1](x_r)
    
        tar_shape = x_l_high.shape
        x_l = _upsample(x_l, tar_shape) # up sample
        x_r = _upsample(x_r, tar_shape)
        x_l_enc = self.sdfa_0[i-1](x_l_high)
        x_l_con = torch.cat((x_l, x_l_enc), 1)
        delta1_s = self.sdfa_1[i-1](x_l_con)
        x_l = bilinear_interpolate_torch_gridsample(
            x_l, x_l.shape[2:], delta1_s)
        delta2_s_enc = self.sdfa_2[i-1](x_l_con)
        x_l_enc = bilinear_interpolate_torch_gridsample(
            x_l_enc, x_l_enc.shape[2:], delta2_s_enc)
        x_l = [x_l + x_l_enc]

        x_r_enc = self.sdfa_0[i-1](x_r_high)
        x_r_con = torch.cat((x_r, x_r_enc), 1)
        delta1_o = self.sdfa_1[i-1](x_r_con)
        x_r = bilinear_interpolate_torch_gridsample(
            x_r, x_r.shape[2:], delta1_o)
        delta2_o_enc = self.sdfa_2[i-1](x_r_con)
        x_r_enc = bilinear_interpolate_torch_gridsample(
            x_r_enc, x_r_enc.shape[2:], delta2_o_enc)
        x_r = [x_r + x_r_enc]
        x_l = torch.cat(x_l, 1)
        x_r = torch.cat(x_r, 1)
        x_l = self.dec_1[i-1](x_l)
        x_r = self.dec_1[i-1](x_r)
        return x_l,x_r
            
    # def _upsample(self, x, shape, is_bilinear=False):
    #     if is_bilinear:
    #         return F.interpolate(x, size=shape[2:], mode="bilinear", align_corners=False)
    #     else:
    #         return F.interpolate(x, size=shape[2:], mode="nearest")
        
    # def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
    #     out_h, out_w = size
    #     n, c, h, w = input.shape
    #     s = 2.0
    #     norm = torch.tensor([[[[(out_w - 1) / s, (out_h - 1) / s]]]
    #                          ]).type_as(input).to(input.device)
    #     w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
    #     h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
    #     grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
    #     grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
    #     grid = grid + delta.permute(0, 2, 3, 1) / norm

    #     output = F.grid_sample(input, grid, align_corners=True)
    #     return output

def _upsample(x, shape, is_bilinear=False):
    if is_bilinear:
        return F.interpolate(x, size=shape[2:], mode="bilinear", align_corners=False)
    else:
        return F.interpolate(x, size=shape[2:], mode="nearest")

def bilinear_interpolate_torch_gridsample(input, size, delta=0):
    out_h, out_w = size
    n, c, h, w = input.shape
    s = 2.0
    norm = torch.tensor([[[[(out_w - 1) / s, (out_h - 1) / s]]]
                            ]).type_as(input).to(input.device)
    w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
    h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
    grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
    grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
    grid = grid + delta.permute(0, 2, 3, 1) / norm

    output = F.grid_sample(input, grid, align_corners=True)
    return output

# class Conv3x3(nn.Module):
#     """Layer to pad and convolve input
#        from https://github.com/nianticlabs/monodepth2
#     """
#     def __init__(self, in_channels, out_channels, use_refl=True, bias=True):
#         super(Conv3x3, self).__init__()

#         if use_refl:
#             self.pad = nn.ReflectionPad2d(1)
#         else:
#             self.pad = nn.ZeroPad2d(1)
#         self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, bias=bias)

#     def forward(self, x):
#         out = self.pad(x)
#         out = self.conv(out)
#         return out

# class ConvBlock(nn.Module):
#     """Layer to perform a convolution followed by ELU
#        from https://github.com/nianticlabs/monodepth2
#     """
#     def __init__(self, in_channels, out_channels, bn=False, nonlin=True,):
#         super(ConvBlock, self).__init__()

#         self.conv = Conv3x3(in_channels, out_channels)
#         if nonlin:
#             self.nonlin = nn.ELU(inplace=True)
#         else:
#             self.nonlin = None
#         if bn:
#             self.bn = nn.BatchNorm2d(out_channels)
#         else:
#             self.bn = None

#     def forward(self, x, wo_conv=False):
#         if not wo_conv:
#             out = self.conv(x)
#         else:
#             out = x
#         if self.bn is not None:
#             out = self.bn(out)
#         if self.nonlin is not None:
#             out = self.nonlin(out)
#         return out