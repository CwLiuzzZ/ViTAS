import torch.nn as nn
from ModelCreated.ViTAS.ViTAS_module.Uni_CA.CA_utils import feature_add_position_cross_feature
from itertools import repeat
import collections.abc

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class PAFM(nn.Module):
    def __init__(self, 
                 channels = [48,64,192,160],
                 num_heads = 8,
                 UsePatch = False,
                 UseWeight = True
                 ):
        super().__init__()    
        self.channels = channels   
        self.UsePatch = UsePatch
        self.UseWeight = UseWeight
        self.drop_path = DropPath(0.1)
        self.patch = nn.ModuleList([nn.Conv2d(channel, channel, kernel_size=(3,3), stride=1,padding=1,padding_mode='replicate') for channel in self.channels[:3]]) if UsePatch else nn.ModuleList([nn.Identity()]*3)
        self.SWCA = nn.ModuleList([CrossAttention(self.channels[i],self.channels[i+1],num_heads=num_heads, UseWeight=UseWeight) for i in range(len(self.channels)-1)]) # i = 0,1,2
                    
    def fuse(self,i,x_l, x_r, x_l_high, x_r_high): # i=3,2,1
        x_l_high_patched = self.patch[i-1](x_l_high)
        x_r_high_patched = self.patch[i-1](x_r_high)
        
        x_l, x_r, x_l_high_patched, x_r_high_patched, = feature_add_position_cross_feature(x_l, x_r, x_l_high_patched, x_r_high_patched,self.channels[i],self.channels[i-1])
        x_l = x_l_high + self.drop_path(self.SWCA[i-1](x_l_high_patched,x_l))
        x_r = x_r_high + self.drop_path(self.SWCA[i-1](x_r_high_patched,x_r))        
        return x_l,x_r

class CrossAttention(nn.Module):
    
    def __init__(self, dim, channel, num_heads=8, qkv_bias=True,UseWeight=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.projq = nn.Linear(dim, dim, bias=qkv_bias) # high res
        self.projqv = nn.Linear(dim, dim, bias=qkv_bias) # high res
        self.projk = nn.Linear(channel, dim, bias=qkv_bias) # low res
        self.projv = nn.Linear(channel, dim, bias=qkv_bias) # low res
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim) # final: high res
        self.DANE = DANE(dim=int(dim/num_heads), reduction=2)    
        
    def forward(self, source, target): # source for high resolution, target for low resolution        
        B,C0,H0,W0 = target.shape
        B,C,H,W = source.shape
        N0 = H0*W0 # N0 = N/4 = H*W/4
        
        source = source.reshape(B,C,H0,2,W0,2).permute(0,2,4,3,5,1).reshape(B,N0,4,C) # B,C,H,W -> B,H/2,W/2,2,2,C -> B,N0,4,C       
        target = target.reshape(B,C0,N0).permute(0,2,1).unsqueeze(-2) # # B,C0,H0,W0 -> B,N0,1,C0
        # source_ = source_.reshape(B,C,H0,2,W0,2).permute(0,2,4,3,5,1).reshape(B,N0,4,C) # B,C,H,W -> B,H/2,W/2,2,2,C -> [B,N0,4,C] 
    
        q = self.projq(source).reshape(B,N0,4,self.num_heads, C// self.num_heads).transpose(-2,-3) # -> [B,N0,head,4,C/head]
        vq = self.projqv(source).reshape(B,N0,4,self.num_heads, C// self.num_heads).transpose(-2,-3) # -> [B,N0,head,4,C/head]
        k = self.projk(target).reshape(B,N0,1,self.num_heads, C// self.num_heads).transpose(-2,-3) # -> [B,N0,head,1,C/head]
        v = self.projv(target).reshape(B,N0,1,self.num_heads, C// self.num_heads).transpose(-2,-3) # -> [B,N0,head,1,C/head]
    
        attn = (q @ k.transpose(-2, -1)) * self.scale # -> [B.N0,head,4,1]
        attn = attn.softmax(dim=-2) # -> [B.N0,head,4,1]     
        source_w = self.DANE(q,k) # [B,N0,head,1,C/head]
        x = (attn*v) # [B.N0,head,4,C/head]
        x = (1-source_w)*(attn*v)*4+source_w*vq # [B,N0,head,4,C/head]
        x = x.transpose(-2,-3).reshape(B,N0,4,C) # [B,N0,4,C]
        x = self.norm(self.proj(x)) # -> [B,N0,4,C]
        x = x.reshape(B,H0,W0,2,2,C).permute(0,5,1,3,2,4).reshape(B,C,H,W).contiguous() # B,H/2,W/2,4,C -> B,C,H,W
        return x

class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(self,channels=1024,flatten_embedding: bool = True,) -> None:
        super().__init__()
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(channels, channels, kernel_size=(3,3), stride=1,padding=1,padding_mode='replicate')
        
    def forward(self, x):
        _, _, H, W = x.shape
        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

class DANE(nn.Module):
    def __init__(self, dim, reduction=8):
        super(DANE, self).__init__()
        
        self.dim = dim
        # self.channel = channel
        self.fc_spatial = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1, bias=False),
        )
        self.avg_pool_spatial = nn.AdaptiveAvgPool2d((1,1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_channel = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.LayerNorm(dim//reduction),
            nn.Linear(dim // reduction, dim, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    # def forward(self, x_tf,x_cnn):
    def forward(self, source, target):
        # source [B,N,h,4,C]
        # target [B,N,h,1,C]
        B,N0,h,_,_ = source.shape
        source = source.permute(0,2,1,3,4) # [B,N,h,4,C] -> [B,h,N,4,C]
        target = target.permute(0,2,4,3,1) # [B,N,h,1,C] -> [B,h,C,1,N]
        
        x_spatial_mask = self.fc_spatial(source) # [B,h,N,4,C] -> [B,h,N,4,1]        
        x_spatial_mask = self.avg_pool_spatial(x_spatial_mask).squeeze(-1) # [B,h,N,4,1] -> [B,h,N,1]
        x_channel_mask = self.fc_channel(self.avg_pool(target).squeeze(-1).permute(0,1,3,2)) # [B,h,C,1,N] -> [B,h,1,C]
        source_w = self.sigmoid(x_spatial_mask.expand(B,h,N0,self.dim) + x_channel_mask.expand(B,h,N0,self.dim)) # [B,h,N0,C]
        return source_w.permute(0,2,1,3).unsqueeze(-2) # B,N0,h,1,C
    

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
