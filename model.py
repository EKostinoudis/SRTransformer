'''
Based on Restormer
https://github.com/swz30/Restormer
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MDTA(nn.Module):
    '''
    multi-Dconv head transposed attention
    '''
    def __init__(self, channels, num_heads, bias=False):
        super().__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # point-wise
        self.qkv_p = nn.Conv2d(channels, channels*3, kernel_size=1, bias=bias)

        # depth-wise
        self.qkv_d = nn.Conv2d(channels*3, channels*3, kernel_size=3, padding=1, groups=channels*3, bias=bias)

        self.projection = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_d(self.qkv_p(x)) # shape: batch, channels*3, height, width
        q, k, v = qkv.chunk(3, dim=1)   # shape: batch, channels,   height, width

        q = q.view(b, self.num_heads, c//self.num_heads, h*w)
        k = k.view(b, self.num_heads, c//self.num_heads, h*w)
        v = v.view(b, self.num_heads, c//self.num_heads, h*w)

        # normalize
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).view(b, c, h, w)
        out = self.projection(out)
        return out

class GDFN(nn.Module):
    '''
    Gated-Dconv Feed-Forward Network
    '''
    def __init__(self, channels, gamma, bias=False):
        super().__init__()

        self.conv_p_1 = nn.Conv2d(channels, channels*gamma*2, kernel_size=1, bias=bias)
        self.conv_d = nn.Conv2d(channels*gamma*2, channels*gamma*2, kernel_size=3, padding=1, groups=channels*gamma*2, bias=bias)
        self.conv_p_2 = nn.Conv2d(channels*gamma, channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x1, x2 = self.conv_d(self.conv_p_1(x)).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.conv_p_2(x)

class ChannelLayerNorm(nn.Module):
    '''
    Apply layer normalization along the channel's dimension
    '''
    def __init__(self, channels):
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # b, c, h, w -> b, h, w, c
        x = self.layernorm(x)
        x = x.permute(0, 3, 1, 2) # b, h, w, c -> b, c, h, w
        return x


class Transformer_Block(nn.Module):
    '''
    Gated-Dconv Feed-Forward Network
    '''
    def __init__(self, channels, num_heads, gamma, bias=False):
        super().__init__()
        self.layernorm_mdta = ChannelLayerNorm(channels)
        self.mdta = MDTA(channels, num_heads, bias=bias)
        self.layernorm_gdfn = ChannelLayerNorm(channels)
        self.gdfn = GDFN(channels, gamma, bias=bias)

    def forward(self, x):
        x = x + self.mdta(self.layernorm_mdta(x))
        x = x + self.gdfn(self.layernorm_gdfn(x))
        return x

class Downsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
            )

    def forward(self, x):
        return self.block(x)

class Upsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features*2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            )

    def forward(self, x):
        return self.block(x)

class SRTransformer(nn.Module):
    def __init__(self, 
                 channels=3, 
                 features=16,
                 num_blocks=[2,4,4],
                 num_heads=[1,2,4],
                 gamma=3, 
                 bias=True
                ):
        super().__init__()

        # low-level feature embeddings
        self.conv_start = nn.Conv2d(channels, features, kernel_size=3, padding=1, bias=bias)
        self.up_0 = Upsample(features)
        curr_features = features // 2

        # encoder 1
        self.enc_1 = nn.Sequential(*[Transformer_Block(curr_features, num_heads[0], gamma, bias=bias) for _ in range(num_blocks[0])])
        self.down_1 = Downsample(curr_features)
        curr_features *= 2

        # encoder 2
        self.enc_2 = nn.Sequential(*[Transformer_Block(curr_features, num_heads[1], gamma, bias=bias) for _ in range(num_blocks[1])])
        self.down_2 = Downsample(curr_features)
        curr_features *= 2

        # encoder 3
        self.enc_3 = nn.Sequential(*[Transformer_Block(curr_features, num_heads[2], gamma, bias=bias) for _ in range(num_blocks[2])])

        # decoder 2
        self.up_2 = Upsample(curr_features)
        curr_features //= 2
        self.reduce_channels_2 = nn.Conv2d(curr_features*2, curr_features, kernel_size=1, bias=bias)
        self.dec_2 = nn.Sequential(*[Transformer_Block(curr_features, num_heads[1], gamma, bias=bias) for _ in range(num_blocks[1])])

        # decoder 1
        self.up_1 = Upsample(curr_features)
        curr_features //= 2
        self.reduce_channels_1 = nn.Conv2d(curr_features*2, curr_features, kernel_size=1, bias=bias)
        self.dec_1 = nn.Sequential(*[Transformer_Block(curr_features, num_heads[0], gamma, bias=bias) for _ in range(num_blocks[0])])

        # final convolution
        self.conv_end = nn.Conv2d(curr_features, channels, kernel_size=3, padding=1, bias=bias)

    '''
        self.init_params()

    def init_params(self):
        for _, data in self.named_parameters():
            if data.dim() > 1:
                nn.init.xavier_uniform_(data)
    '''
    
    def forward(self, x):
        x = self.up_0(self.conv_start(x))

        # encoders
        x1 = self.enc_1(x)
        x2 = self.enc_2(self.down_1(x1))
        x3 = self.enc_3(self.down_2(x2))

        # decoders
        x = self.dec_2(self.reduce_channels_2(torch.cat([self.up_2(x3), x2], 1)))
        x = self.dec_1(self.reduce_channels_1(torch.cat([self.up_1(x), x1], 1)))

        x = self.conv_end(x)
        return x

class SRTransformer2(nn.Module):
    def __init__(self, 
                 channels=3, 
                 features=16,
                 num_blocks=[2,2,4],
                 num_heads=[1,2,4],
                 refinement_blocks = 2,
                 gamma=3, 
                 bias=True
                ):
        super().__init__()

        # low-level feature embeddings
        self.conv_start = nn.Conv2d(channels, features, kernel_size=3, padding=1, bias=bias)
        self.up_0 = Upsample(features)
        curr_features = features // 2

        # encoder 1
        self.enc_1 = nn.Sequential(*[Transformer_Block(curr_features, num_heads[0], gamma, bias=bias) for _ in range(num_blocks[0])])
        self.down_1 = Downsample(curr_features)
        curr_features *= 2

        # encoder 2
        self.enc_2 = nn.Sequential(*[Transformer_Block(curr_features, num_heads[1], gamma, bias=bias) for _ in range(num_blocks[1])])
        self.down_2 = Downsample(curr_features)
        curr_features *= 2

        # encoder 3
        self.enc_3 = nn.Sequential(*[Transformer_Block(curr_features, num_heads[2], gamma, bias=bias) for _ in range(num_blocks[2])])

        # decoder 2
        self.up_2 = Upsample(curr_features)
        curr_features //= 2
        self.reduce_channels_2 = nn.Conv2d(curr_features*2, curr_features, kernel_size=1, bias=bias)
        self.dec_2 = nn.Sequential(*[Transformer_Block(curr_features, num_heads[1], gamma, bias=bias) for _ in range(num_blocks[1])])

        # decoder 1
        self.up_1 = Upsample(curr_features)
        curr_features //= 2
        self.dec_1 = nn.Sequential(*[Transformer_Block(curr_features*2, num_heads[0], gamma, bias=bias) for _ in range(num_blocks[0])])

        # refinement
        self.refinement = nn.Sequential(*[Transformer_Block(curr_features*2, num_heads[0], gamma, bias=bias) for _ in range(refinement_blocks)])

        # final convolution
        self.conv_end = nn.Conv2d(curr_features*2, channels, kernel_size=3, padding=1, bias=bias)
    
    def forward(self, x):
        x = self.up_0(self.conv_start(x))

        # encoders
        x1 = self.enc_1(x)
        x2 = self.enc_2(self.down_1(x1))
        x3 = self.enc_3(self.down_2(x2))

        # decoders
        x = self.dec_2(self.reduce_channels_2(torch.cat([self.up_2(x3), x2], 1)))
        x = self.dec_1(torch.cat([self.up_1(x), x1], 1))

        x = self.refinement(x)

        x = self.conv_end(x)
        return x