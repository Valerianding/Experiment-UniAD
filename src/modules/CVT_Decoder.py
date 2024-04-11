import CVT_DecoderBlock
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from src.utils.builder import build_conv_layer
from collections import OrderedDict
import ConvModule

class CVT_Decoder(nn.Module):
    def __init__(self, dim, blocks, residual=True, factor=2, upsample=True, use_checkpoint=False, init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(CVT_Decoder, self).__init__()

        layers = []
        channels = dim

        for i, out_channels in enumerate(blocks):
            with_relu = i < len(blocks) - 1  # if not last block, with relu
            layer = CVT_DecoderBlock(channels, out_channels, dim, residual, factor, upsample, with_relu=with_relu)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels
        self.use_checkpoint = use_checkpoint
        
        if init_cfg is None:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x):
        b, t = x.size(0), x.size(1)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        y = x
        for layer in self.layers:
            if self.use_checkpoint:
                y = checkpoint(layer, y, x)
            else:
                y = layer(y, x)
        
        y = rearrange(y, '(b t) c h w -> b t c h w', b=b, t=t)
        return y
