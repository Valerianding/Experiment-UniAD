import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from src.utils.builder import build_conv_layer
from collections import OrderedDict
import ConvModule


class SimpleConv2d(nn.module):
    def __init__(self, in_channels, 
                       out_channels, 
                       
                       conv_channels=64,
                       num_conv=1,
                       conv_cfg=dict(type='Conv2d'),
                       norm_cfg=dict(type='BN2d'),
                       bias='auto',
                       init_cfg=None,
                       ):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(SimpleConv2d, self).__init__()
        self.out_channels = out_channels
        if num_conv == 1:
            conv_channels = in_channels

        conv_layers = []
        c_in = in_channels
        for i in range(num_conv-1):
            conv_layers.append(
                ConvModule(
                    c_in,
                    conv_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                )
            )
            c_in = conv_channels
        # No norm and relu in last conv
        conv_layers.append(
            build_conv_layer(
                conv_cfg,
                conv_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
        )
        self.conv_layers = nn.Sequential(*conv_layers)

        if init_cfg is None:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x):
        b, c_in, h_in, w_in = x.size()
        out = self.conv_layers(x)
        assert out.size() == (b, self.out_channels, h_in, w_in)  # sanity check
        return out