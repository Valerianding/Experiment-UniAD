import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import torch.nn as nn

class CVT_DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor, upsample, with_relu=True):
        super().__init__()

        dim = out_channels // factor

        if upsample:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None
        
        self.with_relu = with_relu
        if self.with_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up
        if self.with_relu:
            return self.relu(x)
        return x