# ============================================================
# DEMO CODE: SSPANet Attention Module
#
# Based on the paper:
# "Enhancing brain tumor classification with a novel attention-based
#  explainable deep learning framework"
# Hasan, M.J., Hasan, M., Akter, S., Mahi, A.B.S., & Uddin, M.P.
# Biomedical Signal Processing and Control, Elsevier, 2026
# DOI: https://doi.org/10.1016/j.bspc.2025.108636
#
# Code Owner: Md Jahid Hasan (RMIT University)
# License: MIT License
# ============================================================





import torch
import torch.nn as nn

import torch
from torch import nn
from torch.nn import functional as F

def calculate_std_from_variance(variance, epsilon=1e-6):
    """
    Calculate standard deviation from variance.
    Args:
        variance (torch.Tensor): Variance tensor.
        epsilon (float): Small epsilon value for numerical stability.
    Returns:
        torch.Tensor: Standard deviation tensor.
    """
    return (variance + epsilon).sqrt()

class SpatialAttention(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=nn.BatchNorm2d):
        super(SpatialAttention, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        squared_x = x**2

        x1 = self.pool1(squared_x)
        x1 = calculate_std_from_variance(x1)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)



        x2 = self.pool2(squared_x)
        x2 = calculate_std_from_variance(x2)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)


        # Fusion = HxW
        x_out = self.conv3(x1 + x2)
        x_out = self.sigmoid(x_out)

        # identity
        out = x * x_out

        return out




class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,out_planes,kernel_size,stride=1,padding=0,dilation=1,groups=1,relu=True,
        bn=True,bias=False,):

        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,out_planes,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,
            groups=groups, bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale




class ATTN_Block(nn.Module):

    def __init__(self, in_channels):
        super(ATTN_Block, self).__init__()
        self.channel = ChannelAttention()
        self.spatial = SpatialAttention(in_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_x = self.channel(x)
        spatial_x = self.spatial(x)

        M_F = self.sigmoid(channel_x + spatial_x)

        F = x + (x * M_F)

        return F