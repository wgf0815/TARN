import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Conv2d_(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(Conv2d_, self).__init__()
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.weight=nn.Parameter(torch.ones(out_channels,in_channels,kernel_size,kernel_size))
        torch.nn.init.kaiming_normal_(self.weight)
        self.bias=nn.Parameter(torch.zeros(out_channels))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
        out0 = F.conv2d(x,self.weight,self.bias,stride=self.stride,padding=self.padding)
        out1 = F.conv2d(x,self.weight,self.bias,stride=self.stride,padding=self.padding+1,dilation=2)
        out = torch.where(out0>out1,out0,out1)
        return out

class Conv3x3Block(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, is_pool=False, is_dfe=False):
        super(Conv3x3Block, self).__init__()
        self.block = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, stride=1)]
        self.block += [nn.BatchNorm2d(out_channels, momentum=0.1, affine=True),
                       nn.ReLU(inplace=True)]
 
        if is_pool:
            self.block.append(nn.MaxPool2d(2))           
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        out = self.block(x)
        return out

class Conv4(nn.Module):
    def __init__(self, feature_dim=64, is_fm=False, is_dfe=False):
        super(Conv4, self).__init__()
        self.layer1 = Conv3x3Block(3, feature_dim)
        self.layer2 = Conv3x3Block(feature_dim, feature_dim, is_pool=is_fm)
        self.layer3 = Conv3x3Block(feature_dim, feature_dim, padding=1, is_pool=True, is_dfe=is_dfe)
        self.layer4 = Conv3x3Block(feature_dim, feature_dim, padding=1, is_pool=True, is_dfe=is_dfe)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out 
