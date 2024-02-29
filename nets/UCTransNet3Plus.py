# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 8:59 上午
# @File    : UCTransNet3Plus.py
# @Software: PyCharm
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from nets.CTrans import ChannelTransformer

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    #layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    layers.append(BasicBlock(in_channels, out_channels, activation))
    #layers.append(eespr(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(BasicBlock(out_channels, out_channels, activation))
        #layers.append(eespr(out_channels, out_channels, activation))
    return nn.Sequential(*layers)



class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)

# class CCA(nn.Module):
#     """
#     CCA Block
#     """
#     def __init__(self, F_g, F_x):
#         super().__init__()
#         self.mlp_x = nn.Sequential(
#             Flatten(),
#             nn.Linear(F_x, F_x))
#         self.mlp_g = nn.Sequential(
#             Flatten(),
#             nn.Linear(F_g, F_x))
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, g, x):
#         # channel-wise attention
#         avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#         channel_att_x = self.mlp_x(avg_pool_x)
#         avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
#         channel_att_g = self.mlp_g(avg_pool_g)
#         channel_att_sum = (channel_att_x + channel_att_g)/2.0
#         scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
#         x_after_channel = x * scale
#         out = self.relu(x_after_channel)
#         return out

# class UpBlock_attention(nn.Module):
#     def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2)
#         self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
#         self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
#
#     def forward(self, x, skip_x):
#         up = self.up(x)
#         skip_x_att = self.coatt(g=up, x=skip_x)
#         x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
#         return self.nConvs(x)

def Conv3(in_channels, out_channels, **kwargs):
    "3x3 convolution with padding"
    padding = kwargs.get('padding', 1)
    bias = kwargs.get('bias', False)
    stride = kwargs.get('stride', 1)
    kernel_size = kwargs.get('kernel_size', 3)
    out = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    return out

def Conv1(in_channels, out_channels, **kwargs):
    "1x1 convolution with padding"
    padding = kwargs.get('padding', 0)
    bias = kwargs.get('bias', False)
    stride = kwargs.get('stride', 1)
    kernel_size = kwargs.get('kernel_size', 1)
    out = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv1(in_channels, in_channels//2)
        #self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.bn1 = nn.GroupNorm(num_groups=2,num_channels=in_channels//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3(in_channels//2, in_channels//2)
        self.gn2 = nn.GroupNorm(num_groups=2,num_channels=in_channels//2)
        self.conv3 = Conv1(in_channels//2,in_channels)
        self.gn3 = nn.GroupNorm(num_groups=2,num_channels=in_channels)
        self.activation = get_activation(activation)
        #self.downsample = downsample
        #self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=2,num_channels=out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.gn3(out)

        out += residual
        out = self.activation(out)
        out = self.conv(out)
        out = self.norm(out)
        out = self.activation(out)
        return out

class UCTransNet3Plus(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,img_size=224,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        #3,224,224
        self.inc = ConvBatchNorm(n_channels, in_channels)
        #64,224,224
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        #128,112,112
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        #256,56,56
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        #512,28,28
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        # 512,14,14
        self.conv5 = nn.Conv2d(in_channels*8, in_channels*16,
                              kernel_size=3, padding=1)
        self.convDown1 = nn.Conv2d(in_channels,in_channels//2,kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=2, num_channels=in_channels // 2)
        self.convDown2 = nn.Conv2d(in_channels * 2,in_channels,kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=2, num_channels=in_channels)
        self.convDown3 = nn.Conv2d(in_channels * 4, in_channels * 2,kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(num_groups=2, num_channels=in_channels * 2)
        self.convDown4 = nn.Conv2d(in_channels * 8, in_channels * 4,kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(num_groups=2, num_channels=in_channels * 4)
        # 1024,14,14
        self.mtc = ChannelTransformer(config, vis, img_size,
                                      #64,128,256,512
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=config.patch_sizes)
        # self.up4 = UpBlock_attention(in_channels*16, in_channels*4, nb_Conv=2)
        # self.up3 = UpBlock_attention(in_channels*8, in_channels*2, nb_Conv=2)
        # self.up2 = UpBlock_attention(in_channels*4, in_channels, nb_Conv=2)
        # self.up1 = UpBlock_attention(in_channels*2, in_channels, nb_Conv=2)
        # self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))
        self.last_activation = nn.Sigmoid() # if using BCELoss

        ## -------------Decoder--------------
        self.CatChannels = in_channels
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(in_channels, self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(in_channels*2, self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(in_channels*4, self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(in_channels*8, self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(in_channels*16, self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(in_channels, self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(in_channels*2, self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(in_channels*4, self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(in_channels*16, self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(in_channels, self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(in_channels*2, self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(in_channels*16, self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(in_channels, self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(in_channels*16, self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(in_channels*16, n_classes, 3, padding=1)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels*16, 2, 1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid())

        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')

    def dotProduct(self, seg, cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, x):
        #编码器
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        hd5 = self.conv5(x5) #维度变成1024
        h1,h2,h3,h4,att_weights = self.mtc(x1,x2,x3,x4)
        # h1 = x1+h1
        # h2 = x2+h2
        # h3 = x3+h3
        # h4 = x4+h4

        # x1 = self.convDown1(x1)
        # x1 = self.gn1(x1)
        # x2 = self.convDown2(x2)
        # x2 = self.gn2(x2)
        # x3 = self.convDown3(x3)
        # x3 = self.gn3(x3)
        # x4 = self.convDown4(x4)
        # x4 = self.gn4(x4)
        # h1 = self.convDown1(h1)
        # h1 = self.gn1(h1)
        # h2 = self.convDown2(h2)
        # h2 = self.gn2(h2)
        # h3 = self.convDown3(h3)
        # h3 = self.gn3(h3)
        # h4 = self.convDown4(h4)
        # h4 = self.gn4(h4)
        # h1 = torch.cat((h1, x1), 1)
        # h2 = torch.cat((h2, x2), 1)
        # h3 = torch.cat((h3, x3), 1)
        # h4 = torch.cat((h4, x4), 1)
        # -------------Classification-------------
        cls_branch = self.cls(hd5).squeeze(3).squeeze(2)  # (B,N,1,1)->(B,N)
        # 在一张图片里做处理
        cls_branch_max = cls_branch.argmax(dim=1)  # 一维[1,1]
        cls_branch_max = cls_branch_max[:, np.newaxis].float()  # 变成二维

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)  # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)  # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(hd1)  # 256

        d1 = self.dotProduct(d1, cls_branch_max)
        # d2 = self.dotProduct(d2, cls_branch_max)
        # d3 = self.dotProduct(d3, cls_branch_max)
        # d4 = self.dotProduct(d4, cls_branch_max)
        # d5 = self.dotProduct(d5, cls_branch_max)
        return F.sigmoid(d1)
        # x = self.up4(x5, x4)
        # x = self.up3(x, x3)
        # x = self.up2(x, x2)
        # x = self.up1(x, x1)
        # if self.n_classes ==1:
        #     logits = self.last_activation(self.outc(x))
        # else:
        #     logits = self.outc(x) # if nusing BCEWithLogitsLoss or class>1
        # if self.vis: # visualize the attention maps
        #     return logits, att_weights
        # else:
        #     return logits


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class eespr(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super().__init__()
        #组卷积
        self.conv_group = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, groups=4),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        # 1x1conv branch
        #深度空洞可分离卷积
        self.b1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//4, kernel_size=1,dilation=1,groups=out_channels//4),
            nn.BatchNorm2d(out_channels//4),
            nn.PReLU()
        )

        # 1x1conv -> 3x3conv branch,rate=2,实际5
        self.b2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//4, kernel_size=1,dilation=1,groups=out_channels//4),
            nn.BatchNorm2d(out_channels//4),
            nn.PReLU(),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=2,dilation=2,groups=out_channels//4),
            nn.BatchNorm2d(out_channels//4),
            nn.PReLU(),
        )

        # 1x1conv -> 3x3conv branch,rate=4，实际9
        # we use 2 3x3 conv filters stacked instead
        # of 1 5x5 filters to obtain the same receptive
        # field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//4, kernel_size=1,dilation=1,groups=out_channels//4),
            nn.BatchNorm2d(out_channels//4),
            nn.PReLU(),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=4,dilation=4,groups=out_channels//4),
            nn.BatchNorm2d(out_channels//4),
            nn.PReLU(),
        )

        # 3x3pooling -> 1x1conv
        # same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(out_channels, out_channels//4, kernel_size=1,groups=out_channels//4),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU()
        )
        self.gconv = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels,
                              kernel_size=1, groups=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(out_channels, out_channels//8)
        self.fc2 = nn.Linear(out_channels//8, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def dot(self, input, c_weight):
        B, N, H, W = input.size()
        input = input.view(B, N, H * W)
        c_weight = torch.unsqueeze(c_weight, 2)
        #final = torch.einsum("ijk,ikz->ijz", [c_weight,input])
        final = torch.mul(input,c_weight)
        final = final.view(B, N, H, W)
        return final

    def forward(self, input):
        gc = self.conv_group(input)
        x = gc
        b1 = self.b1(gc)
        b2 = self.b2(gc)
        b3 = self.b3(gc)
        b4 = self.b4(gc)
        # c1 = b1 + b2
        # #c1 = self.relu(c1)
        # c2 = c1 + b3
        # #c2 = self.relu(c2)
        # c3 = c2 + b4
        #c3 = self.relu(c3)
        d = torch.cat([b1, b2, b3, b4], dim=1)
        # d = self.gconv(d)
        # d = self.bn(d)
        # d = d+x
        # d = self.relu(d)
        global_pooling = self.global_pooling(d)
        global_pooling = global_pooling.view(global_pooling.size()[0], -1)
        fc1 = self.fc1(global_pooling)
        fc2 = self.fc2(fc1)
        #对应元素相乘
        out_dot = self.dot(x,fc2)
        #out_add =x+out_dot
        out_concat = torch.cat([d,out_dot],dim=1)
        out_concat = self.gconv(out_concat)
        return out_concat