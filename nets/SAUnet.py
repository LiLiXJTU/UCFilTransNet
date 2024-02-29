# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
#from layers import unetConv2, unetUp_origin
#from init_weights import init_weights
import numpy as np
from torchvision import models

from nets.init_weights import init_weights
from nets.layers import unetConv2, unetUp_origin


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=6):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上


# without bn version
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=128):
        super(ASPP, self).__init__()
        # self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        # self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 1, 1,padding=0,dilation=1),
            nn.ReLU(inplace=True)
        )
        #图片大小不变
        self.atrous_block2 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 3, 1,padding=2,dilation=2),
            nn.ReLU(inplace=True)
        )
        self.atrous_block3 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 3, 1,padding=4,dilation=4),
            nn.ReLU(inplace=True)
        )
        self.atrous_block4 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 3, 1,padding=6,dilation=6),
            nn.ReLU(inplace=True)
        )
        self.conv_1x1_output = nn.Sequential(
            nn.Conv2d(depth * 4, 512,1,1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )


    def forward(self, x):
        # size = x.shape[2:]
        #
        # image_features = self.mean(x)
        # image_features = self.conv(image_features)
        # image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block2 = self.atrous_block2(x)
        atrous_block3 = self.atrous_block3(x)
        atrous_block4 = self.atrous_block4(x)

        net = self.conv_1x1_output(torch.cat([atrous_block1, atrous_block2,
                                              atrous_block3, atrous_block4], dim=1))
        net = nn.LeakyReLU(net)
        return net.negative_slope


class SAUnetPP(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,img_size=224,vis=False):
        super(SAUnetPP, self).__init__()
        self.is_deconv = True
        self.in_channels = n_channels
        self.is_batchnorm = True
        self.is_ds = True
        self.feature_scale = 4

        filters = [32, 64, 128, 256, 512]
        #filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv00_conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, filters[0], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv00 = unetConv2(filters[0], filters[0], self.is_batchnorm)
        self.SE_Block00 = SE_Block(filters[0], reduction=6)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10_conv1 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv10 = unetConv2(filters[1], filters[1], self.is_batchnorm)
        self.SE_Block10 = SE_Block(filters[1], reduction=6)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv20_conv1 = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv20 = unetConv2(filters[2], filters[2], self.is_batchnorm)
        self.SE_Block20 = SE_Block(filters[2], reduction=6)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv30_conv1 = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv30 = unetConv2(filters[3], filters[3], self.is_batchnorm)
        self.SE_Block30 = SE_Block(filters[3], reduction=6)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv40_conv1 = nn.Sequential(
            nn.Conv2d(filters[3], filters[4], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv40 = unetConv2(filters[4], filters[4], self.is_batchnorm)
        self.SE_Block40 = SE_Block(filters[4], reduction=6)
        self.ASPP = ASPP(in_channel=512, depth=128)


        # upsampling
        self.up_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up01_conv1 = nn.Sequential(
            nn.Conv2d(filters[1], filters[0], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.unetConv2_01 = unetConv2(filters[0], filters[0], False)
        self.SE_Block01 = SE_Block(filters[0], reduction=6)

        self.up_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up11_conv1 = nn.Sequential(
            nn.Conv2d(filters[2], filters[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.unetConv2_11 = unetConv2(filters[1], filters[1], False)
        self.SE_Block11 = SE_Block(filters[1], reduction=6)

        self.up_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up21_conv1 = nn.Sequential(
            nn.Conv2d(filters[3], filters[2], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.unetConv2_21 = unetConv2(filters[2], filters[2], False)
        self.SE_Block21 = SE_Block(filters[2], reduction=6)

        self.up_concat31 = unetUp_origin(filters[4], filters[3], self.is_deconv)
        self.up31_conv1 = nn.Sequential(
            nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.unetConv2_31 = unetConv2(filters[3], filters[3], False)
        self.SE_Block31 = SE_Block(filters[3], reduction=6)



        self.up_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up02_conv1 = nn.Sequential(
            nn.Conv2d(96, filters[0], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.unetConv2_02 = unetConv2(filters[0], filters[0], False)
        self.SE_Block02 = SE_Block(filters[0], reduction=6)

        self.up_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up12_conv1 = nn.Sequential(
            nn.Conv2d(192, filters[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.unetConv2_12 = unetConv2(filters[1], filters[1], False)
        self.SE_Block12 = SE_Block(filters[1], reduction=6)

        self.up_concat22 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up22_conv1 = nn.Sequential(
            nn.Conv2d(384, filters[2], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.unetConv2_22 = unetConv2(filters[2], filters[2], False)
        self.SE_Block22 = SE_Block(filters[2], reduction=6)

        self.up_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up03_conv1 = nn.Sequential(
            nn.Conv2d(128, filters[0], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.unetConv2_03 = unetConv2(filters[0], filters[0], False)
        self.SE_Block03 = SE_Block(filters[0], reduction=6)

        self.up_concat13 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up13_conv1 = nn.Sequential(
            nn.Conv2d(256, filters[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.unetConv2_13 = unetConv2(filters[1], filters[1], False)
        self.SE_Block13 = SE_Block(filters[1], reduction=6)

        self.up_concat04 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up04_conv1 = nn.Sequential(
            nn.Conv2d(160, filters[0], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.unetConv2_04 = unetConv2(filters[0], filters[0], False)
        self.SE_Block04 = SE_Block(filters[0], reduction=6)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00_conv1 = self.conv00_conv1(inputs)
        X_00 = self.conv00(X_00_conv1)
        X_00 = self.SE_Block00(X_00)
        X_00 = X_00+X_00_conv1
        maxpool0 = self.maxpool0(X_00)

        X_10_conv1 = self.conv10_conv1(maxpool0)
        X_10 = self.conv10(X_10_conv1)
        X_10 = self.SE_Block10(X_10)
        X_10 = X_10 + X_10_conv1
        maxpool1 = self.maxpool1(X_10)

        X_20_conv1 = self.conv20_conv1(maxpool1)
        X_20 = self.conv20(X_20_conv1)
        X_20 = self.SE_Block20(X_20)
        X_20 = X_20 + X_20_conv1
        maxpool2 = self.maxpool2(X_20)

        X_30_conv1 = self.conv30_conv1(maxpool2)
        X_30 = self.conv30(X_30_conv1)
        X_30 = self.SE_Block30(X_30)
        X_30 = X_30 + X_30_conv1
        maxpool3 = self.maxpool3(X_30)

        X_40_conv1 = self.conv40_conv1(maxpool3)
        X_40 = self.conv40(X_40_conv1)
        X_40 = self.SE_Block40(X_40)
        X_40 = X_40 + X_40_conv1
        X_40 = self.ASPP(X_40)

        # column : 1
        X_01up = self.up_concat01(X_10, X_00)
        X_01_conv1 = self.up01_conv1(X_01up)
        X_01 = self.unetConv2_01(X_01_conv1)
        X_01 = self.SE_Block01(X_01)
        X_01 = X_01 + X_01_conv1

        X_11up = self.up_concat11(X_20, X_10)
        X_11_conv1 = self.up11_conv1(X_11up)
        X_11 = self.unetConv2_11(X_11_conv1)
        X_11 = self.SE_Block11(X_11)
        X_11 = X_11 + X_11_conv1

        X_21up = self.up_concat21(X_30, X_20)
        X_21_conv1 = self.up21_conv1(X_21up)
        X_21 = self.unetConv2_21(X_21_conv1)
        X_21 = self.SE_Block21(X_21)
        X_21 = X_21 + X_21_conv1

        X_31up = self.up_concat31(X_40, X_30)
        X_31_conv1 = self.up31_conv1(X_31up)
        X_31 = self.unetConv2_31(X_31_conv1)
        X_31 = self.SE_Block31(X_31)
        X_31 = X_31 + X_31_conv1
        # column : 2
        X_02up = self.up_concat02(X_11, X_00, X_01)
        X_02_conv1 = self.up02_conv1(X_02up)
        X_02 = self.unetConv2_02(X_02_conv1)
        X_02 = self.SE_Block02(X_02)
        X_02 = X_02 + X_02_conv1

        X_12up = self.up_concat12(X_21, X_10, X_11)
        X_12_conv1 = self.up12_conv1(X_12up)
        X_12 = self.unetConv2_12(X_12_conv1)
        X_12 = self.SE_Block12(X_12)
        X_12 = X_12 + X_12_conv1

        X_22up = self.up_concat22(X_31, X_20, X_21)
        X_22_conv1 = self.up22_conv1(X_22up)
        X_22 = self.unetConv2_22(X_22_conv1)
        X_22 = self.SE_Block22(X_22)
        X_22 = X_22 + X_22_conv1
        # column : 3
        X_03up = self.up_concat03(X_12, X_00, X_01, X_02)
        X_03_conv1 = self.up03_conv1(X_03up)
        X_03 = self.unetConv2_03(X_03_conv1)
        X_03 = self.SE_Block03(X_03)
        X_03 = X_03 + X_03_conv1


        X_13up = self.up_concat13(X_22, X_10, X_11, X_12)
        X_13_conv1 = self.up13_conv1(X_13up)
        X_13 = self.unetConv2_13(X_13_conv1)
        X_13 = self.SE_Block13(X_13)
        X_13 = X_13 + X_13_conv1

        # column : 4
        X_04up = self.up_concat04(X_13, X_00, X_01, X_02, X_03)
        X_04_conv1 = self.up04_conv1(X_04up)
        X_04 = self.unetConv2_04(X_04_conv1)

        X_04 = self.SE_Block04(X_04)
        X_04 = X_04 + X_04_conv1


        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.is_ds:
            return F.sigmoid(final)
        else:
            return F.sigmoid(final_4)
