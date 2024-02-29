# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 8:59 上午
# @File    : UCTransNet.py
# @Software: PyCharm
import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from .CTrans import ChannelTransformer
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import copy

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    #layers.append(BasicBlock(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
        #layers.append(BasicBlock(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

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

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)
class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)
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
        self.bn1 = nn.BatchNorm2d(in_channels//2)
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

#残差模块
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        #relu
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_channels)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(out_channels)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        #没进
        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x
#trans模块
#embedding+encoder
class Transformer(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels, embed_dim,vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, patchsize, img_size, in_channels, embed_dim)
        self.encoder = Encoder(config, vis,embed_dim)

    def forward(self, input_ids):
        embedding_output= self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, patchsize=4, img_size=56, in_channels=64,out_channels=384):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        # patch的数量
        n_patches = (56 // 4) * (56 // 4)

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, out_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        #(b,384,14,14)
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        #(b,196,384)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)d
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
#然后调用block
class Encoder(nn.Module):
        def __init__(self, config, vis,embed_dim):
            super(Encoder, self).__init__()
            self.vis = vis
            self.layer = nn.ModuleList()
            self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
            for _ in range(config.transformer["num_layers"]):
                layer = Block(config, vis,embed_dim)
                self.layer.append(copy.deepcopy(layer))

        def forward(self, hidden_states):
            attn_weights = []
            for layer_block in self.layer:
                hidden_states, weights = layer_block(hidden_states)
                if self.vis:
                    attn_weights.append(weights)
            encoded = self.encoder_norm(hidden_states)
            return encoded
class Block(nn.Module):
    def __init__(self, config, vis,embed_dim):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config,embed_dim,embed_dim*4)
        self.attn = Attention(config, vis,embed_dim)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

#包括attention和Mlp
class Attention(nn.Module):
    def __init__(self, config, vis, embed_dim):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        #self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.attention_head_size = int(embed_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        #config.hidden_size=64
        # self.query = Linear(config.hidden_size, self.all_head_size)
        # self.key = Linear(config.hidden_size, self.all_head_size)
        # self.value = Linear(config.hidden_size, self.all_head_size)
        self.query = Linear(embed_dim, self.all_head_size)
        self.key = Linear(embed_dim, self.all_head_size)
        self.value = Linear(embed_dim, self.all_head_size)

        # self.out = Linear(config.hidden_size, config.hidden_size)
        self.out = Linear(embed_dim, embed_dim)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
class Mlp(nn.Module):
    def __init__(self, config, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes,config,vis, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                  num_med_block=0, last_fusion=False, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(in_channels=inplanes, out_channels=outplanes, res_conv=res_conv, stride=stride, groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(in_channels=outplanes, out_channels=outplanes, stride=2, res_conv=True, groups=groups)
        else:
            self.fusion_block = ConvBlock(in_channels=outplanes, out_channels=outplanes, groups=groups)

        # if num_med_block > 0:
        #     self.med_block = []
        #     for i in range(num_med_block):
        #         self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
        #     self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(config,vis,embed_dim)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        #x_t(b,196,384)
        #x(b,256,56,56) x2(b,64,56,56)
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2) #(b,196,384) ,384是维度
        #融入以后形成新的trans输入，再进行下一个transformer
        x_t,weight = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)
        #x_t为trans的输出
        #x_t_r为trans转为conv
        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)#(b,64,56,56)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t
#卷积送入trans
class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x):
        #(b,384,56,56)
        x = self.conv_project(x)  # [N, C, H, W]
        #(b,196,384),类似嵌入
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        #x = torch.cat([x_t[:, 0][:, None, :], x], dim=1) #(b,1,384)

        return x
class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x.transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r))) #(b,64,14,14)

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride)) #(b,64,56,56)
class Netbase(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,img_size=224,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = 64

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]
        self.conv2 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 1 / 2 [112, 112]

        #conv c2
        stage_1_channel = 64
        self.conv_1 = ConvBlock(in_channels=3, out_channels=stage_1_channel, res_conv=True, stride=1)
        # drop_path_rate = 0.
        # self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # embed_dim = 768
        # patch_size = 16
        # trans_dw_stride = patch_size // 4
        # num_heads = 12
        # mlp_ratio = 4.
        # qkv_bias = False
        # qk_scale = None
        # drop_rate = 0.
        # attn_drop_rate = 0.
        # self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        # self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
        #                      )
        self.trans_1 = Transformer(config, patchsize=4, img_size=224, in_channels=64, embed_dim=384,vis=False)

        # 2~4 stage
        drop_path_rate = 0.
        depth = 12
        stage_2_channel = 128
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.conv_trans_2 = ConvTransBlock(
                                #stage_1_channel=64
                                stage_1_channel, stage_2_channel,config,vis, True, 2, dw_stride=8,
                                embed_dim=384,
                                num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                                drop_rate=0., attn_drop_rate=0.,
                                drop_path_rate=self.trans_dpr[1],
                                num_med_block=0
                            )
        self.conv_trans_3 = ConvTransBlock(
            # stage_1_channel=256
            stage_2_channel, stage_2_channel, config, vis, False, 1, dw_stride=8,
            embed_dim=384,
            num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
            drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=self.trans_dpr[2],
            num_med_block=0
        )
        self.conv_trans_4 = ConvTransBlock(
            # stage_1_channel=256
            stage_2_channel, stage_2_channel, config, vis, False, 1, dw_stride=8,
            embed_dim=384,
            num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
            drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=self.trans_dpr[3],
            num_med_block=0
        )

        # 5~8 stage
        stage_3_channel = 256
        self.conv_trans_5 = ConvTransBlock(
            stage_2_channel, stage_3_channel, config, vis, True, 2, dw_stride=4,
            embed_dim=384,
            num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
            drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=self.trans_dpr[4],
            num_med_block=0
        )
        self.conv_trans_6 = ConvTransBlock(
            stage_3_channel, stage_3_channel, config, vis, False, 1, dw_stride=4,
            embed_dim=384,
            num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
            drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=self.trans_dpr[5],
            num_med_block=0
        )
        self.conv_trans_7 = ConvTransBlock(
            stage_3_channel, stage_3_channel, config, vis, False, 1, dw_stride=4,
            embed_dim=384,
            num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
            drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=self.trans_dpr[6],
            num_med_block=0
        )
        self.conv_trans_8 = ConvTransBlock(
            stage_3_channel, stage_3_channel, config, vis, False, 1, dw_stride=4,
            embed_dim=384,
            num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
            drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=self.trans_dpr[7],
            num_med_block=0
        )
        # 9~12 stage
        stage_4_channel = 512
        self.conv_trans_9 = ConvTransBlock(
            stage_3_channel, stage_4_channel, config, vis, True, 2, dw_stride=4//2,
            embed_dim=384,
            num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
            drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=self.trans_dpr[8],
            num_med_block=0
        )
        self.conv_trans_10 = ConvTransBlock(
            stage_4_channel, stage_4_channel, config, vis, False, 1, dw_stride=4//2,
            embed_dim=384,
            num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
            drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=self.trans_dpr[9],
            num_med_block=0
        )
        self.conv_trans_11 = ConvTransBlock(
            stage_4_channel, stage_4_channel, config, vis, False, 1, dw_stride=4//2,
            embed_dim=384,
            num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
            drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=self.trans_dpr[10],
            num_med_block=0
        )
        stage_5_channel = 1024
        self.conv_trans_12 = ConvTransBlock(
            stage_4_channel, stage_5_channel, config, vis, True, 2, dw_stride=4//4,
            embed_dim=384,
            num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
            drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=self.trans_dpr[11],
            num_med_block=0
        )

        # #3,224,224
        # self.inc = ConvBatchNorm(n_channels, in_channels)
        # #64,224,224
        # self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        # #128,112,112
        # self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        # #256,56,56
        # self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        # #512,28,28
        # self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        # # 512,14,14
        # self.mtc = ChannelTransformer(config, vis, img_size,
        #                               #64,128,256,512
        #                              channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
        #                              patchSize=config.patch_sizes)
        # self.up4 = UpBlock_attention(in_channels*16, in_channels*4, nb_Conv=2)
        # self.up3 = UpBlock_attention(in_channels*8, in_channels*2, nb_Conv=2)
        # self.up2 = UpBlock_attention(in_channels*4, in_channels, nb_Conv=2)
        # self.up1 = UpBlock_attention(in_channels*2, in_channels, nb_Conv=2)
        # self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))
        # self.up4 = UpBlock(config.base_channel*16, config.base_channel*16, nb_Conv=2)
        # self.up3 = UpBlock(config.base_channel*16, config.base_channel*8, nb_Conv=2)
        # self.up2 = UpBlock(config.base_channel*8, config.base_channel*4, nb_Conv=2)
        # self.up1 = UpBlock(config.base_channel*4, config.base_channel, nb_Conv=2)
        # self.outc = nn.Conv2d(config.base_channel, n_classes, kernel_size=(1,1), stride=(1,1))
        # self.last_activation = nn.Sigmoid() # if using BCELoss
        self.last_activation = nn.Sigmoid()  # if using BCELoss

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
        self.h2_PT_hd4_conv = nn.Conv2d(in_channels * 2, self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(in_channels * 4, self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(in_channels * 8, self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(in_channels * 16, self.CatChannels, 3, padding=1)
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
        self.h2_PT_hd3_conv = nn.Conv2d(in_channels * 2, self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(in_channels * 4, self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(in_channels * 16, self.CatChannels, 3, padding=1)
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
        self.h2_Cat_hd2_conv = nn.Conv2d(in_channels * 2, self.CatChannels, 3, padding=1)
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
        self.hd5_UT_hd2_conv = nn.Conv2d(in_channels * 16, self.CatChannels, 3, padding=1)
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
        self.hd5_UT_hd1_conv = nn.Conv2d(in_channels * 16, self.CatChannels, 3, padding=1)
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
        self.outconv5 = nn.Conv2d(in_channels * 16, n_classes, 3, padding=1)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels * 16, 2, 1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid())
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=self.UpChannels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

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
        #(b,64,56,56)
        x_base_trans = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        # 1 stage
        # conv
        #(b,256,56,56)
        x = self.act1(self.bn1(self.conv2(x)))
        #trans
        #应该是embedding
        #(b,296,384),384是维度
        x_t = self.trans_1(x_base_trans)

        # 2 ~ final
        x_2, x_t_2 = self.conv_trans_2(x, x_t)
        x_3, x_t_3 = self.conv_trans_3(x_2, x_t_2)
        x_4, x_t_4 = self.conv_trans_4(x_3, x_t_3)
        x_5, x_t_5 = self.conv_trans_5(x_4, x_t_4)
        x_6, x_t_6 = self.conv_trans_6(x_5, x_t_5)
        x_7, x_t_7 = self.conv_trans_7(x_6, x_t_6)
        x_8, x_t_8 = self.conv_trans_8(x_7, x_t_7)
        x_9, x_t_9 = self.conv_trans_9(x_8, x_t_8)
        x_10, x_t_10 = self.conv_trans_10(x_9, x_t_9)
        x_11, x_t_11 = self.conv_trans_11(x_10, x_t_10)
        x_12, x_t_12 = self.conv_trans_12(x_11, x_t_11)
        h1 = x
        h2 = x_4
        h3 = x_8
        h4 = x_11
        hd5 = x_12

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

        # d1 = self.outconv1(hd1)  # 256
        #
        # #d1 = self.dotProduct(d1, cls_branch_max)
        # return F.sigmoid(d1)
        # #return F.tanh(d1)
        return self.final(hd1)




