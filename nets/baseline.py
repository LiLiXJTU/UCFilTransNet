from functools import partial
from torch.nn.modules.utils import _pair
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np



class BaseLine(nn.Module):

    def __init__(self, in_chan, base_chan, num_classes=1, reduce_size=7, block_list='4', num_blocks=[1, 2, 4],
                 projection='interp', num_heads=[2, 4, 8], attn_drop=0., proj_drop=0., bottleneck=False, maxpool=True,
                 rel_pos=True, aux_loss=False):
        super().__init__()

        self.aux_loss = aux_loss
        self.inc = [BasicBlock(in_chan, base_chan)]
        self.inc.append(BasicBlock(base_chan, base_chan))
        self.inc = nn.Sequential(*self.inc)
        self.down1 = down_block(base_chan, 2 * base_chan, (2, 2), num_block=2)
        self.down2 = down_block(2 * base_chan, 4 * base_chan, (2, 2), num_block=2)
        self.down3 = down_block(4 * base_chan, 8 * base_chan, (2, 2), num_block=2)
        self.down4 = down_block(8 * base_chan, 16 * base_chan, (2, 2), num_block=2)

        # self.filterTrans2 = FilterTrans(2 * base_chan, base_chan,num_block=0,img_size=112)
        # self.filterTrans3 = FilterTrans(4 * base_chan, 2 * base_chan,num_block=0,img_size=56)
        # self.filterTrans4 = FilterTrans(8 * base_chan, 4 * base_chan,num_block=0,img_size=28)
        self.up4 = up_block(2 * base_chan, base_chan, scale=(2, 2), num_block=2)
        self.up3 = up_block(4 * base_chan, 2 * base_chan, scale=(2, 2), num_block=2)
        self.up2 = up_block(8 * base_chan, 4 * base_chan, scale=(2, 2), num_block=2)
        self.up1 = up_block(16 * base_chan, 8 * base_chan, scale=(2, 2), num_block=2)

        self.outc = nn.Conv2d(base_chan, num_classes, kernel_size=1, bias=True)
        self.sig = nn.Sigmoid()


    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #(256,28,28)
        # t4 = self.filterTrans4(x4,x3)
        # t3 = self.filterTrans3(x3,x2)
        # t2 = self.filterTrans2(x1,x2)
        x5 = self.down4(x4)

        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)
        out = self.sig(out)
        return out


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    self.relu,
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                    )

    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residue)

        return out
class BottleneckBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        # self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    self.relu,
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                    )
        self.PSAModule = PSAModule(planes,planes)

    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.PSAModule(out)

        out += self.shortcut(residue)

        return out
class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, scale, num_block, bottleneck=False, pool=True):
        super().__init__()

        block_list = []

        if bottleneck:
            block = 2
        else:
            #False
            block = BasicBlock


        if pool:
            #True
            #block_list.append(SimplifiedLIP(in_ch))
            block_list.append(nn.MaxPool2d(scale))
            block_list.append(block(in_ch, out_ch))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, stride=1))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x):
        return self.conv(x)

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, scale=(2,2),bottleneck=False):
        super().__init__()
        self.scale=scale

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock


        block_list = []
        block_list.append(block(2*out_ch, out_ch))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out

class down_block_4(nn.Module):
    def __init__(self, in_ch, out_ch, scale, num_block, bottleneck=False, pool=True):
        super().__init__()

        block_list = []

        if bottleneck:
            block = 2
        else:
            #False
            block1 = BasicBlock
            block2 = BottleneckBlock


        if pool:
            #True
            #block_list.append(SimplifiedLIP(in_ch))
            block_list.append(nn.MaxPool2d(scale))
            block_list.append(block1(in_ch, out_ch))
        else:
            block_list.append(block1(in_ch, out_ch, stride=2))

        for i in range(num_block-1):
            block_list.append(block2(out_ch, out_ch, stride=1))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x):
        return self.conv(x)
#嵌入2
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

#嵌入1
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, img_size, in_channels,patchsize=1, emd_dim = 384,embeddings_dropout_rate=0.1):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        # patch的数量
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        # patches_grid = (int(224 / 16), int(224 / 16))
        # grid_size = patches_grid
        # patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        # patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        # n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=emd_dim,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, emd_dim))
        self.dropout = nn.Dropout(embeddings_dropout_rate)

    def forward(self, x):
        if x is None:
            return None
        #(B,512,14,14)
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        #嵌入到384维度
        x = x.flatten(2)

        # B, C, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # h, w = int(np.sqrt(hidden)), int(np.sqrt(hidden))
        # x = x.contiguous().view(B, C, h, w)
        # x = self.con_embeddings(x)
        # x = x.flatten(2)

        #(b,512,9)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)d
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,groups=groups)


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias,
                                   stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out


class BasicResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                self.relu,
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residue)

        return out

########################################################################
# Transformer components
class Attention(nn.Module):
    def __init__(self, in_dim, out_dim,dim,h,w,heads=4,  attn_drop=0.,rel_pos=True):
        super(Attention, self).__init__()
        self.num_attention_heads = heads

        self.dim_head = int(dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.dim_head
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        # self.relative_position_encoding = RelativePositionBias(heads, 16, 16)
        # self.query = nn.Linear(dim, self.all_head_size)
        # self.key = nn.Linear(dim, self.all_head_size)
        # self.value = nn.Linear(dim, self.all_head_size)

        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(attn_drop)

        # self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        # self.position_embeddings = nn.Parameter(torch.zeros(1, heads, h*h,h*h))

    # def transpose_for_scores(self, x):
    #     new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    #     x = x.view(*new_x_shape)
    #     return x.permute(0, 2, 1, 3)

    def forward(self, q, x):
        B, N, C = x.shape  # low-dim feature shape
        BH, NH, CH = q.shape  # high-dim feature shape
        aH = bH = int(math.sqrt(NH))

        q = q.view(BH, aH, bH, CH)
        q = q.to(torch.float32)
        # 亮点就是噪声点
        q = torch.fft.rfft2(q, dim=(1, 2), norm='ortho')
        weight_q = torch.view_as_complex(self.complex_weight)
        q = q * weight_q
        q = torch.fft.irfft2(q, s=(aH, bH), dim=(1, 2), norm='ortho')

        a = b = int(math.sqrt(N))
        x = x.view(B, a, b, C)
        x = x.permute(0,3,1,2)
        x = F.interpolate(x,size=(aH,bH))
        x = x.permute(0,2,3,1)
        x = x.to(torch.float32)
        # 亮点就是噪声点
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight_x = torch.view_as_complex(self.complex_weight)
        x = x * weight_x
        x = torch.fft.irfft2(x, s=(aH, bH), dim=(1, 2), norm='ortho')

        mixed_query_layer = q
        mixed_key_layer = x
        mixed_value_layer = x
        mixed_query_layer = rearrange(mixed_query_layer, 'b h w (dim_head heads) -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.num_attention_heads,
                      h=aH, w=bH)
        mixed_key_layer, mixed_value_layer = map(lambda t: rearrange(t, 'b h w (dim_head heads) -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.num_attention_heads, h=aH, w=bH), (mixed_key_layer, mixed_value_layer))

        # query_layer = self.transpose_for_scores(mixed_query_layer)
        # key_layer = self.transpose_for_scores(mixed_key_layer)
        # value_layer = self.transpose_for_scores(mixed_value_layer)

        # attention_scores = torch.matmul(mixed_query_layer, mixed_key_layer.transpose(-1, -2))
        attention_scores = torch.einsum('bhid,bhjd->bhij', mixed_query_layer, mixed_key_layer)
        # attention_scores = attention_scores+self.position_embeddings

        # relative_position_bias = self.relative_position_encoding(aH, bH)
        # attention_scores += relative_position_bias

        # attention_scores = mixed_query_layer*mixed_key_layer
        attention_scores = attention_scores / math.sqrt(self.dim_head)
        # attention_probs = self.softmax(attention_scores)
        attention_probs = self.sigmoid(attention_scores)
        # weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.einsum('bhij,bhjd->bhid', attention_probs, mixed_value_layer)
        #context_layer = attention_probs*mixed_value_layer


        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = rearrange(context_layer, 'b heads (h w) dim_head -> b h w (dim_head heads)', dim_head=self.dim_head, heads=self.num_attention_heads,
                      h=aH, w=bH)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        attention_output = attention_output.reshape(BH, NH, CH)
        return attention_output
class RelativePositionEmbedding(nn.Module):
    # input-dependent relative position
    def __init__(self, dim, shape):
        super().__init__()

        self.dim = dim
        self.shape = shape

        self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dim)) * 0.02)
        self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dim)) * 0.02)

        coords = torch.arange(self.shape)
        relative_coords = coords[None, :] - coords[:, None]  # h, h
        relative_coords += self.shape - 1  # shift to start from 0

        self.register_buffer('relative_position_index', relative_coords)

    def forward(self, q, Nh, H, W, dim_head):
        # q: B, Nh, HW, dim
        B, _, _, dim = q.shape

        # q: B, Nh, H, W, dim_head
        q = rearrange(q, 'b heads (h w) dim_head -> b heads h w dim_head', b=B, dim_head=dim_head, heads=Nh, h=H, w=W)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, 'w')

        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.key_rel_h, 'h')

        return rel_logits_w, rel_logits_h

    def relative_logits_1d(self, q, rel_k, case):

        B, Nh, H, W, dim = q.shape

        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)  # B, Nh, H, W, 2*shape-1

        if W != self.shape:
            # self_relative_position_index origin shape: w, w
            # after repeat: W, w
            relative_index = torch.repeat_interleave(self.relative_position_index, W // self.shape, dim=0)  # W, shape
        relative_index = relative_index.view(1, 1, 1, W, self.shape)
        relative_index = relative_index.repeat(B, Nh, H, 1, 1)

        rel_logits = torch.gather(rel_logits, 4, relative_index)  # B, Nh, H, W, shape
        rel_logits = rel_logits.unsqueeze(3)
        rel_logits = rel_logits.repeat(1, 1, 1, self.shape, 1, 1)

        if case == 'w':
            rel_logits = rearrange(rel_logits, 'b heads H h W w -> b heads (H W) (h w)')

        elif case == 'h':
            rel_logits = rearrange(rel_logits, 'b heads W w H h -> b heads (H W) (h w)')

        return rel_logits
class RelativePositionBias(nn.Module):
    # input-independent relative position attention
    # As the number of parameters is smaller, so use 2D here
    # Borrowed some code from SwinTransformer: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    def __init__(self, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.h = h
        self.w = w

        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * h - 1) * (2 * w - 1), num_heads) * 0.02)

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
        coords_flatten = torch.flatten(coords, 1)  # 2, hw

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        relative_position_index = relative_coords.sum(-1)  # hw, hw

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h,
                                                                                                               self.w,
                                                                                                               self.h * self.w,
                                                                                                      -1)  # h, w, hw, nH

        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias, H // self.h, dim=0)
        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h, W // self.w,
                                                                  dim=1)  # HW, hw, nH

        relative_position_bias_expanded = relative_position_bias_expanded.view(H * W, self.h * self.w,
                                                                               self.num_heads).permute(2, 0,
                                                                                                       1).contiguous().unsqueeze(
            0)

        return relative_position_bias_expanded
###########################################################################
# Unet Transformer building block

class FilterTrans(nn.Module):
    def __init__(self, x1_ch,x2_ch, num_block, img_size,dim = 384,bottleneck=False, heads=4, dim_head=64, attn_drop=0., proj_drop=0.,
                 reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()
        self.embedding1 = Embeddings(img_size=img_size,in_channels=x1_ch)
        self.embedding2 = Embeddings(img_size=img_size*2,in_channels=x2_ch)
        self.ln = nn.LayerNorm(dim)
        self.relu = nn.ReLU(inplace=True)
        #self.mlp = SE_MLP(dim, img_size, img_size, 6)
        self.mlp = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.attn = Attention(x1_ch,x2_ch, dim, h=img_size,w=img_size//2+1,heads=heads, attn_drop=attn_drop,rel_pos=rel_pos)
        self.conv = nn.Conv2d(dim,x1_ch,kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x1, x2):
        # x1: low-dim feature, x2: high-dim feature
        # x2(b,256,28,28)
        # x1(b,128,56,56)
        x1 = self.embedding1(x1)

        # x1(b,56*56,dim)

        x2 = self.embedding2(x2)
        residue = x1
        # x2(b,28*28,dim)

        x1 = self.ln(x1)
        x2 = self.ln(x2)
        # x1 = self.filter1(x1)
        # # x1(b,56*56,dim)
        # x2 = self.filter2(x2)
        # x2(b,28*28,dim)

        out = self.attn(x1,x2)

        # out = self.attn_decoder(x1, x2)

        out = out + residue
        residue = out
        out = self.ln(out)
        out = self.relu(out)
        #out = self.mlp(out)
        out = self.mlp(out.permute(2, 1, 0))
        out = out.permute(2,1,0)
        out += residue

        B, n_patch, hidden = out.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        out = out.permute(0, 2, 1)
        out = out.contiguous().view(B, hidden, h, w)
        out = self.conv(out)

        return out


class GlobalFilter(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        #亮点就是噪声点
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
       #(b,14,8,768)
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        # (b,3,3,768)

        x = x.reshape(B, N, C)

        return x

class Mlp(nn.Module):
    def __init__(self, in_channel=384, mlp_channel=3072,dropout_rate=0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
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

class SE_MLP(nn.Module):
    def __init__(self, ch_in, h,w,reduction=6):
        super(SE_MLP, self).__init__()
        self.h = h
        self.w = w
        self.MAX_pool = nn.AdaptiveMaxPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, _, c = x.size()
        y=x.transpose(1, 2).reshape(b, c, self.h, self.w)
        y = self.MAX_pool(y).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1).transpose(1, 2)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上

class SimplifiedLIP(nn.Module):
    def __init__(self, channels):
        super(SimplifiedLIP, self).__init__()

        rp = channels

        self.logit = nn.Sequential(
            OrderedDict((
                ('conv', nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
                ('bn', nn.InstanceNorm2d(channels, affine=True)),
                ('gate', SoftGate()),
            ))
        )

    def init_layer(self):
        self.logit[0].weight.data.fill_(0.0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x))
        return frac

class SoftGate(nn.Module):
    def __init__(self):
        super(SoftGate, self).__init__()

    def forward(self, x):
        COEFF = 12.0
        return torch.sigmoid(x).mul(COEFF)

def lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return F.avg_pool2d(x*weight, kernel, stride, padding)/F.avg_pool2d(weight, kernel, stride, padding)

class PSAModule(nn.Module):
    def __init__(self,inplanes,planes,conv_kernels=[3,5,7,9],stride=1,conv_groups=[1,4,8,16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplanes,planes//4,kernel_size = conv_kernels[0],padding = conv_kernels[0]//2,
                              stride=stride,groups=conv_groups[0])
        self.conv_2 = conv(inplanes, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                              stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplanes, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                              stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplanes, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                              stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes//4)
        self.split_channel = planes//4
        self.softmax = nn.Softmax(1)
    def forward(self,x):
        b = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
        feats = torch.cat((x1,x2,x3,x4),dim=1)
        feats = feats.view(b,4,self.split_channel,feats.shape[2],feats.shape[3])
        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)
        x_se = torch.cat((x1_se,x2_se,x3_se,x4_se),dim=1)
        attention_vctors = x_se.view(b,4,self.split_channel,1,1)
        attention_vctors = self.softmax(attention_vctors)
        feats_weight = feats * attention_vctors
        for i in range(4):
            x_se_weight_fp = feats_weight[:,i,:,:]
            if i==0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp,out),1)
        return out

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight
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