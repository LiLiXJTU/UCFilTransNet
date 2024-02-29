from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import math
from torch.nn.modules.utils import _pair
import numpy as np
from einops import rearrange
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,groups=groups)
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
class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        residue = x
        out = self.conv(x)
        out = out + self.shortcut(residue)
        out = nn.ReLU(inplace=True)(out)
        return out
class BottleneckBlock(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(BottleneckBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.psa = PSAModule(out_ch, out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        residue = x
        out = self.conv(x)
        out = self.relu(self.bn(self.psa(out)))
        out = out + self.shortcut(residue)
        out = nn.ReLU(inplace=True)(out)
        return out
class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Upsample(scale_factor=2)#默认上采样使用的是最邻近算法
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        x = self.up(x)
        residue = x
        out = self.conv(x)
        out = out + self.shortcut(residue)
        out = nn.ReLU(inplace=True)(out)
        return out
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
        B, N, C = x.shape  # high-dim feature shape
        BH, NH, CH = q.shape  # low-dim feature shape

        a = b = int(math.sqrt(N))
        x = x.view(B, a, b, CH)
        x = x.to(torch.float32)
        # 亮点就是噪声点
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight_x = torch.view_as_complex(self.complex_weight)
        x = x * weight_x
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        aH = bH = int(math.sqrt(NH))
        q = q.view(BH, aH, bH, CH)
        q = q.permute(0,3,1,2)
        q = F.interpolate(q,size=(a,b))
        q = q.permute(0,2,3,1)
        q = q.to(torch.float32)
        # 亮点就是噪声点
        q = torch.fft.rfft2(q, dim=(1, 2), norm='ortho')
        weight_q = torch.view_as_complex(self.complex_weight)
        q = q * weight_q
        q = torch.fft.irfft2(q, s=(a, b), dim=(1, 2), norm='ortho')

        mixed_query_layer = q
        mixed_key_layer = x
        mixed_value_layer = x
        mixed_query_layer = rearrange(mixed_query_layer, 'b h w (dim_head heads) -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.num_attention_heads,
                      h=a, w=b)
        mixed_key_layer, mixed_value_layer = map(lambda t: rearrange(t, 'b h w (dim_head heads) -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.num_attention_heads, h=a, w=b), (mixed_key_layer, mixed_value_layer))

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
                      h=a, w=b)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        attention_output = attention_output.reshape(B, N, C)
        return attention_output
class FilterTrans(nn.Module):
    def __init__(self, x1_ch,x2_ch, num_block, img_size,dim = 384,bottleneck=False, heads=4, dim_head=64, attn_drop=0., proj_drop=0.,
                 reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()
        self.embedding2 = Embeddings(img_size=img_size,in_channels=x1_ch)
        self.embedding1 = Embeddings(img_size=img_size*2,in_channels=x2_ch)
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
        residue = x2
        # x2(b,28*28,dim)

        x1 = self.ln(x1)
        x2 = self.ln(x2)
        out = self.attn(x1,x2)
        out = out + residue
        residue = out
        out = self.ln(out)
        out = self.relu(out)
        out = self.mlp(out.permute(2, 1, 0))
        out = out.permute(2,1,0)
        out += residue

        B, n_patch, hidden = out.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        out = out.permute(0, 2, 1)
        out = out.contiguous().view(B, hidden, h, w)
        out = self.conv(out)

        return out

class FilterTransU_8(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(FilterTransU_8, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = BottleneckBlock(filters[3], filters[4])

        self.filterTrans3 = FilterTrans(4 * n1, 2 * n1,num_block=0,img_size=56)
        self.filterTrans4 = FilterTrans(8 * n1, 4 * n1,num_block=0,img_size=28)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = nn.Sigmoid()

    def forward(self, x):
        x1 = self.Conv1(x)

        e2 = self.Maxpool1(x1)
        x2 = self.Conv2(e2)

        e3 = self.Maxpool2(x2)
        x3 = self.Conv3(e3)

        e4 = self.Maxpool3(x3)
        x4 = self.Conv4(e4)

        e5 = self.Maxpool4(x4)
        x5 = self.Conv5(e5)

        t4 = self.filterTrans4(x3,x4)
        t3 = self.filterTrans3(x2,x3)

        d5 = self.Up5(x5)
        d5 = torch.cat((t4, d5), dim=1)
        # 拼接之前的裁剪？
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((t3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        # conv9通道数是2，不是说二分类，你可以理解为普通的提特征卷积，只不过通道数是2，按道理这个实现是有问题的，
        # 这个conv9通道数是32或16比较合理。
        # 最后conv10将feature转化为通道数为1的logits。最后通道数必为1，然后再sigmoid(二分类)
        # sigmoid将logits归一化到0到1，也即是prob，用于bce loss的计算。
        # binary是二分类不是二进制。
        # 如果是多分类，通常用softmax而不是sigmoid激活，
        # logits的通道数也由1改为类数+1(background)。最后计算cross entropy loss

        # 为什么二分类logits只要一个通道并可以用sigmoid激活呢？
        # 因为假如像素为前景概率为prob，则该像素为背景的概率一定为1-prob，这样就可以计算ce loss，二分类用sigmoid和softmax都可
        # 如果多分类用sigmoid激活无法保证各类prob相加一定为1，
        # 而softmax则是约束了各类prob相加一定为1。根据约束条件的不同就可以延伸出各种loss了

        d1 = self.active(out)

        return d1



