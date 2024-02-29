# -- coding: utf-8 --
import torch
import torchvision
from thop import profile
import Config as config
# Model
from nets.LeViTUnet import Build_LeViT_UNet_384
from nets.TransAttUnet import UNet_Attention_Transformer_Multiscale
from nets.UNet import UNet
from nets.UNet_LHQ import UNet_LHQ
from nets.baseline import BaseLine
from nets.medformer import MedFormer
from nets.uctransnet import UCTransNet
from nets.utnet import UTNet
from nets.vision_transformer import SwinUnet
from nets.vit_seg_modeling import VisionTransformer

config_vit = config.get_CTranS_config()
# -- coding: utf-8 --
import torchvision
from ptflops import get_model_complexity_info

# model = TransVpt(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# model= MedFormer(in_chan=3, num_classes=1)
# model = Build_LeViT_UNet_384(num_classes=1, pretrained=True)
# model = VisionTransformer(config_vit, img_size=224, num_classes=1, zero_head=False, vis=False)
# model = BaseLine(in_chan=3, base_chan=32)
model = UNet_LHQ()
# model = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# model = SwinUnet(config_vit, img_size=224, num_classes=1, zero_head=False, vis=False)
# model = UNet_Attention_Transformer_Multiscale(n_channels=3, n_classes=1, bilinear=True)
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print('flops: ', flops, 'params: ', params)



