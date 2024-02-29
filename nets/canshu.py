
# from UNet3 import UNet_3Plus, UNet_3Plus_DeepSup_CGM
import Config as config
# from nets.convtransformer import Netbase
# from nets.kongdong import Netbase_se_asppp
# from nets.split import Split
from nets.utnet import UTNet
from nets.vit_seg_modeling import VisionTransformer
config_vit = config.get_CTranS_config()
# model_1 = UCTransNet3Plus(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# model_2 = Netbase_se(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# model_3 = TransVpt(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# model_4 = UNet(n_channels=3, n_classes=1)
# model_5 = UNet_3Plus(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
# model_6 = Netbase(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# model_7 = Netbase_se_asppp(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# model_8 = Netbase_zuihou(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# FTransConv = FTransConv(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# #model_9 = Split(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# SAUnetPP = SAUnetPP(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# UCTransNet = UCTransNet(config_vit,n_channels=3, n_classes=1,img_size=224,vis=False)
#unet = UNet()
#unetgit = UNet(3,1)
model =VisionTransformer(config_vit)
# withoutTrans = withoutTrans(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# withoutDTrans = withoutDTrans(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# abposition = abposition(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# relposition = relposition(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
# model = UTNet(in_chan=3, base_chan=32)
# withoutTrans = sum([param.nelement() for param in withoutTrans.parameters()])
#transunet = sum([param.nelement() for param in model_4.parameters()])
#unet = sum([param.nelement() for param in unet.parameters()])
parameter = sum([param.nelement() for param in model.parameters()])
#print("Number of parameters: %.2fM" % (total/1e6))
# print("abposition Number of parameters: %.2fM" % (abposition/1e6))
print("model Number of parameters: %.2fM" % (parameter/1e6))