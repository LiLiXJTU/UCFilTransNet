# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 2:44 下午
# @Author  : Haonan Wang
# @File    : Config.py
# @Software: PyCharm
import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 500
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True # whether use cosineLR or not
n_channels = 3
n_labels = 1
epochs = 200
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 50

pretrain = False
task_name = 'NSCLCnew' # MosMedData infnet_medseg hunhe MoNuSeg
# task_name = 'GlaS'
learning_rate = 2e-4
batch_size = 16
trans_lr = 0

#model_name = 'REUCTransNet'
model_name = 'UNet_LHQ'
#model_name = 'UCTransNet_pretrain'
#model_name = 'UCTransNet' #zuihouliangceng UCTransNet SAUnetPP UTNet TransAttUnet Build_LeViT_UNet_384 withoutTrans

train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
val_dataset = './datasets/'+ task_name+ '/Val_Folder/'  #Test_Folder  Val_Folder
test_dataset = './datasets/'+ task_name+ '/Test_Folder/'
# test_dataset = r'C:\code\UCTransNet-main\datasets\MSD3\Test_Folder\\'
# train_dataset = r'D:\Lqq\NSCLC_DATA\train\\'
# val_dataset = r'D:\Lqq\NSCLC_DATA\val\\'
#test_dataset = 'E:/ll/UCTransNet-main/datasets/infnet_medseg/Test_Folder/'
session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'

patches_grid = (int(224 / 16), int(224 / 16))
##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4  960 896

    #---------------transunet
    # config.transformer.num_heads = 12
    # config.transformer.num_layers = 12

    config.expand_ratio = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    #config.n_classes = 2



    config.patches_grid = (int(224 / 16), int(224 / 16))
    config.n_skip = 3
    #config.skip_channels = [256,128,64,16]
    config.skip_channels = [512, 256, 64, 16]

    # config.pretrained_path_transunet = './model/vit_checkpoint/imagenet21k-R50+ViT-B_16.npz'
    # config.pretrained_path_convit = './model/vit_checkpoint/Conformer_small_patch16.pth'
    # config.pretrained_path_vit = './model/vit_checkpoint/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'

    #预训练模型的
    #config.hidden_size = 768
    config.transformer.mlp_dim = 3072
    config.size = 56

    #其他
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.hidden_size = 384

    #---------------transunet
    # config.transformer.num_heads = 12
    # config.transformer.num_layers = 12
    # config.hidden_size = 768

    #swinunet
    config.pretrained_path_swinunet = './model/swin_tiny_patch4_window7_224.pth'
    config.swin_patchsize = 4
    config.IN_CHANS = 3
    config.swin_EMBED_DIM = 96
    config.SWIN_DEPTHS = [2, 2, 6, 2]
    config.SWIN_NUM_HEADS = [3, 6, 12, 24]
    config.SWIN_WINDOW_SIZE = 7
    config.SWIN_MLP_RATIO = 4.
    config.SWIN_QKV_BIAS = True
    config.SWIN_QK_SCALE = None
    config.DROP_RATE = 0.0
    config.DROP_PATH_RATE = 0.1
    config.SWIN_APE = False
    config.SWIN_PATCH_NORM = True
    config.TRAIN_USE_CHECKPOINT  = True

    return config




# used in testing phase, copy the session name in training phase
test_session = "Test_session_03.19_21h02"