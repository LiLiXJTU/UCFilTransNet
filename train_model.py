# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 8:59 上午
# @Author  : Haonan Wang
# @File    : train.py
# @Software: PyCharm
import torch.optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
from torch.optim.lr_scheduler import MultiStepLR

from nets.FilterTransU_5 import FilterTransU_5
from nets.FilterTransU_5_s3 import FilterTransU_5_s3
from nets.FilterTransU_5_s4 import FilterTransU_5_s4
from nets.FilterTransU_6 import FilterTransU_6
from nets.FilterTransU_8 import FilterTransU_8
from nets.TransAttUnet import UNet_Attention_Transformer_Multiscale
from nets.LeViTUnet import Build_LeViT_UNet_384

from nets.UNet import UNet

from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D

from nets.UCTransNet3Plus import UCTransNet3Plus
from nets.REUCTransNet import REUCTransNet
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch
import Config as config
from torchvision import transforms

from nets.UNet_LHQ import UNet_LHQ
from nets.baseline import BaseLine
from nets.medformer import MedFormer
from nets.uctransnet import UCTransNet
from nets.vision_transformer import SwinUnet
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE
from nets.vit_seg_modeling import VisionTransformer
cudnn.benchmark = True
def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    train_dataset = ImageToImage2D(config.train_dataset, train_tf, image_size=config.img_size)
    val_dataset = ImageToImage2D(config.val_dataset, val_tf, image_size=config.img_size)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=0,
                            pin_memory=True)

    lr = config.learning_rate
    logger.info(model_type)
  # ---------------------------------------------------------------------------------------------
    if model_type == 'UNet_LHQ': #UCTransNet REUCTransNet
        config_vit = config.get_CTranS_config()

        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        # model = abposition(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

       # model = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

        #model = SAUnetPP(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
        #model = VisionTransformer(config_vit,img_size=224, num_classes=21843, zero_head=False, vis=False)
        model = UNet_LHQ()
        #model = UCTransNet3Plus(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
        #model = resnet50(3, 1)
        #model = FilterTransU_5_s4(in_chan=3, base_chan=32)
        # model = SwinUnet(config_vit, img_size=224, num_classes=1, zero_head=False, vis=False)
        #model = UNet_Attention_Transformer_Multiscale(n_channels=3, n_classes=1, bilinear=True)
        # model = UNet(n_channels=3, n_classes=1)
        # model = TransVpt(config_vit, n_channels=config.n_channels, n_classes=config.n_labels,img_size=224, patch_size=16, Prompt_Token_num=10, VPT_type="Shallow")
       # model.load_from(weights=torch.load(config_vit.pretrained_path_convit))
        # prompt_state_dict = model.obtain_prompt()
        # model.load_prompt(prompt_state_dict)

        # model = SwinUnet(config_vit, img_size=224, num_classes=1, zero_head=False, vis=False)#VisionTransformer
        # model.load_from(config_vit)
        #model= MedFormer(in_chan=3, num_classes=1)

        #model = Build_LeViT_UNet_384(num_classes=1, pretrained=True)

        #transunet
        # model = VisionTransformer(config_vit, img_size=224, num_classes=1, zero_head=False, vis=False)
        # model.load_from(weights=np.load(config_vit.pretrained_path_transunet))
        #model.Freeze()

    elif model_type == 'withoutDTrans':
        # config_vit = config.get_CTranS_config()
        # logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        # logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        # logger.info('transformer expand+ ratio: {}'.format(config_vit.expand_ratio))
        # model = Netbase_zuihou(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
        # pretrained_UNet_model_path = "./hunhe/zuihouliangceng12/Test_session_07.22_16h59/models/best_model-zuihouliangceng12.pth.tar"
        # pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda')
        # pretrained_UNet = pretrained_UNet['state_dict']
        # model2_dict = model.state_dict()
        # state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
        # print(state_dict.keys())
        # model2_dict.update(state_dict)
        # model.load_state_dict(model2_dict,strict=False)
        # logger.info('Load successful!')
        # model.Freeze()
        # prompt_state_dict = model.obtain_prompt()
        # model.load_prompt(prompt_state_dict)
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand+ ratio: {}'.format(config_vit.expand_ratio))
        model = Build_LeViT_UNet_384(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
        pretrained_UNet_model_path = "./hunhe/zuihouliangceng12/Test_session_07.22_16h59/models/best_model-zuihouliangceng12.pth.tar"
        pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda')
        pretrained_UNet = pretrained_UNet['state_dict']
        model2_dict = model.state_dict()
        state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
        print(state_dict.keys())
        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict,strict=False)
        logger.info('Load successful!')

    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    # for name, p in model.named_parameters():
    #     if name.startswith('mtc'):
    #         p.requires_grad = False
    #     if name.startswith('inc'):
    #         p.requires_grad = False
    # if torch.cuda.device_count() > 1:
    #     print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
    # model = nn.DataParallel(model, device_ids=[0])
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=5e-4)
    # optimizer = torch.optim.Adam([
    #     {"params":model.mtc.parameters(),"lr":config.trans_lr},
    #     {"params": model.inc.parameters(), "lr": config.trans_lr}
    # ],lr=lr)
    # if config.cosineLR is True:
    #     lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-9)
    #     #lr_scheduler = MultiStepLR(optimizer,milestones=[20,50,100,120,140,200],gamma=0.5)
    # else:
    lr_scheduler = None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
        # img, _ = train_dataset.__getitem__(0)
        # img = img['image'].cuda()
        # img = img.unsqueeze(0)
        # writer.add_graph(model, img)
        # writer.close()
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)
        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                                 optimizer, writer, epoch, lr_scheduler, model_type, logger)

        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch + 1 > 1:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':
    deterministic = True



    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    model = main_loop(model_type=config.model_name, tensorboard=True)
