import numpy
import torch.optim
from medpy.metric import hd95
from scipy.ndimage import _ni_support, generate_binary_structure, binary_erosion, distance_transform_edt
from sklearn.metrics._scorer import metric

from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

from nets.FilterTransU_5_s4 import FilterTransU_5_s4
from nets.UNet import UNet
from nets.uctransnet import UCTransNet
from nets.vision_transformer import SwinUnet

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from nets.UCTransNet3Plus import UCTransNet3Plus
from utils import *
import cv2
#from medpy import metric

# def hd95(result, reference, voxelspacing=None, connectivity=1):
#     hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
#     hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
#     hd95 = numpy.percentile(numpy.hstack((hd1, hd2)), 95)
#     return hd95


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = numpy.asarray(voxelspacing, dtype=numpy.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == numpy.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == numpy.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds

def show_image_with_dice(predict_save, labs, save_path):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # dice_show = "%.3f" % (dice_pred)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))

    # fig, ax = plt.subplots()
    # plt.gca().add_patch(patches.Rectangle(xy=(4, 4),width=120,height=20,color="white",linewidth=1))
    if config.task_name == "lung":
        predict_save = cv2.pyrUp(predict_save,(448,448))
        predict_save = cv2.resize(predict_save,(2000,2000))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path,predict_save * 255)
    else:
        cv2.imwrite(save_path,predict_save * 255)
    # plt.imshow(predict_save * 255,cmap='gray')
    # plt.text(x=10, y=24, s="Dice:" + str(dice_show), fontsize=5)
    # plt.axis("off")
    # remove the white borders
    # height, width = predict_save.shape
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.savefig(save_path, dpi=2000)
    # plt.close()
    return dice_pred, iou_pred

def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()

    output = model(input_img.cuda())
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')
    pre = np.reshape(predict_save, (1,config.img_size, config.img_size))
    #dice_one = metric.binary.dc(pre, labs)
    #sen_one = metric.binary.sensitivity(pre, labs)
    hd_error=0
    try:
        hd_one = hd95(pre, labs, voxelspacing=None, connectivity=1)
    except Exception:
        hd_error = hd_error+1
        hd_one = 0.00000001
        print("An exception happened in the {} mask!".format(vis_save_path))
    #hd_one = hd95(pre, labs, voxelspacing=None, connectivity=1)
    return dice_pred_tmp, iou_tmp,hd_one,hd_error



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session
    if config.task_name is "NSCLCnew":
        test_num = 512
        model_type = config.model_name
        model_path = "./NSCLCnew/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "bre":
        test_num = 744
        model_type = config.model_name
        model_path = "./bre/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
        #model_path = r'C:\code\UCTransNet-main\MSD\FilterTransU_5\Test_session_02.25_20h06\best_model-FilterTransU_5.pth.tar'


    save_path  = config.task_name +'/'+ model_type +'/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')


    if model_type == 'UNet':
        config_vit = config.get_CTranS_config()
        # model = Netbase_se(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)
        model = UNet()
        #model = FilterTransU_5_s4(in_chan=3, base_chan=32)
        #model = FilterTransU_5(in_chan=3, base_chan=32)
        #model = SwinUnet(config_vit, img_size=224, num_classes=1, zero_head=False, vis=False)  # VisionTransformer
        #model = MISSFormer()
    elif model_type == 'UCTransNet':
        config_vit = config.get_CTranS_config()
        model = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
        # model = UCTransNet3Plus(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)
        #model = UTNet(in_chan=3, base_chan=64)
        # model = VisionTransformer(config_vit, img_size=224, num_classes=1, zero_head=False, vis=False)
        #model = Build_LeViT_UNet_384(num_classes=1, pretrained=True)
        # model = UNet_Attention_Transformer_Multiscale(n_channels=3, n_classes=1, bilinear=True)


    else: raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    dice = 0.0
    sensitivity = 0.0
    iou_pred = 0.0
    hd_pred = 0.0
    dice_ens = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path+names[0]+"_lab.jpg", dpi=300)
            plt.close()
            input_img = torch.from_numpy(arr)
            dice_pred_t,iou_pred_t,hd_pred_t,hd_error = vis_and_save_heatmap(model, input_img, None, lab,
                                                          vis_path+names[0],
                                               dice_pred=dice_pred, dice_ens=dice_ens)
            # hd_t = metric.binary.hd95(input_img, lab)
            dice_pred+=dice_pred_t
            iou_pred+=iou_pred_t
            # dice+=dice_one
            # sensitivity+=sen_one
            hd_pred+= hd_pred_t
            torch.cuda.empty_cache()
            pbar.update()
            # if i == 520:
            #     print(dice_pred)
            #     print(iou_pred)

    print ("dice_pred",dice_pred/test_num)
    print ("iou_pred",iou_pred/test_num)
    # print('dice',dice/test_num)
    # print('sensitivity', sensitivity / test_num)
    print('hd',hd_pred)
    print('hd_error',hd_error)
    # print('hd', hd_pred / (test_num-hd_error))
