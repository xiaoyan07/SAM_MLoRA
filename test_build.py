import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import os
import random
import numpy as np
import time
import argparse

from networks.sam_adapter import build_sam_vit_b_adapter_linknet
from networks.sam_adapter import resize_model_pos_embed
from networks.sam_multi_lora import build_sam_vit_b_adapter_linknet_multi_lora
from networks.sam_lora96_96 import build_sam_vit_b_adapter_linknet_lora96_96

import ever as er
import cv2
import copy
from tqdm import tqdm
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='dist_b_adapter',
                    help='the name of model')
parser.add_argument('--SAM_pretrained_path', type=str, default='./sam_vit_b_01ec64.pth',
                    help='path of SAM_pretrained_weight')
parser.add_argument('--viz_dir', type=str, default='./record_bd/viz_sota/',
                    help='Path to save weight path')
parser.add_argument('--val_img_dir', type=str, default='./dataset/eng_img/',
                    help='Path to ')
parser.add_argument('--val_gt_dir', type=str, default='./dataset/eng_mask/',
                    help='Path to ')
parser.add_argument('--image_size', type=int, default=640, help='image crop size')

args = parser.parse_args()

def evaluate(img_dir, gt_dir, viz_dir, model, encoder_global_attn_indexes, img_size=1024):
    val = os.listdir(img_dir)
    model = copy.deepcopy(model)
    model.eval()
    if 'sam' in viz_dir:
        model.enc = resize_model_pos_embed(model.module.enc,
                                           img_size=img_size,
                                           encoder_global_attn_indexes=encoder_global_attn_indexes)

    else:
        model.enc = model.module.enc

    os.makedirs(viz_dir, exist_ok=True)

    pm = er.metric.PixelMetric(2, logdir=None, class_names=['bg', 'building'])
    for i, name in tqdm(enumerate(val), total=len(val), desc='Evaluation'):
        img = cv2.imread(img_dir + name)

        m = img.shape[0]
        n = img.shape[1]
        tem = np.zeros(shape=(512, 512, 3))

        tem[:m, :n, :] = img

        img_in = tem.transpose(2, 0, 1)
        img_in = torch.from_numpy(np.array(img_in, np.float32) / 255.0 * 3.2 - 1.6)
        img_in = img_in.unsqueeze(0)

        with torch.no_grad():
            road_output = model(img_in).squeeze().cpu().numpy()

        mask = road_output[:m, :n]
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        # visualization
        vis_mask = 255 * mask
        _mask = np.concatenate([vis_mask[:, :, None], vis_mask[:, :, None], vis_mask[:, :, None]], axis=2)

        cv2.imwrite(viz_dir + '/' + name[:-4] + '.png', _mask.astype(np.uint8))
        gt = cv2.imread(os.path.join(gt_dir, name.replace('.jpg', '.png')))

        # gt = cv2.imread(os.path.join(gt_dir, name.replace('RGB-PanSharpen', 'Mask')))

        gt = np.where(gt == 255, np.ones_like(gt), np.zeros_like(gt))[:, :, 0]

        pm.forward(gt, mask)

    tb = pm.summary_all()
    return tb

def evaluate_eng(img_dir, gt_dir, viz_dir, model, img_size=1024):
    val = os.listdir(img_dir)
    model = copy.deepcopy(model)
    model.eval()

    os.makedirs(viz_dir, exist_ok=True)

    pm = er.metric.PixelMetric(2, logdir=None, class_names=['bg', 'building'])
    for i, name in tqdm(enumerate(val), total=len(val), desc='Evaluation'):
        img = cv2.imread(img_dir + name)

        m = img.shape[0]
        n = img.shape[1]

        tem = np.zeros(shape=(512, 512, 3))

        tem[:m, :n, :] = img
        img_in = tem.transpose(2, 0, 1)
        img_in = torch.from_numpy(np.array(img_in, np.float32) / 255.0 * 3.2 - 1.6)
        img_in = img_in.unsqueeze(0)

        with torch.no_grad():
            road_output = model(img_in).squeeze().cpu().numpy()

        mask = road_output[:m, :n]
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        # visualization
        vis_mask = 255 * mask
        _mask = np.concatenate([vis_mask[:, :, None], vis_mask[:, :, None], vis_mask[:, :, None]], axis=2)

        cv2.imwrite(viz_dir + '/' + name[:-4] + '.png', _mask.astype(np.uint8))

        gt = cv2.imread(os.path.join(gt_dir, name.replace('jpg', 'png')))

        gt = np.where(gt == 255, np.ones_like(gt), np.zeros_like(gt))[:, :, 0]
        pm.forward(gt, mask)

    tb = pm.summary_all()
    return tb

def load(model, path):
    model.load_state_dict(torch.load(path))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    setup_seed(2333)

    # fix the seed for reproducibility
    cudnn.benchmark = True

    if args.name == 'b_adapter_sam_sp24':
        model, encoder_global_attn_indexes = build_sam_vit_b_adapter_linknet(
            args.SAM_pretrained_path,
            image_size=args.image_size)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.load_state_dict(torch.load('./record_bd/weight/b_adapter_sam_sp24/b_adapter_sam_sp24_60.th'))

    if args.name == 'b_adapter_sam_multi_lora32_sp24':
        model, encoder_global_attn_indexes = build_sam_vit_b_adapter_linknet_multi_lora(args.SAM_pretrained_path, image_size=args.image_size)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.load_state_dict(torch.load('./record_bd/weight/b_adapter_sam_multi_lora32_sp24/b_adapter_sam_multi_lora32_sp24_60.th'))

    if args.name == 'b_adapter_sam_lora96_96_sp24':
        model, encoder_global_attn_indexes = build_sam_vit_b_adapter_linknet_lora96_96(args.SAM_pretrained_path,image_size=args.image_size)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.load_state_dict(torch.load('./record_bd/weight/b_adapter_sam_lora96_96_sp24/b_adapter_sam_lora96_96_sp24_60.th'))

    log_path = './record_bd/log_sota/' + args.name

    os.makedirs(log_path, exist_ok=True)
    mylog = open(log_path + '/' + args.name + '.log', 'w')

    # 2116, 650*650
    # sp_val_img_dir = '/data1/_backups/lxy_backup/data/SpaceNet2Build/test/img/'
    # sp_val_gt_dir = '/data1/_backups/lxy_backup/data/SpaceNet2Build/test/mask/'
    # tb = evaluate(
    #         img_dir=sp_val_img_dir,
    #         gt_dir=sp_val_gt_dir,
    #         viz_dir=args.viz_dir + args.name + '/' + args.name + '_sp',
    #         model=model,
    #         encoder_global_attn_indexes=encoder_global_attn_indexes,
    #         img_size=672
    # )
    # print(tb, file=mylog)

    eng_val_img_dir = '/data1/_backups/lxy_backup/data/large_scale/eng_building_img512/'
    eng_val_gt_dir = '/data1/_backups/lxy_backup/data/large_scale/eng_building_mask512/'
    tb = evaluate_eng(
        img_dir=eng_val_img_dir,
        gt_dir=eng_val_gt_dir,
        viz_dir=args.viz_dir + args.name + '_eng512',
        model=model,
        img_size=512
    )
    print(tb, file=mylog)


    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
