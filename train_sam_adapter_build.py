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
from networks.sam_lora96_96 import build_sam_vit_b_adapter_linknet_lora96_96
from networks.sam_multi_lora import build_sam_vit_b_adapter_linknet_multi_lora

from utils.loss import dice_bce_loss
from utils.data import SP2Build_ImageFolder, Ada_Hist
from functools import reduce
import ever as er
import cv2
import copy
from tqdm import tqdm
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/data1/_backups/lxy_backup/data/SpaceNet2Build/train24/',
                    help='path of the training dataset')
parser.add_argument('--name', type=str, default='dist_b_adapter',
                    help='the name of model')
parser.add_argument('--SAM_pretrained_path', type=str, default='./sam_vit_b_01ec64.pth',
                    help='path of SAM_pretrained_weight')
parser.add_argument('--log_dir', type=str, default='./record_bd/log/',
                    help='path of logs')
parser.add_argument('--weight_dir', type=str, default='./record_bd/weight/',
                    help='Path to save weight path')
parser.add_argument('--viz_dir', type=str, default='./record_bd/viz/',
                    help='Path to save weight path')
parser.add_argument('--val_img_dir', type=str, default='/data1/_backups/lxy_backup/data/SpaceNet2Build/test35/img/',
                    help='Path to ')
parser.add_argument('--val_gt_dir', type=str, default='/data1/_backups/lxy_backup/data/SpaceNet2Build/test35/mask/',
                    help='Path to ')
parser.add_argument('--image_size', type=int, default=640, help='image crop size')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--seed', type=int, default=2333, help='random seed')
parser.add_argument('--base_lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--only_eval', action='store_true', help='')
parser.add_argument('--weight_path', type=str, default=None)
parser.add_argument('--epochs', default=60, type=int,
                    help='number of training epochs')
parser.add_argument('--use_rd_branch', type=bool, default=False, help='whether to use road detail branch')# distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')
parser.add_argument('--device', default='cuda',help='device to use for training / testing')
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
parser.set_defaults(pin_mem=True)

args = parser.parse_args()

def evaluate(img_dir, gt_dir, viz_dir, model, encoder_global_attn_indexes, img_size=1024):
    val = os.listdir(img_dir)
    model = copy.deepcopy(model)
    model.eval()

    if 'sam' in viz_dir:
        model.enc = resize_model_pos_embed(model.module.enc,
                                           img_size=img_size,
                                           encoder_global_attn_indexes=encoder_global_attn_indexes)

    os.makedirs(viz_dir, exist_ok=True)

    pm = er.metric.PixelMetric(2, logdir=None, class_names=['bg', 'building'])
    for i, name in tqdm(enumerate(val), total=len(val), desc='Evaluation'):
        img = cv2.imread(img_dir + name)
        if 'paris' or 'khart' in viz_dir:
            img = Ada_Hist(img)

        m = img.shape[0]
        n = img.shape[1]

        if 'paris' or 'khart' in viz_dir:
            tem = np.zeros(shape=(672, 672, 3))

        if 'whu' in viz_dir:
            tem = np.zeros(shape=(512, 512, 3))

        tem[:m, :n, :] = img
        img_in = tem.transpose(2, 0, 1)
        img_in = torch.from_numpy(np.array(img_in, np.float32) / 255.0 * 3.2 - 1.6)
        img_in = img_in.unsqueeze(0)

        with torch.no_grad():
            road_output = model(img_in.to(args.device)).squeeze().cpu().numpy()

        mask = road_output[:m, :n]
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        # visualization
        vis_mask = 255 * mask
        _mask = np.concatenate([vis_mask[:, :, None], vis_mask[:, :, None], vis_mask[:, :, None]], axis=2)

        cv2.imwrite(viz_dir + '/' + name[:-4] + '.png', _mask.astype(np.uint8))

        if 'paris' or 'khart' in viz_dir:
            gt = cv2.imread(os.path.join(gt_dir, name.replace('RGB-PanSharpen', 'Mask')))

        if 'whu' in viz_dir:
            print()
            gt = cv2.imread(os.path.join(gt_dir, name.replace('jpg', 'png')))

        gt = np.where(gt == 255, np.ones_like(gt), np.zeros_like(gt))[:, :, 0]
        pm.forward(gt, mask)

    tb = pm.summary_all()
    return tb

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save(model, path):
    torch.save(model.state_dict(), path)

def load(model, path):
    model.load_state_dict(torch.load(path))

def update_lr(optimizer, old_lr, new_lr, factor=False):
    if factor:
        new_lr = old_lr / new_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    print('update learning rate: %f -> %f' % (old_lr, new_lr))
    return new_lr

def main(args):
    log_path = os.path.join(args.log_dir, args.name)
    os.makedirs(log_path, exist_ok=True)

    save_path = os.path.join(args.weight_dir, args.name)
    os.makedirs(save_path, exist_ok=True)

    mylog = open(log_path + '/' + args.name + '.log', 'w')
    tic = time.time()
    no_optim = 0
    train_epoch_best_loss = 100.

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    setup_seed(seed)

    cudnn.benchmark = True

    image_list = filter(lambda x: x.find('png') != -1, os.listdir(args.data_dir))
    train_list = list(map(lambda x: x[:-4], image_list))

    # dataset_train = build_ImageFolder(train_list, args.data_dir)
    dataset_train = SP2Build_ImageFolder(train_list, args.data_dir, size=(640, 640))
    # print(dataset_train)

    if log_path is not None:
        os.makedirs(log_path, exist_ok=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    if args.name == 'b_adapter_sam_sp24':
        model, encoder_global_attn_indexes = build_sam_vit_b_adapter_linknet(
            args.SAM_pretrained_path,
            image_size=args.image_size)

    if args.name == 'b_adapter_sam_lora96_96_sp24':
        model, encoder_global_attn_indexes = build_sam_vit_b_adapter_linknet_lora96_96(
            args.SAM_pretrained_path,
            image_size=args.image_size)

    if args.name == 'b_adapter_sam_multi_lora32_sp24':
        model, encoder_global_attn_indexes = build_sam_vit_b_adapter_linknet_multi_lora(
            args.SAM_pretrained_path,
            image_size=args.image_size)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if 'lora' in args.name:
        lr = 1e-4
    else:
        lr = args.base_lr

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    train_loss = dice_bce_loss()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(1, args.epochs+1):
        data_loader_iter = iter(data_loader_train)
        train_epoch_loss = 0

        for img, mask in data_loader_iter:
            optimizer.zero_grad()
            pred = model.forward(img.to(device))
            loss = train_loss(mask.to(device), pred)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        train_epoch_loss /= len(data_loader_iter)
        print('********', file=mylog)
        print('epoch:', epoch, '    time:', int(time.time() - tic), file=mylog)
        print('train_loss:', train_epoch_loss, file=mylog)
        print('SHAPE:', (args.image_size, args.image_size), file=mylog)

        print('********')
        print('epoch:', epoch, '    time:', int(time.time() - tic))
        print('train_loss:', train_epoch_loss)
        print('SHAPE:', (args.image_size, args.image_size))

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            save(model_without_ddp, save_path + '/' + args.name + '.th')

        if epoch % 10 == 0:
            load(model_without_ddp, save_path + '/' + args.name + '.th')
            lr = update_lr(optimizer, lr, 2.0, factor=True)
            save(model_without_ddp, save_path + '/' + args.name + '_' + str(epoch) + '.th')

            paris_img_dir = '/data1/_backups/lxy_backup/data/SpaceNet2Build/test_paris/img/'
            paris_gt_dir = '/data1/_backups/lxy_backup/data/SpaceNet2Build/test_paris/mask/'
            tb = evaluate(
                img_dir=paris_img_dir,
                gt_dir=paris_gt_dir,
                viz_dir=args.viz_dir + args.name + '_paris',
                model=model_without_ddp,
                encoder_global_attn_indexes=encoder_global_attn_indexes,
                img_size=672
            )
            print(tb, file=mylog)

            khart_img_dir = '/data1/_backups/lxy_backup/data/SpaceNet2Build/test_khart/img/'
            khart_gt_dir = '/data1/_backups/lxy_backup/data/SpaceNet2Build/test_khart/mask/'
            tb = evaluate(
                img_dir=khart_img_dir,
                gt_dir=khart_gt_dir,
                viz_dir=args.viz_dir + args.name + '_khart',
                model=model_without_ddp,
                encoder_global_attn_indexes=encoder_global_attn_indexes,
                img_size=672
            )
            print(tb, file=mylog)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    save(model_without_ddp, save_path + '/' + args.name + '_60.th')
    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
