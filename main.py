from __future__ import print_function, division
import argparse
import os
from pickle import FALSE, TRUE
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from datasets import __datasets__
import gc
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from models.pbcstereo import PBCStereo

parser = argparse.ArgumentParser(description='PBCStereo')
parser.add_argument('--model', default='PBCStereo', help='select a model structure')
parser.add_argument('--dataset', choices=__datasets__.keys(), default='sceneflow', help='dataset name')
parser.add_argument('--datapath', default='/home/jump/dataset/sceneflow/', help='data path')
parser.add_argument('--savepath', default='./png/', help='save path')
parser.add_argument('--trainlist', default='./filenames/SceneFlow_cleanpass_train.txt', help='training list')
parser.add_argument('--vallist', default='./filenames/SceneFlow_cleanpass_val.txt', help='validating list')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--groups', type=int, default=48, help='number of groups of Cost Aggregation')
parser.add_argument('--logtxt', default='./save/sceneflow C=64/log.txt', help='train data')
parser.add_argument('--loadmodel', default=None, help='load model')
parser.add_argument('--savemodel', default='./save/sceneflow C=64/', help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

datapath = args.datapath
StereoDataset = __datasets__[args.dataset]
kitti_train = args.trainlist
kitti_val = args.vallist
kitti_train_dataset = StereoDataset(datapath, kitti_train, True)
kitti_val_dataset = StereoDataset(datapath, kitti_val, False)
TrainImgLoader = DataLoader(kitti_train_dataset, batch_size=12, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
ValImgLoader = DataLoader(kitti_val_dataset, batch_size=12, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
#torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if (args.model == 'PBCStereo'):
    model = PBCStereo(args.groups)
    print('PBCStereo')
else:
    print('wrong model')
    # return -1

if (args.dataset == 'kitti_12'):
    print('kitti_12')
elif (args.dataset == 'kitti'):
    print('kitti_15')
elif (args.dataset == 'sceneflow'):
    print('sceneflow')
else:
    print('wrong dataset')
    #return -1

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
#with open(args.logtxt, 'a', encoding='utf-8') as f:
#    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])), file=f)
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)


def train(imgL, imgR, disp_L):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
    #---------
    disp_true = disp_L.cuda()
    mask = (disp_true > 0) & (disp_true < 192)
    mask.detach_()
    #----
    optimizer.zero_grad()
    pred = model(imgL, imgR)
    output = torch.unsqueeze(pred, 1)
    loss = F.smooth_l1_loss(output[mask], disp_true[mask], reduction='mean')  ###因为warning改过 size_average=True?
    loss.backward()
    optimizer.step()
    #torch.cuda.empty_cache()
    return loss.item()


def val(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_true = Variable(torch.FloatTensor(disp_true))
    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

        #---------
        mask = (disp_true > 0) & (disp_true < 192)
        #----

        with torch.no_grad():
            output = model(imgL, imgR)

        output = torch.unsqueeze(output, 1)
        torch.cuda.empty_cache()
        if len(disp_true[mask]) == 0:
            loss = 0
        else:
            loss = torch.mean(torch.abs(output[mask] - disp_true[mask]))  # end-point-error
        return loss


def main():

    min_epe = 192
    max_epo = 0
    start_full_time = time.time()

    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0
        total_test_loss = 0

        ## training ##
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            imgL_crop, imgR_crop, disp_crop_L = sample['left'], sample['right'], sample['disparity']
            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            if batch_idx % 1 == 0:
                print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
        #with open(args.logtxt, 'a', encoding='utf-8') as f:
        #    print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)), file=f)

        ## val ##
        for batch_idx, sample in enumerate(ValImgLoader):
            imgL, imgR, disp_L = sample['left'], sample['right'], sample['disparity']
            test_loss = val(imgL, imgR, disp_L)
            print('Iter %d EPE error in val = %.3f' % (batch_idx, test_loss))
            total_test_loss += test_loss

        print('epoch %d total EPE error in val = %.3f' % (epoch, total_test_loss / len(ValImgLoader)))
        #with open(args.logtxt, 'a', encoding='utf-8') as f:
        #    print('epoch %d total EPE error in val = %.3f' % (epoch, total_test_loss / len(ValImgLoader)), file=f)
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()
        print(lr)
        #SAVE
        if (total_test_loss / len(ValImgLoader)) < min_epe:
            min_epe = total_test_loss / len(ValImgLoader)
            max_epo = epoch
            savefilename = args.savemodel + 'sceneflow_' + str(max_epo) + '.tar'
            torch.save(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss / len(TrainImgLoader),
                    'test_loss': total_test_loss / len(ValImgLoader),
                }, savefilename)

        print('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))


if __name__ == '__main__':
    main()
