from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from vehiclereid.data_manager import ImageDataManager
from vehiclereid import models
from vehiclereid.losses import CrossEntropyLoss, TripletLoss, CenterLoss, DeepSupervision
from vehiclereid.utils.iotools import check_isfile, mkdir_if_missing
from vehiclereid.utils.avgmeter import AverageMeter
from vehiclereid.utils.loggers import Logger, RankLogger
from vehiclereid.utils.torchtools import count_num_param, accuracy, \
    load_pretrained_weights, save_checkpoint, resume_from_checkpoint
from vehiclereid.utils.visualtools import visualize_ranked_results
from vehiclereid.utils.generaltools import set_random_seed
from vehiclereid.eval_metrics import evaluate
from vehiclereid.optimizers import init_optimizer
from vehiclereid.lr_schedulers import init_lr_scheduler
import shutil
import pickle

from re_ranking_features import re_ranking

# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args

    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))

    print('==========\nArgs:{}\n=========='.format(args))

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    print('Initializing image data manager')
    dm = ImageDataManager(use_gpu, **dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()
    galleryloader = testloader_dict[args.target_names[0]]['gallery']
    queryloader = testloader_dict[args.target_names[0]]['query']

    model = models.init_model(name=args.arch, num_classes= 37, loss={'xent', 'htri', 'center'},
                              pretrained=not args.no_pretrained, use_gpu=use_gpu)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    feature_dim = model.feature_dim

    load_pretrained_weights(model, args.load_weights)

    model = nn.DataParallel(model).cuda() if use_gpu else model

    model.eval()

    query_pid2label = {}
    gallery_pid2label = {}

    with torch.no_grad():
        for batch_idx, (imgs, pids, camids, img_paths) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()
            outputs, _ = model(imgs)
            outputs = outputs.data.cpu()
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(img_paths)):
                query_pid2label[img_paths[i][-10:]] = int(predicted[i])
    
    with open("query_color.pkl", 'wb') as f:
        pickle.dump(query_pid2label, f)
    #print(query_pid2label)

    with open("test_track2pid.pkl", 'rb') as f:
        gallery_track2pid = pickle.load(f)

    with torch.no_grad():
        for batch_idx, (imgs, pids, camids, img_paths) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            outputs, _ = model(imgs)
            outputs = outputs.data.cpu()
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(img_paths)):
                gallery_pid2label[img_paths[i][-10:]] = int(predicted[i])
    
    with open("test_color.pkl", 'wb') as f:
        pickle.dump(gallery_pid2label, f)
    #print(gallery_pid2label)

    track2label = {}
    for i in range(len(gallery_track2pid)):
        track2label[i] = gallery_track2pid[i].copy()
        for j in range(len(track2label[i])):
            track2label[i][j] = gallery_pid2label[track2label[i][j]]
        track2label[i] = most_common(track2label[i])
    
    with open("test_track2color.pkl", "wb") as f:
        pickle.dump(track2label, f)
    
    print(track2label)

def most_common(lst):
    return max(set(lst), key=lst.count)

if __name__ == '__main__':
    main()