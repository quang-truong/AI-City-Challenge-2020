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

from re_ranking_features import re_ranking_fea, re_ranking_dist

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

    model = models.init_model(name=args.arch, num_classes= dm.num_train_pids, loss={'xent', 'htri'},
                              pretrained=not args.no_pretrained, use_gpu=use_gpu)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model = nn.DataParallel(model).cuda() if use_gpu else model

    if args.evaluate:
        for name in args.target_names:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            test(model, galleryloader, use_gpu)

        return

    if args.extract_features:
        for name in args.target_names:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            extract_features(model, queryloader, galleryloader, use_gpu)

        return

    criterion_xent = CrossEntropyLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_htri = TripletLoss(margin=args.margin)
    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)
    
    

    time_start = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    print('=> Start training')
    '''
    if args.fixbase_epoch > 0:
        print('Train {} for {} epochs while keeping other layers frozen'.format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, fixbase=True)

        print('Done. All layers are open to train for {} epochs'.format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)
    '''
    for epoch in range(args.start_epoch, args.max_epoch):
        train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu)

        print(epoch, optimizer.param_groups[0]['lr'])

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                epoch + 1) == args.max_epoch:
            print('=> Test')
            
            if (args.source_names[0] == "AIC20_ReID_Type"):
                galleryloader = testloader_dict[args.target_names[0]]['gallery']
                classes = (0, 1, 2, 3, 4, 5)

                correct = 0
                total = 0
                
                class_correct = list(0. for i in range(6))
                class_total = list(0. for i in range(6))

                model.eval()
                with torch.no_grad():
                    for batch_idx, (imgs, pids, _, _) in enumerate(galleryloader):
                        if use_gpu:
                            imgs = imgs.cuda()
                        outputs, _ = model(imgs)
                        outputs = outputs.data.cpu()
                        _, predicted = torch.max(outputs.data, 1)
                        total += pids.size(0)
                        correct += (predicted == pids).sum().item()
                        
                        c = (predicted == pids.data).squeeze()
                        for i in range(pids.size(0)):
                            pid = pids.data[i]
                            class_correct[pid] += c[i].item()
                            class_total[pid] += 1
                print('Accuracy of the network: {:4f}'.format(correct/total))
                print('Accuracy by classes')
                for i in range(6):
                    if (class_total[i] == 0):
                        continue
                    print('Accuracy of %d : %f ' % (
                        classes[i], class_correct[i] / class_total[i]))
                rank1 = np.nan

            elif (args.source_names[0] == "AIC20_ReID_CamID"):
                test_camid(model, galleryloader, use_gpu)

            elif (args.source_names[0] == "AIC20_ReID_Color"):
                test_color(model, galleryloader, use_gpu)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'arch': args.arch,
                'optimizer': optimizer.state_dict(),
            }, args.save_dir)
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))

def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    xent_losses = AverageMeter()
    htri_losses = AverageMeter()
    loss_losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()

    model.train()
    for p in model.parameters():
        p.requires_grad = True    # open all layers

    end = time.time()
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        outputs, features = model(imgs)
        if isinstance(outputs, (tuple, list)):
            xent_loss = DeepSupervision(criterion_xent, outputs, pids)
        else:
            xent_loss = criterion_xent(outputs, pids)

        if isinstance(features, (tuple, list)):
            htri_loss = DeepSupervision(criterion_htri, features, pids)
        else:
            htri_loss = criterion_htri(features, pids)

        loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        xent_losses.update(xent_loss.item(), pids.size(0))
        htri_losses.update(htri_loss.item(), pids.size(0))
        loss_losses.update(loss.item(), pids.size(0))
        accs.update(accuracy(outputs, pids)[0])

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Xent {xent.val:.4f} ({xent.avg:.4f})\t'
                  'Htri {htri.val:.4f} ({htri.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader),
                batch_time=batch_time,
                xent=xent_losses,
                htri=htri_losses,
                loss=loss_losses,
                acc=accs
            ))

        end = time.time()

def test_camid(model, galleryloader, use_gpu):
    classes = tuple([i for i in range(36)])

    correct = 0
    total = 0
    
    class_correct = list(0. for i in range(36))
    class_total = list(0. for i in range(36))

    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, pids, _, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            outputs, _ = model(imgs)
            outputs = outputs.data.cpu()
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            print(pids)
            total += pids.size(0)
            correct += (predicted == pids).sum().item()
            
            c = (predicted == pids.data).squeeze()
            for i in range(pids.size(0)):
                pid = pids.data[i]
                class_correct[pid] += c[i].item()
                class_total[pid] += 1
    print('Accuracy of the network: {:4f}'.format(correct/total))
    print('Accuracy by classes')
    for i in range(36):
        if (class_total[i] == 0):
            continue
        print('Accuracy of %d : %f ' % (
            classes[i], class_correct[i] / class_total[i]))

def test_color(model, galleryloader, use_gpu):
    classes = tuple([i for i in range(10)])

    correct = 0
    total = 0
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, pids, _, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            outputs, _ = model(imgs)
            outputs = outputs.data.cpu()
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            print(pids)
            total += pids.size(0)
            correct += (predicted == pids).sum().item()
            
            c = (predicted == pids.data).squeeze()
            for i in range(pids.size(0)):
                pid = pids.data[i]
                class_correct[pid] += c[i].item()
                class_total[pid] += 1
    print('Accuracy of the network: {:4f}'.format(correct/total))
    print('Accuracy by classes')
    for i in range(10):
        if (class_total[i] == 0):
            continue
        print('Accuracy of %d : %f ' % (
            classes[i], class_correct[i] / class_total[i]))

def extract_features(model, queryloader, galleryloader, use_gpu):
    model.eval()
    with torch.no_grad():
        gf, g_pids, g_camids, gids = [], [], [], []
        for batch_idx, (imgs, pids, camids, ids) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            _, features = model(imgs)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
            gids.extend(ids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        qf, q_pids, q_camids, qids = [], [], [], []
        for batch_idx, (imgs, pids, camids, ids) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()
            _, features = model(imgs)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            qids.extend(ids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

    #x1 = np.concatenate((gf.numpy(), qf.numpy()), axis=0)
    #gids.extend(qids)


    #with open(args.arch + "-" + args.target_names[0] + "-"+ "features.pkl", 'wb') as f:
    #    pickle.dump((x1, gids), f)

    
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    qg_distmat = distmat.numpy()
    print('Computed QG Distance Matrix')

    m, n = qf.size(0), qf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, qf.t())
    qq_distmat = distmat.numpy()
    print('Computed QQ Distance Matrix')

    m, n = gf.size(0), gf.size(0)
    distmat = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, gf, gf.t())
    gg_distmat = distmat.numpy()
    print('Computed GG Distance Matrix')

    distmat = compute_original_distmat(qg_distmat, qq_distmat, gg_distmat)

    with open(args.arch + "-" + args.target_names[0] + "-"+ "type-original_distmat.pkl", 'wb') as f:
        pickle.dump(distmat, f)

    print('Saved original distance matrix!')
    print(distmat)
    print(distmat.shape)

def compute_original_distmat(q_g_dist, q_q_dist, g_g_dist):
    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    return original_dist

if __name__ == '__main__':
    main()