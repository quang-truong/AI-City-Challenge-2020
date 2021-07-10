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
from vehiclereid.losses import CrossEntropyLoss, TripletLoss, DeepSupervision
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
from torch.nn import functional as F
import shutil
import pickle
from scipy import stats

from re_ranking_features import re_ranking_fea, re_ranking_dist
from sklearn.preprocessing import normalize

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

    if args.combine_eval_dist is not None:
        print("Combined Evaluate only")
        name = args.target_names[0]
        dataset = dm.return_testdataset_by_name(name)
        combine_eval_dist(dataset)
        return

    if args.combine_eval_fea is not None:
        print("Combined Evaluate only")
        name = args.target_names[0]
        dataset = dm.return_testdataset_by_name(name)
        combine_eval_fea(dataset)
        return

    if args.combine_predict is not None:
        print('Combined Predict only')
        name = args.target_names[0]
        dataset = dm.return_testdataset_by_name(name)
        ranks= 100
        save_dir = args.save_dir
        combine_predict(dataset, ranks, save_dir)
        return

    print('Initializing model: {}'.format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent', 'htri'},
                              pretrained=not args.no_pretrained, use_gpu=use_gpu)
    print('Model size: {:.3f} M'.format(count_num_param(model)))


    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model = nn.DataParallel(model).cuda() if use_gpu else model

    criterion_xent = CrossEntropyLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_htri = TripletLoss(margin=args.margin)

    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)

    if args.predict:
        print('Predict only')
        name = args.target_names[0]
        queryloader = testloader_dict[name]['query']
        galleryloader = testloader_dict[name]['gallery']
        dataset = dm.return_testdataset_by_name(name)
        ranks= [1,5,10,20,100]
        save_dir = args.save_dir
        predict(model, queryloader, galleryloader, dataset, use_gpu, ranks, save_dir)
        return

    if args.evaluate:
        print('Evaluate only')

        for name in args.target_names:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            distmat = test(model, queryloader, galleryloader, use_gpu, return_distmat=True)

            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, 'ranked_results', name),
                    topk=20
                )
        return

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

        if not (args.source_names[0] == "AIC20_ReID_Full"):
            if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                    epoch + 1) == args.max_epoch:
                print('=> Test')

                if (args.source_names[0] == "AIC20_ReID_Simu_Color"):
                    galleryloader = testloader_dict[args.target_names[0]]['gallery']
                    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

                    correct = 0
                    total = 0
                    
                    class_correct = list(0. for i in range(12))
                    class_total = list(0. for i in range(12))

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
                    for i in range(12):
                        if (class_total[i] == 0):
                            continue
                        print('Accuracy of %d : %f ' % (
                            classes[i], class_correct[i] / class_total[i]))
                    rank1 = np.nan
                
                elif (args.source_names[0] == "AIC20_ReID_Simu_Type"):
                    galleryloader = testloader_dict[args.target_names[0]]['gallery']
                    classes = (0, 1, 2, 3, 4, 5, 6)

                    correct = 0
                    total = 0
                    
                    class_correct = list(0. for i in range(7))
                    class_total = list(0. for i in range(7))

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
                    for i in range(7):
                        if (class_total[i] == 0):
                            continue
                        print('Accuracy of %d : %f ' % (
                            classes[i], class_correct[i] / class_total[i]))
                    rank1 = np.nan
                
                else:
                    for name in args.target_names:
                        print('Evaluating {} ...'.format(name))
                        queryloader = testloader_dict[name]['query']
                        galleryloader = testloader_dict[name]['gallery']
                        rank1 = test(model, queryloader, galleryloader, use_gpu)
                        ranklogger.write(name, epoch + 1, rank1)

                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'rank1': rank1,
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'optimizer': optimizer.state_dict(),
                }, args.save_dir)
        else:
            if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                    epoch + 1) == args.max_epoch:
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'rank1': np.nan,
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'optimizer': optimizer.state_dict(),
                }, args.save_dir)
                print("Model has been trained and saved at ", args.save_dir)

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))
    if not (args.source_names[0] == "AIC20_ReID_Full"):
        ranklogger.show_summary()


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


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20, 100], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print('Computing CMC and mAP')
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, args.target_names)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')

    if (args.source_names[0] == "AIC20_ReID"):

        print('Re-ranking')
        rr_distmat = re_ranking_fea(qf.numpy(), gf.numpy(), k1= 20, k2= 6, lambda_value = 0.3, Minibatch = 100)

        print('Computing CMC and mAP')
        # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, args.target_names)
        rr_cmc, rr_mAP = evaluate(rr_distmat, q_pids, g_pids, q_camids, g_camids)

        print('Results ----------')
        print('Re-ranking mAP: {:.1%}'.format(rr_mAP))
        print('Re-ranking CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, rr_cmc[r - 1]))
        print('------------------')

        if return_distmat:
            return rr_distmat
        return rr_cmc[0]
    
    else:
        return cmc[0]

def cp_img_to(src, dst, rank, prefix):
        """
        - src: image path or tuple (for vidreid)
        - dst: target directory
        - rank: int, denoting ranked position, starting from 1
        - prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

def predict(model, queryloader, galleryloader, dataset, use_gpu, ranks, save_dir):
    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()
            features = model(imgs)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            features = model(imgs)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        qg_distmat = distmat.numpy()
        #qg_distmat = cosine_distance(qf, gf)
        #qg_distmat = qg_distmat.numpy()
        print(qg_distmat.shape)
        print('Computed QG Distance Matrix')

        m, n = qf.size(0), qf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, qf.t())
        qq_distmat = distmat.numpy()
        #qq_distmat = cosine_distance(qf, qf)
        #qq_distmat = qq_distmat.numpy()
        print('Computed QQ Distance Matrix')

        m, n = gf.size(0), gf.size(0)
        distmat = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, gf, gf.t())
        gg_distmat = distmat.numpy()
        #gg_distmat = cosine_distance(gf, gf)
        #gg_distmat = gg_distmat.numpy()
        print('Computed GG Distance Matrix')

        #print('Computing CMC and mAP')
        # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, args.target_names)
        #cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

        #print('Results ----------')
        #print('mAP: {:.1%}'.format(mAP))
        #print('CMC curve')
        #for r in ranks:
        #    print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
        #print('------------------')

        #rr_distmat = re_ranking_dist(qg_distmat, qq_distmat, gg_distmat, k1= 75, k2= 10, lambda_value = 0.3)
        rr_distmat = re_ranking_dist(qg_distmat, qq_distmat, gg_distmat, k1= 20, k2= 6, lambda_value = 0.3)
        #rr_distmat = re_ranking_fea(qf.numpy(), gf.numpy(),k1 = 20,k2 = 6,lambda_value = 0.3, MemorySave = False, Minibatch = 100)
        print('Re-ranked')

        #print('Computing CMC and mAP')
        # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, args.target_names)
        #rr_cmc, rr_mAP = evaluate(rr_distmat, q_pids, g_pids, q_camids, g_camids)

        #print('Results ----------')
        #print('Re-ranking mAP: {:.1%}'.format(rr_mAP))
        #print('Re-ranking CMC curve')
        #for r in ranks:
        #    print('Rank-{:<3}: {:.1%}'.format(r, rr_cmc[r - 1]))
        #print('------------------')

        with open(args.arch + "-" + args.target_names[0] + "-"+ "distmat.pkl", 'wb') as f:
            pickle.dump([rr_distmat, q_pids, q_camids, g_pids, g_camids], f)

        print('Saved distance matrix!')

    print("Done")

def combine_eval_dist(dataset, ranks=[1, 5, 10, 20, 100]):
    with open("train_track2pid.pkl", "rb") as f:
        train_track2pid = pickle.load(f)
    with open("train_pid2track.pkl", "rb") as f:
        train_pid2track = pickle.load(f)

    query, gallery = dataset
    distmat_tmp = 0
    divisor = 0
    n = 0
    for i in args.combine_eval_dist:
        with open(i, 'rb') as f:
            rr_distmat, q_pids, q_camids, g_pids, g_camids = pickle.load(f)

        num_q, num_g = rr_distmat.shape
    
        assert num_q == len(query)
        assert num_g == len(gallery)

        rr_distmat = track_average_dist(rr_distmat, query, gallery, train_track2pid, train_pid2track)
        
        cmc, mAP = evaluate(rr_distmat, q_pids, g_pids, q_camids, g_camids)
        print('Results ----------')
        print('Re-ranking mAP: {:.1%}'.format(mAP))
        print('Re-ranking CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
        print('------------------')
        distmat_tmp += rr_distmat*args.combine_lambda[n]
        divisor += args.combine_lambda[n]
        n += 1

    distmat_tmp /= divisor
    num_q, num_g = distmat_tmp.shape
    
    assert num_q == len(query)
    assert num_g == len(gallery)

    print('Computing CMC and mAP')
    cmc, mAP = evaluate(distmat_tmp, q_pids, g_pids, q_camids, g_camids)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')

    return

def combine_eval_fea(dataset, ranks=[1, 5, 10, 20, 100]):
    with open("train_track2pid.pkl", "rb") as f:
        train_track2pid = pickle.load(f)
    with open("train_pid2track.pkl", "rb") as f:
        train_pid2track = pickle.load(f)
    
    query, gallery = dataset
    qg_distmat = 0
    qq_distmat = 0
    gg_distmat = 0
    divisor = 0
    c = 0
    for i in args.combine_eval_fea:
        with open(i, 'rb') as f:
            qf, gf, q_pids, q_camids, g_pids, g_camids = pickle.load(f)
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        qg_distmat_tmp = distmat.numpy()
        print('Computed Distance Matrix')

        m, n = qf.size(0), qf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, qf.t())
        qq_distmat_tmp = distmat.numpy()
        print('Computed Distance Matrix')

        m, n = gf.size(0), gf.size(0)
        distmat = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, gf, gf.t())
        gg_distmat_tmp = distmat.numpy()
        print('Computed Distance Matrix')

        print('Computing CMC and mAP')
        # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, args.target_names)
        cmc, mAP = evaluate(qg_distmat_tmp, q_pids, g_pids, q_camids, g_camids)

        print('Results ----------')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
        print('------------------')
        
        divisor += args.combine_lambda[c]
        c += 1
        qg_distmat += qg_distmat_tmp
        qq_distmat += qq_distmat_tmp
        gg_distmat += gg_distmat_tmp

    qg_distmat /= divisor
    qq_distmat /= divisor
    gg_distmat /= divisor

    rr_distmat = re_ranking_dist(qg_distmat, qq_distmat, gg_distmat)
    print('Re-ranked')

    rr_distmat = track_average_dist(rr_distmat, query, gallery, train_track2pid, train_pid2track)
        
    cmc, mAP = evaluate(rr_distmat, q_pids, g_pids, q_camids, g_camids)
    print('Results ----------')
    print('Re-ranking mAP: {:.1%}'.format(mAP))
    print('Re-ranking CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')
    
    return

def track_average_dist(distmat, query, gallery, track2pid, pid2track):
    avg_track = [[[] for j in range(len(track2pid))] for i in range(distmat.shape[0])]

    num_q, num_g = distmat.shape
    
    assert num_q == len(query)
    assert num_g == len(gallery)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        for g_idx in range(num_g):
            gimg_path, gpid, gcamid = gallery[g_idx]
            avg_track[q_idx][pid2track[gimg_path[-10:]]].append(distmat[q_idx][g_idx])

    for i in range(len(avg_track)):
        for j in range(len(avg_track[i])):
            if (len(avg_track[i][j]) == 0):
                continue
            #print(avg_track[i][j])
            avg_track[i][j] = sum(avg_track[i][j])/len(avg_track[i][j])
            #avg_track[i][j] = max(avg_track[i][j])
            #print(avg_track[i][j])

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        for g_idx in range(num_g):
            gimg_path, gpid, gcamid = gallery[g_idx]
            distmat[q_idx][g_idx] = avg_track[q_idx][pid2track[gimg_path[-10:]]]
    
    return distmat

def combine_predict(dataset, ranks, save_dir):
    with open("test_track2pid.pkl", "rb") as f:
        test_track2pid = pickle.load(f)
    with open("test_pid2track.pkl", "rb") as f:
        test_pid2track = pickle.load(f)

    query, gallery = dataset
    distmat_tmp = 0
    divisor = 0
    n = 0
    for i in args.combine_predict:
        with open(i, 'rb') as f:
            rr_distmat, q_pids, q_camids, g_pids, g_camids = pickle.load(f)

        num_q, num_g = rr_distmat.shape
    
        assert num_q == len(query)
        assert num_g == len(gallery)

        #rr_distmat = track_average_dist(rr_distmat, query, gallery, test_track2pid, test_pid2track)
        distmat_tmp += rr_distmat*args.combine_lambda[n]
        divisor += args.combine_lambda[n]
        n += 1

    distmat_tmp /= divisor
    num_q, num_g = distmat_tmp.shape

    distmat_tmp = track_average_dist(distmat_tmp, query, gallery, test_track2pid, test_pid2track)
    
    assert num_q == len(query)
    assert num_g == len(gallery)

    print('Visualizing top-100 ranks')
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))

    indices = np.argsort(distmat_tmp, axis=1)
    mkdir_if_missing(save_dir)

    submission = ""
    path = osp.join(save_dir, "camid_dist_result")
    mkdir_if_missing(path)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):
            qdir = osp.join(save_dir, osp.basename(qimg_path[0]))
        else:
            qdir = osp.join(save_dir, osp.basename(qimg_path))
        mkdir_if_missing(qdir)
        cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        filename = osp.join(path, qimg_path[-10:-4].zfill(6) + ".txt")
        ls = []

        for g_idx in indices[q_idx, :]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            img_id = gimg_path[-10:-4]
            ls.append(str(int(img_id)))
            if rank_idx != 100:
                submission += str(int(img_id))
                submission += " "
                cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery')
                rank_idx += 1                
            else:
                submission += str(int(img_id))
                submission += "\n"
                cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery')
                rank_idx += 1
            if rank_idx > 100:
                break

        with open(filename, 'w') as f:
            for item in ls:
                f.write("%s\n" % item)
        
        
    
    file = open(osp.join(save_dir, "track2.txt"), "w")
    file.write(submission)
    file.close()

    print("Done")

if __name__ == '__main__':
    main()
