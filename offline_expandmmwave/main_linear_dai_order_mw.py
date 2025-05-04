# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import utils

import models_dai
from cosmo import LinearClassifierAttn, MLP

import wandb

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def get_args_parser():
    parser = argparse.ArgumentParser(description='Linear probe evaluation', add_help=False)
    parser.add_argument('--dataset', default='utd', help='dataset name')
    parser.add_argument('--root', default='', type=str, help='path to dataset root')  
    parser.add_argument('--metadata', default='', type=str,
                        help='path to metadata file (see README for details)')  
    parser.add_argument('--guide_flag', type=int, default=0)
    parser.add_argument('--output-dir', default='./', type=str)
    # Model
    parser.add_argument('--model', default='CLIP_VITS16', type=str)
    parser.add_argument('--ssl-mlp-dim', default=4096, type=int,
                        help='hidden dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-emb-dim', default=256, type=int,
                        help='output embed dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-scale', default=1.0, type=float,
                        help='loss scale for SimCLR objective')
    parser.add_argument('--ssl-temp', default=0.1, type=float,
                        help='softmax temperature for SimCLR objective')
    parser.add_argument('--num_class', type=int, default=55, help='num_class')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_base_patch16_224',
                        help='model architecture: (default: ViT-B/16)')
    parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                        help='number of data loading workers (default: 64)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N',
                        help='number of samples per-device/per-gpu ')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.001, type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to CLIP pretrained checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--latest', default='', type=str)
    parser.add_argument('--mm_seed')
    return parser

best_acc1 = 0


def main(args):
    args.distributed = False

    global best_acc1

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # create model
    print("=> creating model: {}".format(args.model))
    model = getattr(models_dai, args.model)(ssl_mlp_dim=args.ssl_mlp_dim, ssl_emb_dim=args.ssl_emb_dim)
    model.cuda(args.gpu)

    latest = args.latest
    if os.path.isfile(latest):
        print("=> loading checkpoint '{}'".format(latest))
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        latest_checkpoint = torch.load(latest, map_location=loc)
        model.load_state_dict(latest_checkpoint['state_dict'], strict=False)

    args.start_epoch = 0

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # classifier = LinearClassifierAttn(num_classes=args.num_class, guide=args.guide_flag)
    # classifier = MLP(input_dim=512, hidden_dim=256, num_classes=args.num_class, num_layers=1, num_features=1) #nn.Linear(512, args.num_class)
    classifier = MLP(input_dim=512*2, hidden_dim=256, num_classes=args.num_class, num_layers=1, num_features=2, weights=[0.3, 0.7])
    classifier.cuda(args.gpu)

    # init_lr = args.lr * int(args.batch_size / utils.get_world_size()) / 256
    args.workers = int((args.workers + utils.get_world_size() - 1) / utils.get_world_size())

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    classifier_parameters = filter(lambda p: p.requires_grad, classifier.parameters())
    parameters = list(classifier_parameters)

    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    print("=> creating dataset")  

    # Load pre-encoded data
    def load_encoded_data(load_path):
        with open(load_path, 'rb') as f:
            encoded_data = pickle.load(f)
        encoded_wifi, encoded_mmwave, labels = zip(*encoded_data)
        return list(encoded_wifi), list(encoded_mmwave), list(labels)

    encoded_data_path = 'encoded_train_data.pkl'
    X_wifi_train, X_mmwave_train, y_train = load_encoded_data(encoded_data_path)
    print('Load encoded dataset done')

    encoded_data_path = 'encoded_test_data.pkl'
    X_wifi_test, X_mmwave_test, y_test = load_encoded_data(encoded_data_path)
    print('Loaded encoded test dataset')

    # Creating a one-shot learning split
    train_indices = []
    test_indices = []

    unique_labels = np.unique(y_train)
    for label in unique_labels:
        label_indices = np.where(np.array(y_train) == label)[0]
        random.seed(0)
        random.shuffle(label_indices)
        train_indices.append(label_indices[0])  # Take one sample per class for training

    # Prepare training and testing data
    X_wifi_train_one_shot = [X_wifi_train[idx] for idx in train_indices]
    X_mmwave_train_one_shot = [X_mmwave_train[idx] for idx in train_indices]
    y_train_one_shot = [y_train[idx] for idx in train_indices]

    batch_size = args.batch_size

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_wifi_train_one_shot, X_mmwave_train_one_shot, y_train_one_shot)), batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        list(zip(X_wifi_test, X_mmwave_test, y_test)), batch_size=512, shuffle=False)

    # Print some statistics
    print(f"Number of training samples: {len(train_indices)}")
    print(f"Number of testing samples: {len(test_indices)}")
    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.wandb:
        wandb.init(project='slip', config=args, resume='allow')        

    print(args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, args.lr, epoch, args)

        # train for one epoch
        train_stats = train(train_loader, model, classifier, criterion, optimizer, epoch, args)

        if (epoch + 1) % args.eval_freq != 0:
            continue

        # evaluate on validation set
        final = epoch == args.epochs-1
        val_stats = validate(test_loader, model, classifier, criterion, args, final=final)
        acc1 = val_stats['acc1']

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        print('best_acc1', best_acc1)

        if utils.is_main_process(): # only the first GPU saves checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}
        if args.wandb:
            wandb.log(log_stats)  

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, 'linear_{}_lr={}_log.txt'.format(args.dataset, args.lr)), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
    
    return best_acc1


def train(train_loader, model, classifier, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()
    classifier.train()

    end = time.time()
    for i, (wifi_data, mmwave_data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = torch.tensor(target).cuda()

        output = model(wifi_data, mmwave_data)
        output = classifier(output['csi_embed'], output['mmwave_embed'])

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), len(mmwave_data))
        top1.update(acc1.item(), len(mmwave_data))
        top5.update(acc5.item(), len(mmwave_data))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return {'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg}


def validate(val_loader, model, classifier, criterion, args, final=True):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    if final:
        num_classes = 55
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        end = time.time()
        for i, (wifi_data, mmwave_data, target) in enumerate(val_loader):

            target = torch.tensor(target).cuda()

            output = model(wifi_data, mmwave_data)
            output = classifier(output['csi_embed'], output['mmwave_embed'])

            if final:
                # Get predictions
                _, pred = output.topk(1, 1, True, True)
                pred = pred.t()

                # Update confusion matrix
                for t, p in zip(target.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), len(mmwave_data))
            top1.update(acc1.item(), len(mmwave_data))
            top5.update(acc5.item(), len(mmwave_data))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    if final:
        # print_confusion_analysis(confusion_matrix, range(27)) # Prints the misclassified data for each class
        plot_confusion_matrix(confusion_matrix, range(55))

    return {'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg}

def print_confusion_analysis(confusion_matrix, classes):
    for i in classes:
        correct = confusion_matrix[i, i]
        total = confusion_matrix[i, :].sum()
        acc = 100.0 * correct / total
        misclassified_as = np.argsort(confusion_matrix[i, :])[-2]  # The class that this class is most frequently misclassified as
        print(f'Class {i} - Accuracy: {acc:.2f}% - Most frequently confused with: {classes[misclassified_as]}')

def plot_confusion_matrix(confusion_matrix, classes):
    plt.figure(figsize=(10, 10))

    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", 
                xticklabels=classes, yticklabels=classes, fmt="d", 
                cbar=True, linewidths=.5)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig('./Confusion Matrix.png')
    plt.close()



def save_checkpoint(state, is_best, output_dir):
    ckpt_path = f'{output_dir}/linear_checkpoint.pt'
    best_path = f'{output_dir}/linear_best.pt'
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copyfile(ckpt_path, best_path)


def sanity_check(state_dict, pretrained_weights, linear_keyword, visual_keyword):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore linear layer
        if '%s.weight' % linear_keyword in k or '%s.bias' % linear_keyword in k:
            continue

        # name in pretrained model
        k_pre = visual_keyword + k[len('module.'):] \
            if k.startswith('module.') else visual_keyword + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def objective(trial):
    # Suggest hyperparameters to be tuned
    # lr = trial.suggest_categorical('lr', [0.01, 0.1, 0.5])
    # batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    mm_seed = trial.suggest_categorical('mm_seed', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    # Update the args with the suggested hyperparameters
    args = get_args_parser().parse_args()
    # args.lr = lr
    # args.batch_size = batch_size
    args.mm_seed = mm_seed

    # Call the main function with the updated args
    acc1 = main(args)
    
    return acc1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Linear probe evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    acc1 = main(args)

    # args = get_args_parser().parse_args()
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=10)

    # print('Best hyperparameters:', study.best_params)
    
    # with open("experiment_results_wifi_mmwave.txt", "a") as file:
    #     file.write(f"Model: {args.latest}\n")
    #     # file.write(str(acc1))
    #     # file.write('\n')
        
    #     best_trial = study.best_trial
    #     file.write(f'Best is trial {best_trial.number} with value: {best_trial.value}.\n')
    #     file.write(f'Best hyperparameters: {best_trial.params}\n')
