import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging

from model.MEFL import MEFARG
from dataset import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env


def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'BP4D':
        trainset = BP4D(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 2)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        valset = BP4D(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 2)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'DISFA':
        trainset = DISFA(conf.dataset_path, train=True, fold=conf.fold, transform=image_train(
            crop_size=conf.crop_size, img_size=conf.image_size), crop_size=conf.crop_size, stage=1)
        if conf.weighted_sampling:
            disfa_weight_map = {0: 1./877405., 1: 1./56642.,
                                2: 1./46597., 3: 1./45985., 4: 1./15936., 5: 1./3947.}
            weights = get_sampler_weights(trainset.data_list, disfa_weight_map)
            weights = torch.DoubleTensor(weights)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights, len(weights))
            train_loader = DataLoader(trainset, batch_size=conf.batch_size,
                                      sampler=sampler, num_workers=conf.num_workers)
        else:
            train_loader = DataLoader(
                trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        valset = DISFA(conf.dataset_path, train=False, fold=conf.fold,
                       transform=image_test(crop_size=conf.crop_size, img_size=conf.image_size), stage=1)
        val_loader = DataLoader(
            valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)
        
    return train_loader, val_loader, len(trainset), len(valset)


# Train
def train(conf, net, train_loader, optimizer, epoch, criterion):
    losses = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    for batch_idx, (inputs,  targets) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len)
        targets= targets.float()
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs, outputs_relation = net(inputs)
        wa_loss = criterion[0](outputs, targets)
        loss = wa_loss
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))

    return losses.avg


# Val
def val(net, val_loader, criterion):
    losses = AverageMeter()
    mae_avg = AverageMeter()
    mse_avg = AverageMeter()
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    net.eval()
    statistics_list = None
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        targets = targets.float()
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = net(inputs)
            loss = criterion[0](outputs, targets)
            mae = mae_loss(outputs, targets)
            mse = mse_loss(outputs, targets)
            losses.update(loss.data.item(), inputs.size(0))
            mae_avg.update(mae.data.item(), inputs.size(0))
            mse_avg.update(mse.data.item(), inputs.size(0))
    #         update_list = statistics(outputs, targets.detach(), 0.5)
    #         statistics_list = update_statistics_list(statistics_list, update_list)
    # mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    # mean_acc, acc_list = calc_acc(statistics_list)
    return losses.avg, mae_avg.avg, mse_avg.avg


def main(conf):
    if conf.dataset == 'BP4D':
        dataset_info = BP4D_infolist
    elif conf.dataset == 'DISFA':
        dataset_info = DISFA_infolist

    start_epoch = 0
    # data
    train_loader,val_loader,train_data_num,val_data_num = get_dataloader(conf)
    train_weight = torch.from_numpy(np.loadtxt(os.path.join(conf.dataset_path, 'list', conf.dataset+'_weight_fold'+str(conf.fold)+'.txt')))
    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold + 1, conf.N_fold, val_data_num))
    net = MEFARG(num_classes=conf.num_classes, backbone=conf.arc)

    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()

    # criterion = [WeightedAsymmetricLoss(weight=train_weight), nn.CrossEntropyLoss()]
    criterion = [nn.MSELoss()]
    optimizer = optim.AdamW(net.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    print('the init learning rate is ', conf.learning_rate)

    best_val_loss = np.inf
    #train and val
    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        train_loss = train(conf,net,train_loader,optimizer,epoch,criterion)
        # val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader, criterion)
        val_loss, val_mae, val_mse = val(
            net, val_loader, criterion)
        # log
        # infostr = {'Epoch:  {}   train_loss: {:.5f} wa_loss: {:.5f} edge_loss: {:.5f} val_loss: {:.5f}  val_mean_f1_score {:.2f},val_mean_acc {:.2f}'
        #         .format(epoch + 1, train_loss, wa_loss, edge_loss, val_loss, 100.* val_mean_f1_score, 100.* val_mean_acc)}
        # logging.info(infostr)
        # infostr = {'F1-score-list:'}
        # logging.info(infostr)
        # infostr = dataset_info(val_f1_score)
        # logging.info(infostr)
        # infostr = {'Acc-list:'}
        # logging.info(infostr)
        # infostr = dataset_info(val_acc)
        # logging.info(infostr)

        infostr = {'Epoch:  {}   train_loss: {:.5f}  val_loss: {:.5f} val_mae: {:.5f} val_mse: {:.5f}'
                   .format(epoch + 1, train_loss, val_loss, val_mae, val_mse)}
        logging.info(infostr)

        # save checkpoints
        if (best_val_loss > val_loss):
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(
                conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'))
            
        # if (epoch+1) % 4 == 0:
        #     checkpoint = {
        #         'epoch': epoch,
        #         'state_dict': net.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }
        #     torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold + 1) + '.pth'))

        # checkpoint = {
        #     'epoch': epoch,
        #     'state_dict': net.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }
        # torch.save(checkpoint, os.path.join(conf['outdir'], 'cur_model_fold' + str(conf.fold + 1) + '.pth'))


# ---------------------------------------------------------------------------------


if __name__=="__main__":
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

