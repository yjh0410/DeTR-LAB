from __future__ import division

import os
import time
import argparse
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import distributed_utils
from utils.vis_tools import vis_data
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import CollateFunc, build_dataset, build_dataloader, get_total_grad_norm

from utils.solver.optimizer import build_optimizer
from utils.solver.lr_scheduler import build_lr_scheduler
from utils.solver.warmup_schedule import build_warmup

from config import build_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='DeTR')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--grad_clip_norm', type=float, default=0.1,
                        help='grad clip.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='path to save weight')
    parser.add_argument('--eval_epoch', default=10, type=int, 
                        help='after eval epoch, the model is evaluated on val dataset.')
    parser.add_argument('--fp16', dest="fp16", action="store_true", default=False,
                        help="Adopting mix precision training.")
    parser.add_argument('--vis', dest="vis", action="store_true", default=False,
                        help="visualize input data.")

    # model
    parser.add_argument('-v', '--version', default='detr_r50', type=str,
                        help='build DeTR')
    parser.add_argument('--aux_loss', action='store_true', default=False,
                        help="Use auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--use_nms', action='store_true', default=False,
                        help="Use NMS")
    parser.add_argument('-p', '--pretrained', default=None, type=str,
                        help='use coco pretrained')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')
    
    # train trick
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # dist
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    cfg = build_config(args)

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(cfg, args, device)

    # dataloader
    dataloader = build_dataloader(args, dataset, cfg['batch_size'], CollateFunc())

    # build model
    model, criterion = build_model(
        args=args, 
        cfg=cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=True,
        pretrained=args.pretrained,
        resume=args.resume
        )

    # set train mode
    model.to(device).train()
    criterion.to(device).train()

    # SyncBatchNorm
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # compute FLOPs and Params
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model_without_ddp)
        FLOPs_and_Params(model=model_copy, 
                         min_size=cfg['test_min_size'], 
                         max_size=cfg['test_max_size'], 
                         device=device)
        del model_copy
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()

    # optimizer
    base_lr = cfg['base_lr'] * cfg['batch_size'] * distributed_utils.get_world_size()
    backbone_lr = base_lr * cfg['bk_lr_ratio']
    optimizer, start_epoch = build_optimizer(cfg, model_without_ddp, base_lr, backbone_lr, args.resume)
    
    # lr scheduler
    lr_scheduler = build_lr_scheduler(cfg, optimizer, args.resume)

    # warmup scheduler
    warmup_scheduler = build_warmup(cfg, base_lr)

    # training configuration
    max_epoch = cfg['max_epoch']
    epoch_size = len(dataloader)
    best_map = -1.
    warmup = not args.no_warmup

    t0 = time.time()
    # start training loop
    for epoch in range(start_epoch, max_epoch):
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)            

        # train one epoch
        for iter_i, (images, targets, masks) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            # warmup
            if ni < cfg['wp_iter'] and warmup:
                warmup_scheduler.warmup(ni, optimizer)

            elif ni == cfg['wp_iter'] and warmup:
                # warmup is over
                print('Warmup is over')
                warmup = False
                warmup_scheduler.set_lr(optimizer, base_lr, base_lr)

            # visualize input data
            if args.vis:
                vis_data(images, targets, masks)
                continue

            # to device
            images = images.to(device)
            masks = masks.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # inference
            outputs = model(images, mask=masks)

            # loss
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce            
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            # check loss
            if torch.isnan(losses):
                print('loss is NAN !!')
                continue

            # Backward and Optimize
            optimizer.zero_grad()
            losses.backward()
            if args.grad_clip_norm > 0.:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            else:
                total_norm = get_total_grad_norm(model.parameters())
            optimizer.step()

            # display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                cur_lr_dict = {'lr': cur_lr[0], 'lr_bk': cur_lr[1]}
                # basic infor
                log =  '[Epoch: {}/{}]'.format(epoch+1, max_epoch)
                log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
                log += '[lr: {:.6f}][lr_bk: {:.6f}]'.format(cur_lr_dict['lr'], cur_lr_dict['lr_bk'])
                # loss infor
                for k in loss_dict_reduced_scaled.keys():
                    log += '[{}: {:.2f}]'.format(k, loss_dict_reduced_scaled[k])

                # other infor
                log += '[time: {:.2f}]'.format(t1 - t0)
                log += '[gnorm: {:.2f}]'.format(total_norm)
                log += '[size: [{}, {}]]'.format(cfg['train_min_size'], cfg['train_max_size'])

                # print log infor
                print(log, flush=True)
                
                t0 = time.time()

        lr_scheduler.step()
        
        # evaluation
        if (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == max_epoch:
            # check evaluator
            if distributed_utils.is_main_process():
                if evaluator is None:
                    print('No evaluator ... save model and go on training.')
                    print('Saving state, epoch: {}'.format(epoch + 1))
                    weight_name = '{}_epoch_{}.pth'.format(args.version, epoch + 1)
                    checkpoint_path = os.path.join(path_to_save, weight_name)
                    torch.save({'model': model_without_ddp.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'args': args}, 
                                checkpoint_path)                      
                    
                else:
                    print('eval ...')
                    # set eval mode
                    model_without_ddp.trainable = False
                    model_without_ddp.eval()

                    # evaluate
                    evaluator.evaluate(model_without_ddp)

                    cur_map = evaluator.map
                    if cur_map > best_map:
                        # update best-map
                        best_map = cur_map
                        # save model
                        print('Saving state, epoch:', epoch + 1)
                        weight_name = '{}_epoch_{}_{:.2f}.pth'.format(args.version, epoch + 1, best_map*100)
                        checkpoint_path = os.path.join(path_to_save, weight_name)
                        torch.save({'model': model_without_ddp.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'lr_scheduler': lr_scheduler.state_dict(),
                                    'epoch': epoch,
                                    'args': args}, 
                                    checkpoint_path)                      

                    # set train mode.
                    model_without_ddp.trainable = True
                    model_without_ddp.train()
        
            if args.distributed:
                # wait for all processes to synchronize
                dist.barrier()


if __name__ == '__main__':
    train()
