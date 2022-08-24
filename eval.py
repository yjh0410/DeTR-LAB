import argparse
import os

import torch

from evaluator.coco_evaluator import COCOAPIEvaluator
from dataset.transforms import build_transform
from utils.misc import load_weight

from config import build_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='DETR Library')

    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')

    # model
    parser.add_argument('-v', '--version', default='detr_r50', type=str,
                        help='build DETR')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--aux_loss', action='store_true', default=False, 
                        help='use intermediate output.')
    parser.add_argument('--use_nms', action='store_true', default=False, 
                        help='use NMS.')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')

    return parser.parse_args()



def coco_test(model, data_dir, device, transform, test=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        testset=True,
                        transform=transform)

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        testset=False,
                        transform=transform)

    # COCO evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    if args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 91
        data_dir = os.path.join(args.root, 'COCO')
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 91
        data_dir = os.path.join(args.root, 'COCO')
    else:
        print('unknow dataset !! we only support voc, coco-val, coco-test !!!')
        exit(0)


    # config
    cfg = build_config(args)

    # transform
    transform = build_transform(
        is_train=False, 
        pixel_mean=cfg['pixel_mean'],
        pixel_std=cfg['pixel_std'],
        min_size=cfg['test_min_size'],
        max_size=cfg['test_max_size'],
        random_size=None
    )

    # build model
    model, _ = build_model(
        args=args, 
        cfg=cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False
        )

    # load trained weight
    model = load_weight(model, args.weight)
    model.to(device).eval()

    # evaluation
    with torch.no_grad():
        if args.dataset == 'coco-val':
            coco_test(model, data_dir, device, transform, test=False)
        elif args.dataset == 'coco-test':
            coco_test(model, data_dir, device, transform, test=True)
