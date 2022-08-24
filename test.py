import argparse
import cv2
import os
import time
import numpy as np
import torch

from dataset.coco import build_coco, coco_class_labels
from dataset.transforms import build_transform
from utils.misc import load_weight

from config import build_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='DETR Library')

    # basic
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('--vis_thresh', default=0.5, type=float,
                        help='visualize threshold')
    parser.add_argument('--aux_loss', action='store_true', default=False, 
                        help='use intermediate output.')
    parser.add_argument('--use_nms', action='store_true', default=False, 
                        help='use NMS.')

    # model
    parser.add_argument('-v', '--version', default='detr_r50', type=str,
                        help='build DETR')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')

    return parser.parse_args()



def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, 
              bboxes, 
              scores, 
              cls_inds, 
              vis_thresh, 
              class_colors, 
              class_names):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(cls_inds[i])
            cls_color = class_colors[cls_id]
                
            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img
        

def test(args,
         model, 
         device, 
         dataset,
         transform,
         class_colors=None, 
         class_names=None, 
         show=False):
    num_images = len(dataset)
    save_path = os.path.join('det_results/', args.dataset, args.version)
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        image, _ = dataset.pull_image(index)

        orig_h = image.height
        orig_w = image.width

        # prepare
        x = transform(image)[0]
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # inference
        bboxes, scores, cls_inds = model(x)
        print("detection time used ", time.time() - t0, "s")
        
        # rescale
        bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]] * orig_w, a_min=0., a_max=orig_w)
        bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]] * orig_h, a_min=0., a_max=orig_h)

        # visulize results
        image = np.array(image)[..., (2, 1, 0)].astype(np.uint8)
        image = image.copy()
        img_processed = visualize(
                            img=image,
                            bboxes=bboxes,
                            scores=scores,
                            cls_inds=cls_inds,
                            vis_thresh=args.vis_thresh,
                            class_colors=class_colors,
                            class_names=class_names
                            )
        if show:
            cv2.imshow('detection', img_processed)
            cv2.waitKey(0)
        # save result
        cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)


if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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

    # dataset
    if args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        num_classes = 91
        class_names = coco_class_labels
        # dataset
        dataset = build_coco(
            root=data_dir,
            transform=None,
            is_train=False,
            return_masks=False
            )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

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

    # run
    test(args=args,
         model=model, 
         device=device, 
         dataset=dataset,
         transform=transform,
         class_colors=class_colors,
         class_names=class_names,
         show=args.show)
