import argparse
import cv2
import os
import time
import numpy as np
from PIL import Image
import torch

from dataset.coco import coco_class_labels
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
    parser.add_argument('-vs', '--vis_thresh', default=0.5, type=float,
                        help='visualize threshold')
    parser.add_argument('--aux_loss', action='store_true', default=False, 
                        help='use intermediate output.')
    parser.add_argument('--use_nms', action='store_true', default=False, 
                        help='use NMS.')

    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')

    parser.add_argument('--path_to_img', default='dataset/demo/images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='dataset/demo/videos/',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_save', default='det_results/demos/',
                        type=str, help='The path to save the detection results')

    # model
    parser.add_argument('-v', '--version', default='detr_r50', type=str,
                        help='build DETR')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
    
    return parser.parse_args()
                    

def plot_bbox_labels(img, bbox, label, cls_color, test_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    # plot title bbox
    cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * test_scale), y1), cls_color, -1)
    # put the test on the title bbox
    cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, test_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, bboxes, scores, cls_inds, class_colors, vis_thresh=0.3):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(cls_inds[i])
            cls_color = class_colors[cls_id]
            mess = '%s: %.2f' % (coco_class_labels[cls_id], scores[i])
            img = plot_bbox_labels(img, bbox, mess, cls_color, test_scale=ts)

    return img


def detect(args,
           model, 
           device, 
           transform, 
           mode='image', 
           path_to_img=None, 
           path_to_vid=None, 
           path_to_save=None):
    # class color
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(91)]
    save_path = os.path.join(path_to_save, mode)
    os.makedirs(save_path, exist_ok=True)

    # ------------------------- Camera ----------------------------
    if mode == 'camera':
        print('use camera !!!')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if ret:
                if cv2.waitKey(1) == ord('q'):
                    break

                # cv2 -> PIL
                frame_pil = Image.fromarray(frame.astype(np.uint8))

                orig_h = frame_pil.height
                orig_w = frame_pil.width

                # prepare
                x = transform(frame_pil)[0]
                x = x.unsqueeze(0).to(device)
                
                # inference
                t0 = time.time()
                bboxes, scores, cls_inds = model(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")

                # rescale
                bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]] * orig_w, a_min=0., a_max=orig_w)
                bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]] * orig_h, a_min=0., a_max=orig_h)

                # visulize results
                frame_processed = visualize(img=frame, 
                                            bboxes=bboxes,
                                            scores=scores, 
                                            cls_inds=cls_inds,
                                            class_colors=class_colors,
                                            vis_thresh=args.vis_thresh)

                cv2.imshow('detection result', frame_processed)
                cv2.waitKey(1)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    # ------------------------- Image ----------------------------
    elif mode == 'image':
        for i, img_id in enumerate(os.listdir(path_to_img)):
            image = cv2.imread(path_to_img + '/' + img_id, cv2.IMREAD_COLOR)

            # cv2 -> PIL
            image_pil = Image.fromarray(image.astype(np.uint8))

            orig_h = image_pil.height
            orig_w = image_pil.width

            # prepare
            x = transform(image_pil)[0]
            x = x.unsqueeze(0).to(device)

            # inference
            t0 = time.time()
            bboxes, scores, cls_inds = model(x)
            t1 = time.time()
            print("detection time used ", t1-t0, "s")

            # rescale
            bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]] * orig_w, a_min=0., a_max=orig_w)
            bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]] * orig_h, a_min=0., a_max=orig_h)

            # visulize results
            img_processed = visualize(img=image, 
                                      bboxes=bboxes,
                                      scores=scores, 
                                      cls_inds=cls_inds,
                                      class_colors=class_colors,
                                      vis_thresh=args.vis_thresh)
            
            if args.show:
                cv2.imshow('detection', img_processed)
                cv2.waitKey(0)

            # save results
            cv2.imwrite(os.path.join(save_path, str(i).zfill(6)+'.jpg'), img_processed)

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        video = cv2.VideoCapture(path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        save_path = os.path.join(save_path, cur_time+'.avi')
        fps = 30.0
        out = cv2.VideoWriter(save_path, fourcc, fps, save_size)
        print(save_path)

        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                # cv2 -> PIL
                frame_pil = Image.fromarray(frame.astype(np.uint8))

                orig_h = frame_pil.height
                orig_w = frame_pil.width

                # prepare
                x = transform(frame_pil)[0]
                x = x.unsqueeze(0).to(device)

                # inference
                t0 = time.time()
                bboxes, scores, cls_inds = model(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")

                # rescale
                bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]] * orig_w, a_min=0., a_max=orig_w)
                bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]] * orig_h, a_min=0., a_max=orig_h)

                # visualize results
                frame_processed = visualize(img=frame, 
                                            bboxes=bboxes,
                                            scores=scores, 
                                            cls_inds=cls_inds,
                                            class_colors=class_colors,
                                            vis_thresh=args.vis_thresh)

                if args.show:
                    cv2.imshow('detection', frame_processed)
                    cv2.waitKey(1)

                # save results
                frame_processed_resize = cv2.resize(frame_processed, save_size)
                out.write(frame_processed_resize)
            else:
                break
        video.release()
        out.release()
        cv2.destroyAllWindows()


def run():
    np.random.seed(0)

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

    # build model
    model, _ = build_model(
        args=args, 
        cfg=cfg,
        device=device, 
        num_classes=91, 
        trainable=False
        )

    # load trained weight
    model = load_weight(model, args.weight)
    model.to(device).eval()

    # run
    detect(
        args=args,
        model=model, 
        device=device,
        transform=transform,
        mode=args.mode,
        path_to_img=args.path_to_img,
        path_to_vid=args.path_to_vid,
        path_to_save=args.path_to_save
        )


if __name__ == '__main__':
    run()
