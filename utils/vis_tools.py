import numpy as np
import cv2
import torch

def vis_data(images, targets, masks):
    """
        images: (tensor) [B, 3, H, W]
        targets: (list) a list of targets
        masks: (tensor) [B, H, W]
    """
    batch_size = images.size(0)
    # vis data
    rgb_mean=[0.485, 0.456, 0.406]
    rgb_std=[0.229, 0.224, 0.225]

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(91)]

    for bi in range(batch_size):
        # mask
        mask = masks[bi]
        image_tensor = images[bi]
        index = torch.nonzero(~mask)

        # pad image
        # to numpy
        pad_image = image_tensor.permute(1, 2, 0).cpu().numpy()
        # denormalize
        pad_image = ((pad_image * rgb_std + rgb_mean)*255).astype(np.uint8)
        # to BGR
        pad_image = pad_image[..., (2, 1, 0)]

        # valid image without pad
        valid_image = image_tensor[:, :index[-1, 0]+1, :index[-1, 1]+1]
        valid_image = valid_image.permute(1, 2, 0).cpu().numpy()
        valid_image = ((valid_image * rgb_std + rgb_mean)*255).astype(np.uint8)
        valid_image = valid_image[..., (2, 1, 0)]

        valid_image = valid_image.copy()
        img_h, img_w = valid_image.shape[:2]

        targets_i = targets[bi]
        tgt_boxes = targets_i['boxes']
        tgt_labels = targets_i['labels']

        # to numpy
        mask = mask.cpu().numpy() * 255
        mask = mask.astype(np.uint8)

        
        for box, label in zip(tgt_boxes, tgt_labels):
            cx, cy, w, h = box
            x1 = int((cx - w * 0.5) * img_w)
            y1 = int((cy - h * 0.5) * img_h)
            x2 = int((cx + w * 0.5) * img_w)
            y2 = int((cy + h * 0.5) * img_h)

            cls_id = int(label)
            color = class_colors[cls_id]

            valid_image = cv2.rectangle(valid_image, (x1, y1), (x2, y2), color, 2)

        cv2.imshow('pad image', pad_image)
        cv2.waitKey(0)

        cv2.imshow('valid image', valid_image)
        cv2.waitKey(0)

        cv2.imshow('mask', mask)
        cv2.waitKey(0)

