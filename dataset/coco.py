# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

try:
    from .transforms import build_transform
except:
    from transforms import build_transform


coco_class_labels = ('background',
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.class_ids = sorted(self.coco.getCatIds())

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)


    def pull_image(self, idx):
        id = self.ids[idx]
        image = super(CocoDetection, self)._load_image(id)

        return image, id


    def pull_anno(self, idx):
        id = self.ids[idx]
        target = super(CocoDetection, self)._load_target(id)

        return target, id


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks


    def convert_coco_poly_to_mask(self, segmentations, height, width):
        masks = []
        for polygons in segmentations:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
            masks.append(mask)
        if masks:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        return masks


    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = self.convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


# build coco dataset
def build_coco(root, transform, is_train=False, return_masks=False, testset=False):
    mode = 'instances'
    PATHS = {
        "train": ("{}/train2017".format(root), "{}/annotations/{}_train2017.json".format(root, mode)),
        "val": ("{}/val2017".format(root), "{}/annotations/{}_val2017.json".format(root, mode)),
        "test": ("{}/test2017".format(root), "{}/annotations/image_info_test-dev2017.json".format(root)),
    }

    # image set
    if is_train:
        img_folder, ann_file = PATHS["train"]
    else:
        if testset:
            img_folder, ann_file = PATHS["test"]
        else:
            img_folder, ann_file = PATHS["val"]

    # dataset
    dataset = CocoDetection(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=transform,
        return_masks=return_masks
        )

    return dataset


if __name__ == '__main__':
    import numpy as np
    import cv2

    # transform
    min_size = 800
    max_size = 1333
    random_size = [600, 700, 800]
    pixel_mean = [0.485, 0.456, 0.406]
    pixel_std = [0.229, 0.224, 0.225]
    is_train = True
    transform = build_transform(
        is_train=is_train,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        min_size=min_size,
        max_size=max_size,
        random_size=random_size
    )

    # dataset
    dataset = build_coco(
        root='/mnt/share/ssd2/dataset/COCO',
        transform=transform,
        is_train=is_train,
        return_masks=False
        )
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(91)]
    print('Data length: ', len(dataset))

    for i in range(1000):
        image, target = dataset[i]
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        image = (image * pixel_std + pixel_mean) * 255.
        image = image.astype(np.uint8)
        # to BGR
        image = image[..., (2, 1, 0)]

        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            cx, cy, w, h = box
            x1 = int((cx - w * 0.5) * img_w)
            y1 = int((cy - h * 0.5) * img_h)
            x2 = int((cx + w * 0.5) * img_w)
            y2 = int((cy + h * 0.5) * img_h)

            cls_id = int(label)
            color = class_colors[cls_id]

            # class name
            label = coco_class_labels[cls_id]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # put the test on the bbox
            cv2.putText(image, label, (x1, y1 - 5), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)
