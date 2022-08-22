import json
import tempfile
import torch
import numpy as np
from dataset.coco import build_coco

try:
    from pycocotools.cocoeval import COCOeval
except:
    print("It seems that the COCOAPI is not installed.")


class COCOAPIEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, data_dir, device, testset=False, transform=None):
        """
        Args:
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """
        # build coco dataset
        self.dataset = build_coco(
            root=data_dir,
            transform=transform,
            is_train=False,
            return_masks=False,
            testset=testset
            )
        self.transform = transform
        self.device = device
        self.testset = testset

        self.map = 0.
        self.ap50_95 = 0.
        self.ap50 = 0.


    def evaluate(self, model):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        model.eval()
        ids = []
        data_dict = []
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))

        # start testing
        for index in range(num_images): # all the data in val2017
            if index % 500 == 0:
                print('[Eval: %d / %d]'%(index, num_images))

            # load an image
            image, id = self.dataset.pull_image(index)

            orig_h = image.height
            orig_w = image.width

            # prepare
            x = self.transform(image)[0]
            x = x.unsqueeze(0).to(self.device)
            
            ids.append(int(id))

            # inference
            with torch.no_grad():
                outputs = model(x)
                bboxes, scores, labels = outputs
                # rescale
                bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]] * orig_w, a_min=0., a_max=orig_w)
                bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]] * orig_h, a_min=0., a_max=orig_h)

            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                coco_cls_id = self.dataset.class_ids.index(int(labels[i]))
                label = self.dataset.class_ids[coco_cls_id]
                
                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[i]) # object score * class score
                A = {"image_id": int(id), "category_id": label, "bbox": bbox,
                     "score": score} # COCO json format

                data_dict.append(A)

        annType = ['segm', 'bbox', 'keypoints']

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            if self.testset:
                json.dump(data_dict, open('coco_test-dev.json', 'w'))
                cocoDt = cocoGt.loadRes('coco_test-dev.json')
                return -1, -1
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, 'w'))
                cocoDt = cocoGt.loadRes(tmp)
                cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
                cocoEval.params.imgIds = ids
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()

                ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]
                print('ap50_95 : ', ap50_95)
                print('ap50 : ', ap50)
                self.map = ap50_95
                self.ap50_95 = ap50_95
                self.ap50 = ap50

                return ap50, ap50_95
        else:
            return 0, 0

